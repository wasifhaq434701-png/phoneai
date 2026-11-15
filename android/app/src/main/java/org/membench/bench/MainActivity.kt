package org.membench.bench

import android.app.Activity
import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import java.io.File

/** The whole app: one screen built in code (no layouts, no AndroidX) so memory goes to the model. */
class MainActivity : Activity() {
    private lateinit var conditions: TextView
    private lateinit var modelLabel: TextView
    private lateinit var log: TextView
    private lateinit var buttons: List<Button>
    private val budgets = listOf(0, 1000, 2000, 4000, 8000)
    private val minutes = (0..120 step 5).toList()
    private var busy = false
    private lateinit var threads: Spinner

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        fun text(size: Float = 14f) = TextView(this).apply { textSize = size }.also { col.addView(it) }
        fun button(label: String, onClick: () -> Unit) = Button(this).apply { text = label; setOnClickListener { onClick() } }
            .also { col.addView(it) }
        fun spinner(labels: List<String>, selected: Int) = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, labels)
            setSelection(selected)
        }.also { col.addView(it) }

        text(22f).text = "MemBench"
        conditions = text()
        modelLabel = text()
        val import = button("Import model…") { pickModel() }
        val items = spinner(listOf("Items: none (sustained only)", "Items: up to 1K", "Items: up to 2K", "Items: up to 4K",
            "Items: all (up to 8K)"), budgets.size - 1)
        val sustained = spinner(minutes.map { "Sustained run: $it min" }, minutes.indexOf(30))
        val repeats = spinner((1..5).map { "Repeats: $it" + if (it > 1) " (waits ≥15 min to cool down between runs)" else "" }, 2)
        val cores = Runtime.getRuntime().availableProcessors()
        val fast = Probe.fastCores()
        threads = spinner((1..cores).map { "Threads: $it" + if (it == fast) " (fast cores)" else "" }, fast - 1)
        val energy = CheckBox(this).apply { text = "Energy run (on battery, $ENERGY_START_MAX% → $ENERGY_STOP%, airplane mode)" }.also { col.addView(it) }
        val natural = CheckBox(this).apply { text = "Natural answers: stop at the end of the answer (max $ANSWER_MAX tokens), for the accuracy check" }
            .also { col.addView(it) }
        val firstQ = spinner((1..QUESTIONS).map { "Start at question $it" + if (it == 1) "" else " (continue a stopped run)" }, 0)
        val start = button("Start") {
            // an energy run is a single run: the battery window doesn't hold several
            val n = if (energy.isChecked) 1 else repeats.selectedItemPosition + 1
            start(energy.isChecked) { r, m ->
                r.benchSeries(m, budgets[items.selectedItemPosition], minutes[sustained.selectedItemPosition] * 60.0, energy.isChecked, n,
                    natural.isChecked, firstQ.selectedItemPosition + 1)
            }
        }
        // one pass per charge when the battery window can't hold all of them; each pass is its own results file
        val passChoices = listOf((1..GRID_PASSES).toList()) + (1..GRID_PASSES).map { listOf(it) }
        val passes = spinner(listOf("Grid passes: all $GRID_PASSES") + (1..GRID_PASSES).map { "Grid pass $it only (one charge each)" }, 0)
        val grid = button("Energy grid (4 prompts × 16/64/256 tokens)") {
            val p = passChoices[passes.selectedItemPosition]
            start(energy.isChecked) { r, m -> r.grid(m, energy.isChecked, p) }
        }
        // always an energy run: the idle baseline is only meaningful on battery, in the same window as the grid
        val idle = button("Idle baseline (10 min, on battery)") { start(true) { r, m -> r.idle(m, 10 * 60.0) } }
        val share = button("Share latest results") { share() }
        buttons = listOf(import, start, grid, idle, share)
        log = text(11f).apply { typeface = Typeface.MONOSPACE }

        val root = ScrollView(this).apply { addView(col); setPadding(pad, pad, pad, pad) }
        // Android 15+ draws under the system bars: add their size to the padding.
        root.setOnApplyWindowInsetsListener { v, insets ->
            val b = insets.getInsets(WindowInsets.Type.systemBars())
            v.setPadding(b.left + pad, b.top + pad, b.right + pad, b.bottom + pad)
            insets
        }
        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    private fun refresh() {
        conditions.text = Probe(this).summary()
        val m = Runner.model(this)
        modelLabel.text = m?.let { "Model: ${it.name} (${gb(it.length())})" }
            ?: "No model: tap Import model, or adb push a .gguf to ${getExternalFilesDir(null)}/models/"
        buttons.forEach { it.isEnabled = !busy }
    }

    private fun say(line: String) = runOnUiThread { log.text = line + "\n" + log.text }

    /** Runs `job` on a background thread with the screen kept on at a fixed brightness. */
    private fun start(energyRun: Boolean, job: (Runner, File) -> Unit) {
        refresh()  // battery and charger may have changed since the screen was drawn
        val model = Runner.model(this) ?: return say("No model yet: import one first.")
        val runner = Runner(this, threads.selectedItemPosition + 1, ::say)
        if (energyRun) runner.energyGuard()?.let { return say(it) }
        log.text = ""
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.attributes = window.attributes.apply { screenBrightness = BRIGHTNESS }
        inBackground {
            try {
                job(runner, model)
            } finally {
                runOnUiThread {
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    window.attributes = window.attributes.apply { screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE }
                }
            }
        }
    }

    /** Disables the buttons while `work` runs on a background thread. */
    private fun inBackground(work: () -> Unit) {
        busy = true
        refresh()
        Thread { try { work() } finally { runOnUiThread { busy = false; refresh() } } }.start()
    }

    private fun pickModel() = startActivityForResult(
        Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), PICK_MODEL)

    @Deprecated("Activity result API needs AndroidX; this app has no dependencies")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri = data?.data ?: return
        if (requestCode == PICK_MODEL && resultCode == RESULT_OK) importModel(uri)
    }

    /** Copies the picked GGUF into the app's private storage, hashing it on the way. */
    private fun importModel(uri: Uri) {
        val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else null
        } ?: "model.gguf"
        if (!name.endsWith(".gguf")) return say("Not a .gguf file: $name")
        val dir = File(filesDir, "models").apply { mkdirs() }
        val dest = File(dir, name.replace(Regex("[^A-Za-z0-9._-]"), "_"))
        inBackground {
            say("importing $name …")
            dir.listFiles()?.forEach { it.delete() }  // one model at a time: storage is the limit
            try {
                val part = File(dest.path + ".part")
                val sha = contentResolver.openInputStream(uri)!!.use { Runner.sha256(it, part) }
                part.renameTo(dest)
                File(dest.path + ".sha256").writeText(sha)
                say("imported ${dest.name} (${gb(dest.length())}), sha256 $sha")
            } catch (e: Exception) {
                dir.listFiles()?.forEach { it.delete() }
                say("import FAILED: $e")
            }
        }
    }

    private fun share() {
        val f = getExternalFilesDir(null)?.listFiles { f -> f.name.startsWith("bench-") && f.name.endsWith(".json") }
            ?.maxByOrNull { it.lastModified() } ?: return say("No results yet.")
        val uri = Uri.parse("content://$packageName.results/${f.name}")
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("application/json")
            .putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), f.name))
    }

    companion object {
        const val PICK_MODEL = 1
        const val QUESTIONS = 20  // LongMemEval dev questions in prompts.json
    }
}
