package org.membench.bench

import android.content.ContentProvider
import android.content.ContentValues
import android.database.MatrixCursor
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import java.io.File

/** Read-only access to results JSON files for the share sheet (FileProvider needs AndroidX). Not exported:
 *  other apps only reach the one file a share grants them. */
class ResultsProvider : ContentProvider() {
    private fun file(uri: Uri): File {
        val name = uri.lastPathSegment ?: ""
        require(Regex("bench-[A-Za-z0-9._,-]+\\.json").matches(name)) { "not a results file" }
        return File(context!!.getExternalFilesDir(null), name)
    }

    override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor =
        ParcelFileDescriptor.open(file(uri), ParcelFileDescriptor.MODE_READ_ONLY)

    override fun query(uri: Uri, projection: Array<String>?, selection: String?, args: Array<String>?, sort: String?) =
        MatrixCursor(arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE)).apply {
            file(uri).let { addRow(arrayOf(it.name, it.length())) }
        }

    override fun getType(uri: Uri) = "application/json"
    override fun onCreate() = true
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, selection: String?, args: Array<String>?) = 0
    override fun update(uri: Uri, values: ContentValues?, selection: String?, args: Array<String>?) = 0
}
