import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// llama.cpp b11836 source goes in ../third_party/llama.cpp (not in git); prompts come from ../prompts/prompts.json.
// Release signing: set BENCH_KEYSTORE to a keystore.properties (storeFile, alias, password); unset = unsigned release APK.
val repo = rootDir.resolve("..")
val keystore = File(System.getenv("BENCH_KEYSTORE") ?: "/nonexistent")

android {
    namespace = "org.membench.bench"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "org.membench.bench"
        minSdk = 31  // Build.SOC_MODEL
        targetSdk = 36
        versionCode = 2
        versionName = "1.1"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                // Same CPU flags as experiments/android_bench.sh, so app and adb numbers are comparable.
                val llama = repo.resolve("third_party/llama.cpp").canonicalPath
                // Strip local paths from __FILE__ strings (llama.cpp asserts) so the APK holds no builder's paths.
                val flags = "-march=armv8.2-a+dotprod+i8mm -ffile-prefix-map=$llama=llama.cpp -ffile-prefix-map=${repo.canonicalPath}=."
                arguments += listOf("-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_FLAGS=$flags", "-DCMAKE_CXX_FLAGS=$flags", "-DLLAMA_DIR=$llama")
            }
        }
    }

    signingConfigs {
        if (keystore.exists()) create("release") {
            val p = Properties().apply { keystore.inputStream().use { load(it) } }
            storeFile = keystore.resolveSibling(p.getProperty("storeFile"))
            storePassword = p.getProperty("password")
            keyAlias = p.getProperty("alias")
            keyPassword = p.getProperty("password")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
            signingConfig = signingConfigs.findByName("release")
        }
    }

    externalNativeBuild { cmake { path("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("prompts").get().asFile)
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

// prompts.json goes into the APK unchanged, so its SHA-256 matches the Mac and iPhone runs.
val copyPrompts = tasks.register<Copy>("copyPrompts") {
    from(repo.resolve("prompts/prompts.json"))
    into(layout.buildDirectory.dir("prompts"))
}
tasks.named("preBuild") { dependsOn(copyPrompts) }

dependencies { testImplementation("junit:junit:4.13.2") }
