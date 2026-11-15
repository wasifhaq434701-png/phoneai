// JNI bridge for the benchmark: load a GGUF once; each call = fresh context, prompt token ids in, greedy tokens out,
// with a CLOCK_BOOTTIME timestamp (= SystemClock.elapsedRealtimeNanos) before prefill and after every new token.
#include <jni.h>
#include <time.h>
#include <algorithm>
#include <string>
#include <vector>
#include "llama.h"

static const int kPrefillStep = 2048;  // like mlx-lm on the Mac and the iPhone app

static jlong now_ns() {
    timespec t{};
    clock_gettime(CLOCK_BOOTTIME, &t);
    return (jlong) t.tv_sec * 1000000000LL + t.tv_nsec;
}

static void fail(JNIEnv *env, const std::string &msg) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), msg.c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_membench_bench_Llama_load(JNIEnv *env, jobject, jstring path) {
    llama_backend_init();
    auto p = llama_model_default_params();
    p.load_mode = LLAMA_LOAD_MODE_NONE;  // read the weights into RAM like the iPhone app (no mmap)
    const char *s = env->GetStringUTFChars(path, nullptr);
    llama_model *m = llama_model_load_from_file(s, p);
    env->ReleaseStringUTFChars(path, s);
    if (!m) fail(env, "llama.cpp could not load the model");
    return (jlong) m;
}

extern "C" JNIEXPORT void JNICALL
Java_org_membench_bench_Llama_free(JNIEnv *, jobject, jlong model) {
    llama_model_free((llama_model *) model);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_membench_bench_Llama_systemInfo(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

// Returns the new token ids; `times` (size nNew + 1) gets [start, token 1, …]. With stopAtEos the answer ends at the
// first end-of-generation token (included in the result), otherwise EOS is ignored and exactly nNew tokens come back.
extern "C" JNIEXPORT jintArray JNICALL
Java_org_membench_bench_Llama_generate(JNIEnv *env, jobject, jlong handle, jintArray ids, jint nNew, jint threads,
                                            jlongArray times, jboolean stopAtEos) {
    auto *model = (llama_model *) handle;
    const int n = env->GetArrayLength(ids);
    std::vector<llama_token> prompt(n);
    env->GetIntArrayRegion(ids, 0, n, prompt.data());

    // KV sized to this call only (prompt + new tokens), as the iPhone app pre-sizes it.
    auto cp = llama_context_default_params();
    cp.n_ctx = n + nNew;
    cp.n_batch = cp.n_ubatch = kPrefillStep;
    cp.n_outputs_max = 1;  // logits for the last position only: no vocab-sized row per prompt token
    cp.n_threads = cp.n_threads_batch = threads;
    cp.no_perf = true;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fail(env, "llama_init_from_model failed (out of memory?)");
        return nullptr;
    }
    const llama_vocab *voc = llama_model_get_vocab(model);
    const int vocab = llama_vocab_n_tokens(voc);
    llama_batch b = llama_batch_init(kPrefillStep, 0, 1);
    auto push = [&](llama_token tok, int pos, bool logits) {
        b.token[b.n_tokens] = tok;
        b.pos[b.n_tokens] = pos;
        b.n_seq_id[b.n_tokens] = 1;
        b.seq_id[b.n_tokens][0] = 0;
        b.logits[b.n_tokens] = logits;
        b.n_tokens++;
    };

    std::vector<jlong> t{now_ns()};
    std::vector<jint> out;
    int pos = 0, rc = 0;
    for (int i = 0; i < n && rc == 0; i += kPrefillStep) {
        b.n_tokens = 0;
        for (int j = i; j < std::min(n, i + kPrefillStep); j++) push(prompt[j], pos++, j == n - 1);
        rc = llama_decode(ctx, b);
    }
    while (rc == 0) {
        const float *lg = llama_get_logits_ith(ctx, -1);  // waits for the computation
        out.push_back((jint) (std::max_element(lg, lg + vocab) - lg));
        t.push_back(now_ns());
        if ((int) out.size() == nNew || (stopAtEos && llama_vocab_is_eog(voc, out.back()))) break;
        b.n_tokens = 0;
        push(out.back(), pos++, true);
        rc = llama_decode(ctx, b);
    }
    llama_batch_free(b);
    llama_free(ctx);
    if (rc != 0) {
        fail(env, "llama_decode failed (" + std::to_string(rc) + ") after " + std::to_string(out.size()) + " tokens");
        return nullptr;
    }
    env->SetLongArrayRegion(times, 0, (jsize) t.size(), t.data());
    jintArray res = env->NewIntArray((jsize) out.size());
    env->SetIntArrayRegion(res, 0, (jsize) out.size(), out.data());
    return res;
}
