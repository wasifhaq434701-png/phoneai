import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// The runtime boundary. An Android (llama.cpp) or Core AI backend implements the same three calls.
protocol LLMEngine {
    func load(directory: URL) async throws
    /// Greedy generation from exact token ids with a fresh KV cache. Streams token ids to `onToken`;
    /// returning false stops early (EOS handling and the stop button are the caller's choice).
    func generate(promptIDs: [Int], maxTokens: Int, kvBits: Int?, onToken: (Int) -> Bool) throws
    func unload()
}

enum EngineError: Error { case notLoaded }

final class MLXEngine: LLMEngine {
    /// mlx-lm's default on the Mac (every paper number); mlx-swift-lm would default to 512.
    static let macPrefillStep = 2048
    var prefillStep = MLXEngine.macPrefillStep
    private var model: (any LanguageModel)?

    /// Model only, no tokenizer: the benchmark feeds token ids. Mirrors LLMModelFactory._load.
    func load(directory: URL) async throws {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        let model = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: base.modelType)
        try await loadWeights(modelDirectory: directory, model: model,
                              perLayerQuantization: base.perLayerQuantization)
        self.model = model
    }

    func generate(promptIDs: [Int], maxTokens: Int, kvBits: Int?, onToken: (Int) -> Bool) throws {
        guard let model else { throw EngineError.notLoaded }
        // MLX errors (e.g. Metal out of memory) throw instead of aborting the app, so the caller can record them.
        try withError { try autoreleasepool {
            let parameters = GenerateParameters(maxTokens: maxTokens, kvBits: kvBits, temperature: 0)
            var cache = try model.newCache(parameters: parameters)
            // One allocation of the final size: growing in 256-token steps concatenates, so old + new KV coexist
            // (≈2.4 GB of KV at once in the last 8K chunk on a 4.34 GB iPhone). Arithmetic is unchanged.
            for c in cache { (c as? KVCacheSimple)?.step = promptIDs.count + maxTokens }
            // mlx-lm's prefill: chunks of ≤ prefillStep with no logits computed, then only the last token goes
            // through the iterator. mlx-swift-lm's own prefill runs up to a whole step through the LM head
            // (logits for every position: +1.2 GB peak at 1K on the iPhone 12 Pro).
            var rest = promptIDs[...]
            while rest.count > 1 {
                let n = min(prefillStep, rest.count - 1)
                autoreleasepool {  // as mlx-swift-lm's prefill: drop each chunk's array wrappers (and buffers) now
                    _ = model(MLXArray(rest.prefix(n).map(Int32.init))[.newAxis], cache: cache)
                    maybeQuantizeKVCache(cache: &cache, kvBits: kvBits)  // from the first chunk, as in the Mac's kvb runs
                    eval(cache)
                }
                Memory.clearCache()
                rest = rest.dropFirst(n)
            }
            var tokens = try TokenIterator(
                input: LMInput(tokens: MLXArray(rest.map(Int32.init))), model: model, cache: cache, parameters: parameters)
            while let token = tokens.next(), onToken(token) {}  // the iterator never stops at EOS
        } }
    }

    func unload() {
        model = nil
        Memory.clearCache()
    }
}
