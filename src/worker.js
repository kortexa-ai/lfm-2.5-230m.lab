import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
} from "@huggingface/transformers";

// LFM2.5-230M — the smallest LFM2.5, small enough to run comfortably in-browser.
// LFM2.5-230M-ONNX ships fp16 / q4 / q4f32 / q8 weights (no q4f16), so we default
// to q4f32: 4-bit weights with fp32 compute — the widest-compatible WebGPU option.
const MODEL_ID = "LiquidAI/LFM2.5-230M-ONNX";
const DTYPE = "q4f32";

let tokenizer = null;
let model = null;

// Check whether the model files are already in the browser's Cache Storage,
// so the UI can say "cached" instead of implying a fresh multi-hundred-MB download.
async function isModelCached() {
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    return keys.some(
      (req) =>
        req.url.includes(MODEL_ID.replace("/", "%2F")) ||
        req.url.includes(MODEL_ID)
    );
  } catch {
    return false;
  }
}

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === "check") {
    self.postMessage({ type: "cache_status", data: { cached: await isModelCached() } });
    return;
  }

  if (type === "load") {
    try {
      self.postMessage({ type: "status", data: "Loading tokenizer…" });
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: (p) => self.postMessage({ type: "progress", data: p }),
      });

      self.postMessage({ type: "status", data: "Loading model onto the GPU (WebGPU)…" });
      model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
        device: "webgpu",
        dtype: DTYPE,
        progress_callback: (p) => self.postMessage({ type: "progress", data: p }),
      });

      self.postMessage({ type: "loaded" });
    } catch (err) {
      self.postMessage({ type: "error", data: String(err?.message || err) });
    }
    return;
  }

  if (type === "generate") {
    if (!tokenizer || !model) {
      self.postMessage({ type: "error", data: "Model is not loaded yet." });
      return;
    }
    try {
      const inputs = tokenizer.apply_chat_template(data.messages || [], {
        add_generation_prompt: true,
        return_dict: true,
      });

      self.postMessage({
        type: "generate_start",
        data: { promptTokens: inputs.input_ids.dims[1] },
      });

      const startTime = performance.now();
      let totalTokens = 0;

      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => {
          totalTokens++;
          const elapsed = (performance.now() - startTime) / 1000;
          self.postMessage({
            type: "token",
            data: { text, tokenCount: totalTokens, tokensPerSec: totalTokens / elapsed },
          });
        },
      });

      await model.generate({
        ...inputs,
        max_new_tokens: data.maxTokens || 1024,
        temperature: data.temperature ?? 0.7,
        do_sample: (data.temperature ?? 0.7) > 0,
        streamer,
      });

      const elapsed = (performance.now() - startTime) / 1000;
      self.postMessage({
        type: "generate_done",
        data: { tokenCount: totalTokens, elapsed, tokensPerSec: totalTokens / elapsed },
      });
    } catch (err) {
      self.postMessage({ type: "error", data: String(err?.message || err) });
    }
    return;
  }
};
