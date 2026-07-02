# LFM2.5-230M WebGPU

Chat with [Liquid AI](https://www.liquid.ai/)'s **LFM2.5-230M** language model running
**entirely in the browser** — ONNX Runtime on WebGPU, no server and no API. The model
weights download once from Hugging Face, cache in the browser, and every token after
that is generated on your own GPU. Nothing you type leaves the page.

A [kortexa.ai lab](https://lab.kortexa.ai) experiment.

## How it works

- [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) loads
  [`LiquidAI/LFM2.5-230M-ONNX`](https://huggingface.co/LiquidAI/LFM2.5-230M-ONNX) with
  `device: "webgpu"`.
- Quantization is selectable on the load screen — each variant caches separately:
  - **q4** (`model_q4.onnx`, ~200 MB) — smallest & fastest, the default, best on phones.
  - **q8** (~470 MB) — higher quality.
  - **fp16** (~450 MB) — best quality; coincidentally the same size as q8, because a small
    model's embedding / lm_head dominate and stay high-precision.
- Inference runs in a Web Worker so the UI stays responsive; tokens stream in via
  `TextStreamer`. The bigger variants use proportionally more GPU memory and decode
  slower (memory-bandwidth-bound) — q4 is the right pick for memory-constrained devices.

## Develop

```bash
npm install
npm run dev      # https://localhost:8040/lfm-2-5-230m/
```

WebGPU is required — use a recent Chrome/Edge (or a WebGPU-enabled browser) on a machine
with a GPU.

## Build & deploy

```bash
npm run build    # -> dist/
```

Served in production at **https://lab.kortexa.ai/lfm-2-5-230m/** via the hub's dynamic
nginx discovery: this repo's `nginx/lfm-2-5-230m.lab` fragment is auto-included by
`lab.kortexa.ai`, and the hub lists it from `lab/public/labs.json`.
