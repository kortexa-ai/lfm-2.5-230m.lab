import { useEffect, useRef, useState, useCallback } from "react";

const MODEL_LABEL = "LiquidAI LFM2.5-230M";

// Quantizations LFM2.5-230M-ONNX actually ships, with real download/GPU-memory sizes.
// q8 (~470MB) is coincidentally the same size as fp16 because the embedding / lm_head
// of a small model dominate and stay high-precision — so fp16 is the better "quality" step.
const DTYPES = [
  { id: "q4", label: "q4", sub: "4-bit", size: "~200 MB", note: "smallest & fastest — best on phones" },
  { id: "q8", label: "q8", sub: "8-bit", size: "~470 MB", note: "higher quality" },
  { id: "fp16", label: "fp16", sub: "16-bit", size: "~450 MB", note: "best quality, same size as q8" },
];
const dtypeInfo = (id) => DTYPES.find((d) => d.id === id) || DTYPES[0];

export default function App() {
  const [webgpu, setWebgpu] = useState(true);
  const [dtype, setDtype] = useState("q4"); // selected on the load screen
  const [activeDtype, setActiveDtype] = useState(null); // what's actually loaded
  const [modelReady, setModelReady] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState({}); // file -> percent
  const [cached, setCached] = useState(false);
  const [error, setError] = useState(null);

  const [messages, setMessages] = useState([]); // {role, content}
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [stats, setStats] = useState(null); // {tokensPerSec, tokenCount}

  const workerRef = useRef(null);
  const scrollRef = useRef(null);
  const requestedDtypeRef = useRef("q4");

  // (Re)create the inference worker and wire up its messages.
  const spawnWorker = useCallback(() => {
    workerRef.current?.terminate();
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    workerRef.current = worker;

    worker.onmessage = (e) => {
      const { type, data } = e.data;
      if (type === "cache_status") setCached(data.cached);
      else if (type === "status") setStatus(data);
      else if (type === "progress") {
        if (data.status === "progress" && data.file) {
          setProgress((prev) => ({ ...prev, [data.file]: Math.round(data.progress || 0) }));
        }
      } else if (type === "loaded") {
        setActiveDtype(requestedDtypeRef.current);
        setModelReady(true);
        setLoadingModel(false);
        setStatus("");
        setProgress({});
      } else if (type === "generate_start") {
        setMessages((m) => [...m, { role: "assistant", content: "" }]);
      } else if (type === "token") {
        setMessages((m) => {
          const next = m.slice();
          const last = next[next.length - 1];
          if (last && last.role === "assistant") next[next.length - 1] = { ...last, content: last.content + data.text };
          return next;
        });
        setStats({ tokensPerSec: data.tokensPerSec, tokenCount: data.tokenCount });
      } else if (type === "generate_done") {
        setGenerating(false);
        setStats({ tokensPerSec: data.tokensPerSec, tokenCount: data.tokenCount });
      } else if (type === "error") {
        setError(data);
        setGenerating(false);
        setLoadingModel(false);
      }
    };

    worker.postMessage({ type: "check" });
    return worker;
  }, []);

  useEffect(() => {
    if (typeof navigator !== "undefined" && !navigator.gpu) {
      setWebgpu(false);
      return;
    }
    spawnWorker();
    return () => workerRef.current?.terminate();
  }, [spawnWorker]);

  // Auto-scroll to the latest message.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const loadModel = useCallback(() => {
    setError(null);
    setLoadingModel(true);
    setStatus("Starting…");
    requestedDtypeRef.current = dtype;
    workerRef.current?.postMessage({ type: "load", data: { dtype } });
  }, [dtype]);

  // Unload and return to the picker so a different quant can be loaded.
  const switchQuant = useCallback(() => {
    setModelReady(false);
    setActiveDtype(null);
    setMessages([]);
    setStats(null);
    setError(null);
    setInput("");
    spawnWorker(); // fresh worker (re-checks cache for the current selection)
  }, [spawnWorker]);

  const send = useCallback(() => {
    const text = input.trim();
    if (!text || generating || !modelReady) return;
    const convo = [...messages, { role: "user", content: text }];
    setMessages(convo);
    setInput("");
    setGenerating(true);
    setStats(null);
    setError(null);
    workerRef.current?.postMessage({
      type: "generate",
      data: { messages: convo, temperature: 0.7, maxTokens: 1024 },
    });
  }, [input, generating, modelReady, messages]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const overallProgress = (() => {
    const vals = Object.values(progress);
    if (!vals.length) return null;
    return Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
  })();

  const sel = dtypeInfo(dtype);

  return (
    <main className="page">
      <header className="hero">
        <div className="brand">
          <span className="dot" />
          <p className="eyebrow">kortexa.ai lab</p>
        </div>
        <h1>{MODEL_LABEL}</h1>
        <p className="lede">
          A 230M-parameter language model running <strong>entirely in your browser</strong> —
          ONNX Runtime on WebGPU. No server, no API. Your GPU does the thinking, and nothing
          you type leaves this page.
        </p>
      </header>

      {!webgpu ? (
        <section className="panel notice">
          <h2>WebGPU not available</h2>
          <p>
            This demo needs WebGPU. Try a recent Chrome, Edge, or a WebGPU-enabled browser on a
            machine with a GPU. On Safari, enable the WebGPU feature flag.
          </p>
        </section>
      ) : !modelReady ? (
        <section className="panel loader">
          <h2>Choose a quantization</h2>
          <p className="muted">
            Smaller = faster download and less GPU memory. Bigger = better quality. The weights
            cache in this browser after the first load of each variant.
          </p>

          <div className="quant-options" role="radiogroup" aria-label="Quantization">
            {DTYPES.map((d) => (
              <button
                key={d.id}
                type="button"
                role="radio"
                aria-checked={dtype === d.id}
                className={`quant ${dtype === d.id ? "selected" : ""}`}
                onClick={() => setDtype(d.id)}
                disabled={loadingModel}
              >
                <span className="quant-head">
                  <span className="quant-label">{d.label}</span>
                  <span className="quant-sub">{d.sub}</span>
                </span>
                <span className="quant-size">{d.size}</span>
                <span className="quant-note">{d.note}</span>
              </button>
            ))}
          </div>

          {overallProgress !== null ? (
            <div className="progress">
              <div className="bar" style={{ width: `${overallProgress}%` }} />
              <span className="pct">{overallProgress}%</span>
            </div>
          ) : null}
          {status ? <p className="status">{status}</p> : null}

          <button className="primary" onClick={loadModel} disabled={loadingModel}>
            {loadingModel ? "Loading…" : `Download & run ${sel.label} (${sel.size})`}
          </button>
          {error ? <p className="error">{error}</p> : null}
        </section>
      ) : (
        <section className="chat">
          <div className="messages" ref={scrollRef}>
            {messages.length === 0 ? (
              <p className="empty">Say hello 👋 — {MODEL_LABEL} ({activeDtype}) is running on your GPU.</p>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={`msg ${m.role}`}>
                  <div className="who">{m.role === "user" ? "You" : MODEL_LABEL}</div>
                  <div className="bubble">
                    {m.content || (generating && i === messages.length - 1 ? <span className="caret" /> : "")}
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={generating ? "Generating…" : "Message LFM2.5-230M…  (Enter to send)"}
              rows={1}
              disabled={generating}
            />
            <button className="primary" onClick={send} disabled={generating || !input.trim()}>
              Send
            </button>
          </div>
          <div className="footer">
            <span>
              {stats
                ? `${stats.tokenCount} tokens · ${stats.tokensPerSec.toFixed(1)} tok/s`
                : "running locally · WebGPU"}
              {" · "}
              <button className="linkish" onClick={switchQuant} disabled={generating}>
                {activeDtype} ⇄ switch quant
              </button>
            </span>
            {error ? <span className="error"> · {error}</span> : null}
          </div>
        </section>
      )}
    </main>
  );
}
