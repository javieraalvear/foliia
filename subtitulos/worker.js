// =============================================================================
//  worker.js — carga del modelo Whisper e inferencia, fuera del hilo principal.
//
//  Protocolo (postMessage):
//    main -> worker  { type: 'load', model: 'base' | 'small' }
//                    { type: 'transcribe', id, segId, kind: 'interim'|'final', audio: Float32Array }
//    worker -> main  { type: 'progress', progress (0-100), file }
//                    { type: 'warming' }
//                    { type: 'ready', device, model, loadMs }
//                    { type: 'result', id, segId, kind, text, ms, audioSec }
//                    { type: 'error', code, message, id? }
// =============================================================================

import { pipeline, env } from './lib/transformers.min.js';

// -----------------------------------------------------------------------------
// 1. Configuración offline: todo se carga desde disco local
// -----------------------------------------------------------------------------
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = './models/';
env.useBrowserCache = false; // los archivos ya están en disco; no duplicarlos en Cache Storage

// Binarios de onnxruntime-web servidos desde /lib (por defecto irían a un CDN)
env.backends.onnx.wasm.wasmPaths = {
  mjs: new URL('./lib/ort-wasm-simd-threaded.asyncify.mjs', self.location.href).href,
  wasm: new URL('./lib/ort-wasm-simd-threaded.asyncify.wasm', self.location.href).href,
};

const MODELS = {
  base: 'onnx-community/whisper-base',
  small: 'onnx-community/whisper-small',
};

// dtype por archivo ONNX según backend (ver download-models.sh)
const DTYPES = {
  webgpu: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  wasm: { encoder_model: 'q8', decoder_model_merged: 'q4' },
};

let transcriber = null;
let currentDevice = null;

// -----------------------------------------------------------------------------
// 2. Detección de WebGPU
// -----------------------------------------------------------------------------
async function detectDevice() {
  if (!('gpu' in navigator)) {
    console.warn('[worker] WebGPU no está disponible en este navegador → usando WASM (más lento).');
    return 'wasm';
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('requestAdapter() devolvió null');
    return 'webgpu';
  } catch (err) {
    console.warn('[worker] WebGPU no disponible (' + err.message + ') → usando WASM (más lento).');
    return 'wasm';
  }
}

// -----------------------------------------------------------------------------
// 3. Carga del modelo (con fallback WebGPU → WASM)
// -----------------------------------------------------------------------------
function progressCallback(info) {
  if (info.status === 'progress_total') {
    self.postMessage({ type: 'progress', progress: info.progress, file: info.name });
  }
}

async function createPipeline(modelId, device) {
  return pipeline('automatic-speech-recognition', modelId, {
    device,
    dtype: DTYPES[device],
    progress_callback: progressCallback,
  });
}

function isNotFound(err) {
  const m = String(err?.message || err);
  return /not found locally|could not locate file|404|failed to fetch/i.test(m);
}

async function load(modelKey) {
  const modelId = MODELS[modelKey] || MODELS.base;
  const t0 = performance.now();

  // Liberar el modelo anterior si se cambia de base <-> small
  if (transcriber) {
    try { await transcriber.dispose(); } catch { /* noop */ }
    transcriber = null;
  }

  let device = await detectDevice();
  try {
    transcriber = await createPipeline(modelId, device);
  } catch (err) {
    if (isNotFound(err)) throw Object.assign(err, { code: 'MODEL_NOT_FOUND' });
    if (device !== 'webgpu') throw err;
    console.warn('[worker] Falló la carga en WebGPU, reintentando en WASM:', err);
    device = 'wasm';
    transcriber = await createPipeline(modelId, device);
  }
  currentDevice = device;

  // Calentamiento: la primera ejecución compila shaders en la GPU (varios segundos)
  self.postMessage({ type: 'warming' });
  await runWhisper(new Float32Array(16000));

  const loadMs = Math.round(performance.now() - t0);
  console.log(`[worker] ${modelId} listo en ${device} (${loadMs} ms)`);
  self.postMessage({ type: 'ready', device, model: modelKey, loadMs });
}

// -----------------------------------------------------------------------------
// 4. Inferencia
// -----------------------------------------------------------------------------
async function runWhisper(audio) {
  const out = await transcriber(audio, {
    language: 'spanish',
    task: 'transcribe',
    return_timestamps: false,
    max_new_tokens: 128,       // un segmento de ≤15 s no necesita más
    no_repeat_ngram_size: 4,   // reduce bucles de repetición típicos de Whisper
  });
  return (Array.isArray(out) ? out[0]?.text : out?.text) || '';
}

async function transcribe({ id, segId, kind, audio }) {
  if (!transcriber) {
    self.postMessage({ type: 'error', code: 'NOT_READY', message: 'Modelo no cargado', id });
    return;
  }
  const t0 = performance.now();
  const text = await runWhisper(audio);
  const ms = Math.round(performance.now() - t0);
  self.postMessage({
    type: 'result', id, segId, kind, text: text.trim(), ms,
    audioSec: +(audio.length / 16000).toFixed(2), device: currentDevice,
  });
}

// -----------------------------------------------------------------------------
// 5. Mensajes. Se procesa un mensaje a la vez; el hilo principal se encarga de
//    no encolar interinos (descarta si estamos ocupados).
// -----------------------------------------------------------------------------
let chain = Promise.resolve();
self.addEventListener('message', (e) => {
  const msg = e.data;
  chain = chain.then(async () => {
    try {
      if (msg.type === 'load') await load(msg.model);
      else if (msg.type === 'transcribe') await transcribe(msg);
    } catch (err) {
      console.error('[worker]', err);
      self.postMessage({
        type: 'error',
        id: msg.id,
        code: err.code || (isNotFound(err) ? 'MODEL_NOT_FOUND' : 'INFERENCE_ERROR'),
        message: String(err?.message || err),
      });
    }
  });
});
