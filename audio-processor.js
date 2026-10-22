// =============================================================================
//  audio-processor.js — AudioWorklet de captura y remuestreo.
//
//  Recibe audio del micrófono a la frecuencia del AudioContext (normalmente
//  48 kHz), lo mezcla a mono, lo remuestrea a 16 kHz (lo que espera Whisper) y
//  envía al hilo principal bloques de FRAME_SIZE muestras Float32 junto con
//  su energía RMS (para el VAD). El buffer se transfiere, no se copia.
// =============================================================================

const TARGET_RATE = 16000;
const FRAME_SIZE = 512; // 32 ms a 16 kHz

class ResamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // sampleRate es global dentro de AudioWorkletGlobalScope
    this.ratio = sampleRate / TARGET_RATE; // p. ej. 3.0 para 48 kHz
    this.acc = 0;      // suma de muestras de entrada en la ventana actual
    this.accN = 0;     // nº de muestras acumuladas
    this.pos = 0;      // posición fraccionaria dentro de la ventana
    this.frame = new Float32Array(FRAME_SIZE);
    this.frameIdx = 0;
  }

  // Remuestreo por promedio de ventana (filtro "box"): actúa a la vez como
  // filtro anti-aliasing simple y decimador. Suficiente para voz.
  pushSample(x) {
    this.acc += x;
    this.accN++;
    this.pos += 1;
    if (this.pos >= this.ratio) {
      this.pos -= this.ratio;
      this.emit(this.acc / this.accN);
      this.acc = 0;
      this.accN = 0;
    }
  }

  emit(y) {
    this.frame[this.frameIdx++] = y;
    if (this.frameIdx === FRAME_SIZE) {
      let sum = 0;
      for (let i = 0; i < FRAME_SIZE; i++) sum += this.frame[i] * this.frame[i];
      const rms = Math.sqrt(sum / FRAME_SIZE);
      const buf = this.frame;
      this.port.postMessage({ samples: buf, rms }, [buf.buffer]);
      this.frame = new Float32Array(FRAME_SIZE);
      this.frameIdx = 0;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channels = input.length;
    const n = input[0].length;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let c = 0; c < channels; c++) s += input[c][i];
      this.pushSample(s / channels);
    }
    return true; // mantener vivo el procesador
  }
}

registerProcessor('resampler-processor', ResamplerProcessor);
