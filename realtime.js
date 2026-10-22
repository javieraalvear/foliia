// =============================================================================
//  realtime.js — cliente de transcripción en vivo con la Realtime API de OpenAI.
//
//  Flujo:
//    1. Pide a serve.py un token temporal (la clave real vive en .env, nunca
//       llega al navegador).
//    2. Abre un WebSocket directo con OpenAI usando ese token.
//    3. Recibe audio 16 kHz del pipeline local, lo remuestrea a 24 kHz PCM16
//       y lo envía SOLO mientras el VAD detecta voz: la transcripción se cobra
//       por duración de audio, así que no se paga el silencio.
//    4. Al final de cada frase envía input_audio_buffer.commit.
//    5. Emite deltas (texto interino) y transcripciones completas (final).
// =============================================================================

const WS_URL = 'wss://api.openai.com/v1/realtime?intent=transcription';
const PRICE_PER_MIN = 0.017; // USD, gpt-live-transcribe (developers.openai.com/api/docs/pricing)

// -----------------------------------------------------------------------------
// Remuestreo 16 kHz → 24 kHz (interpolación lineal, con estado entre bloques)
// -----------------------------------------------------------------------------
class Upsampler {
  constructor(inRate = 16000, outRate = 24000) {
    this.step = inRate / outRate; // 2/3
    this.prev = 0;
    this.pos = 1; // posición en [prev, ...bloque]; 1 = primera muestra del bloque
  }
  reset() { this.prev = 0; this.pos = 1; }
  process(input) {
    const n = input.length + 1;
    const out = new Int16Array(Math.ceil((n - this.pos) / this.step) + 1);
    let k = 0, p = this.pos;
    const at = (i) => (i === 0 ? this.prev : input[i - 1]);
    while (p < n - 1) {
      const i = Math.floor(p), f = p - i;
      const s = at(i) * (1 - f) + at(i + 1) * f;
      out[k++] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
      p += this.step;
    }
    this.pos = p - (n - 1);
    this.prev = input[input.length - 1];
    return out.subarray(0, k);
  }
}

function int16ToBase64(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

// -----------------------------------------------------------------------------
// Cliente
// -----------------------------------------------------------------------------
export class RealtimeTranscriber {
  /**
   * @param {object} cb callbacks:
   *   onDelta(itemId, delta), onCompleted(itemId, transcript),
   *   onStatus('connecting'|'open'|'closed'|'error'), onError(message)
   */
  constructor(cb) {
    this.cb = cb;
    this.ws = null;
    this.status = 'closed';
    this.queue = [];          // mensajes en espera mientras conecta
    this.upsampler = new Upsampler();
    this.secondsSent = 0;     // para estimar costo
    this.failures = 0;
    this.retryAt = 0;         // espera progresiva tras un fallo (no reintentar por cada bloque)
    this.connecting = null;
  }

  get open() { return this.ws?.readyState === WebSocket.OPEN; }
  get costUSD() { return (this.secondsSent / 60) * PRICE_PER_MIN; }

  setStatus(s) { this.status = s; this.cb.onStatus?.(s); }

  // Conecta (o reutiliza la conexión). Seguro de llamar varias veces.
  connect() {
    if (this.open) return Promise.resolve();
    if (this.connecting) return this.connecting;
    if (performance.now() < this.retryAt) return Promise.reject(new Error('esperando para reintentar'));
    this.connecting = this._connect()
      .catch((err) => {
        // 2, 4, 8… hasta 30 s. El audio en espera de ese intento se descarta.
        this.retryAt = performance.now() + Math.min(30000, 1000 * 2 ** this.failures);
        this.queue.length = 0;
        throw err;
      })
      .finally(() => { this.connecting = null; });
    return this.connecting;
  }

  async _connect() {
    this.setStatus('connecting');
    const t0 = performance.now();

    // 1. Token temporal desde serve.py
    let token;
    try {
      const r = await fetch('/api/realtime-token', { method: 'POST' });
      token = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(token.error || `HTTP ${r.status}`);
    } catch (err) {
      this.failures++;
      this.setStatus('error');
      this.cb.onError?.('No se pudo obtener el token de OpenAI: ' + err.message);
      throw err;
    }

    // 2. WebSocket directo con OpenAI (el token va como subprotocolo)
    await new Promise((resolve, reject) => {
      const ws = new WebSocket(WS_URL, ['realtime', 'openai-insecure-api-key.' + token.value]);
      this.ws = ws;
      let opened = false;

      ws.onopen = () => {
        opened = true;
        this.failures = 0;
        // La configuración ya viaja en el token; se reenvía por si acaso (idempotente)
        if (token.session) ws.send(JSON.stringify({ type: 'session.update', session: token.session }));
        for (const m of this.queue.splice(0)) ws.send(m);
        console.log(`[realtime] conectado en ${Math.round(performance.now() - t0)} ms ` +
                    `(${token.session?.audio?.input?.transcription?.model || 'modelo por defecto'})`);
        this.setStatus('open');
        resolve();
      };
      ws.onmessage = (e) => this._onEvent(JSON.parse(e.data));
      ws.onerror = () => {
        if (!opened) {
          this.failures++;
          this.setStatus('error');
          this.cb.onError?.('No se pudo conectar con OpenAI (¿sin internet?).');
          reject(new Error('ws error'));
        }
      };
      ws.onclose = (e) => {
        if (this.ws === ws) this.ws = null;
        console.log(`[realtime] conexión cerrada (${e.code}${e.reason ? ': ' + e.reason : ''})`);
        if (opened) this.setStatus('closed');   // se reconecta sola con la próxima frase
      };
    });
  }

  _onEvent(ev) {
    switch (ev.type) {
      case 'conversation.item.input_audio_transcription.delta':
        this.cb.onDelta?.(ev.item_id, ev.delta || '');
        break;
      case 'conversation.item.input_audio_transcription.completed':
        this.cb.onCompleted?.(ev.item_id, ev.transcript || '', ev.usage);
        break;
      case 'conversation.item.input_audio_transcription.failed':
        console.error('[realtime] transcripción fallida:', ev.error);
        this.cb.onCompleted?.(ev.item_id, '');
        break;
      case 'error':
        // Commit con buffer vacío no es grave (p. ej. segmento descartado)
        if (ev.error?.code === 'input_audio_buffer_commit_empty') break;
        console.error('[realtime] error:', ev.error);
        this.cb.onError?.('OpenAI: ' + (ev.error?.message || 'error desconocido'));
        break;
    }
  }

  _send(obj) {
    const msg = JSON.stringify(obj);
    if (this.open) { this.ws.send(msg); return; }
    this.queue.push(msg);
    if (this.queue.length > 500) this.queue.shift();   // ~15 s de audio como máximo
    this.connect().catch(() => {});
  }

  // Envía un bloque Float32 a 16 kHz
  append(samples16k) {
    const pcm = this.upsampler.process(samples16k);
    if (!pcm.length) return;
    this.secondsSent += samples16k.length / 16000;
    this._send({ type: 'input_audio_buffer.append', audio: int16ToBase64(pcm) });
  }

  // Fin de frase: pedir la transcripción final
  commit() {
    this._send({ type: 'input_audio_buffer.commit' });
    this.upsampler.reset();
  }

  // Descarta el audio de una "frase" que resultó ser ruido
  clear() {
    this._send({ type: 'input_audio_buffer.clear' });
    this.upsampler.reset();
  }

  close() {
    this.queue.length = 0;
    try { this.ws?.close(1000, 'cliente'); } catch { /* noop */ }
    this.ws = null;
    this.setStatus('closed');
  }
}
