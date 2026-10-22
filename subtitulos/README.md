# Subtítulos en vivo (Whisper local)

Subtítulos en español en tiempo real sobre la cámara, al estilo de los subtítulos en vivo de OBS.
La transcripción usa Whisper dentro del navegador (Transformers.js 4.3.1 + WebGPU). **El audio nunca sale del equipo.**

Pensada para macOS con Apple Silicon (M1 o superior) y Chrome.

## Instalación (una sola vez, requiere internet)

```bash
cd subtitulos
./download-models.sh             # whisper-base + whisper-small (~900 MB)
# o bien
./download-models.sh --solo-base # solo whisper-base (~250 MB)
```

El script descarga en `lib/` Transformers.js y los binarios WASM de onnxruntime-web, y en `models/` los pesos ONNX de `onnx-community/whisper-base` y `whisper-small`. Si lo vuelves a ejecutar, salta los archivos que ya están descargados.

## Uso (offline)

```bash
cd subtitulos
python3 -m http.server 8000
```

Abre **http://localhost:8000** en Chrome, pulsa **Iniciar** y acepta los permisos de cámara y micrófono.
La primera carga tarda unos segundos porque compila los shaders en la GPU.

Mueve el mouse para ver el panel de controles: cámara, micrófono, modelo, estilo, tamaño, sensibilidad del VAD (con medidor de nivel), espejo y pausa. Las preferencias se guardan en el navegador.

| Tecla | Acción |
|---|---|
| F | Pantalla completa |
| M | Espejo |
| Espacio | Pausa |
| C | Limpiar subtítulos |
| + / − | Tamaño de texto |
| H | Mostrar/ocultar panel |

**Indicador de estado** (punto en la esquina superior izquierda): gris = cargando o en pausa, verde = escuchando, azul = transcribiendo, rojo = error.

La consola (⌥⌘J) muestra la latencia de cada inferencia, el RTF (tiempo de inferencia ÷ duración del audio) y los textos que descartó el filtro anti-alucinaciones.

## Cómo funciona

- `audio-processor.js` (AudioWorklet): pasa el audio a mono, lo remuestrea a 16 kHz y envía bloques de 32 ms con su energía RMS.
- `index.html`: el VAD por energía RMS abre un segmento cuando detecta voz. Mientras la persona habla, transcribe el segmento cada ~1 s (texto **interino**, opacidad 0,6). Tras 700 ms de silencio lo transcribe por última vez (texto **final**). Fuerza el cierre de los segmentos que superan 15 s. Si el worker está ocupado, descarta los interinos y solo encola los finales.
- `worker.js`: Whisper en WebGPU (encoder fp32 + decoder q4). Si no hay WebGPU, cae automáticamente a WASM (encoder q8 + decoder q4).

## Rendimiento medido (M-series, Chrome)

| Modelo | Backend | Latencia por inferencia |
|---|---|---|
| whisper-base | WebGPU | ~0,5–0,85 s |
| whisper-small | WebGPU | ~1,5–1,7 s |
| whisper-base | WASM | ~3,5 s |

## Limitaciones conocidas

- **No es streaming real.** Whisper procesa ventanas de 30 s, así que cada transcripción interina vuelve a procesar todo el segmento. El texto interino puede cambiar hasta que se vuelve final.
- **Alucinaciones.** El filtro descarta las frases típicas («Amara.org», «Gracias por ver el video», «Suscríbete»…), las anotaciones como [Música] y las repeticiones del segmento anterior, pero no lo atrapa todo. Si un «Gracias» real dura menos de ~0,5 s, puede descartarse.
- **VAD por energía.** No usa Silero. Con ruido de fondo constante (ventiladores, música), sube o baja la «Sensibilidad VAD» guiándote por el medidor: la marca blanca es el umbral.
- **El fallback WASM es lento y usa un solo hilo**, porque `python3 -m http.server` no envía las cabeceras COOP/COEP que hacen falta para usar hilos.
- whisper-small consume más memoria de GPU y tarda más. Si no está descargado, la app vuelve a whisper-base sola.
- Hay que abrir la app desde `http://localhost`, no desde `file://`, porque el navegador exige un contexto seguro para usar la cámara y el micrófono.
- Probado en Chrome. Safari y Firefox tienen un soporte de WebGPU distinto y no están probados.
