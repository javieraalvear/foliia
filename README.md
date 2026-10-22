# Subtítulos en vivo (Whisper local)

Subtítulos en español en tiempo real sobre la cámara, al estilo de los subtítulos en vivo de OBS.
Hay dos motores de transcripción, que se eligen en la pantalla inicial o en el panel:

- **Local** (por defecto): Whisper dentro del navegador (Transformers.js 4.3.1 + WebGPU). Funciona sin internet y **el audio nunca sale del equipo**.
- **OpenAI Realtime**: transcripción en la nube con `gpt-live-transcribe`. Transcribe mejor el español, pero requiere internet y una clave de API de pago, y el audio con voz se envía a OpenAI.

Pensada para macOS con Apple Silicon. **Navegador recomendado: Safari 26 o superior**, que trae WebGPU activado de serie. También funciona en Firefox y Chrome.

## Instalación (una sola vez, requiere internet)

```bash
cd ~/Documents/UDD/folia/foliia
./download-models.sh             # whisper-base + whisper-small (~560 MB)
# o bien
./download-models.sh --solo-base # solo whisper-base (~125 MB)
```

El script descarga en `lib/` Transformers.js y los binarios WASM de onnxruntime-web, y en `models/` los pesos ONNX de `onnx-community/whisper-base` y `whisper-small`. Si lo vuelves a ejecutar, salta los archivos que ya están descargados.

## Uso (offline)

**1. Levanta el servidor local.** Abre la Terminal (⌘ + Espacio, escribe «Terminal» y pulsa Enter) y pega esto:

```bash
cd ~/Documents/UDD/folia/foliia && python3 serve.py
```

Verás el mensaje `Sirviendo en http://localhost:8000 (Ctrl+C para salir)`. Deja esa ventana abierta: el servidor funciona mientras la Terminal siga abierta.

**2. Abre Safari** y entra en **http://localhost:8000**. Pulsa **Iniciar** y acepta el permiso de cámara y micrófono.

**Para apagarlo:** vuelve a la Terminal y pulsa **Ctrl + C**.

Si aparece un error como `Address already in use`, ya hay algo usando el puerto 8000. Usa otro puerto:

```bash
python3 serve.py 8080
```

En ese caso, abre http://localhost:8080 en Safari.

Notas:

- **Usa `serve.py` y no `python3 -m http.server`.** `serve.py` envía las cabeceras COOP/COEP que permiten usar WebAssembly multihilo, y eso hace el decoder ~25 % más rápido. Con `http.server` la app también funciona, pero con un solo hilo.
- Para que Safari no pida permiso en cada sesión, ve a Safari → Ajustes → Sitios web → Cámara / Micrófono → `localhost` → *Permitir*.
- Para ver la consola con las latencias, activa Safari → Ajustes → Avanzado → «Mostrar funciones para desarrolladores web» y pulsa ⌥⌘C.

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

## Motor OpenAI Realtime (opcional)

1. Crea una clave de API en https://platform.openai.com/settings/organization/api-keys (necesita crédito o un método de pago).
2. Abre el archivo `.env` de esta carpeta y pega la clave:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. Reinicia el servidor (Ctrl + C y `python3 serve.py`). Debe decir `OpenAI Realtime: clave cargada desde .env`.
4. En la app, elige **OpenAI Realtime** en la pantalla inicial o en el panel (selector «Motor»).

**Seguridad:** la clave solo la lee `serve.py`. El navegador recibe tokens temporales de 10 minutos, `serve.py` no entrega `.env` aunque se pida por URL, y `.env` está en `.gitignore`. No compartas ese archivo.

**Costo:** $0.017 por minuto de audio enviado, unos US$1 por hora de habla continua. La app solo envía los tramos con voz que detecta el VAD, así que **el silencio no se paga**. El panel muestra los minutos enviados y el costo estimado de la sesión.

Opciones en `.env`:

| Variable | Por defecto | Qué hace |
|---|---|---|
| `OPENAI_REALTIME_MODEL` | `gpt-live-transcribe` | Modelo de transcripción en vivo |
| `OPENAI_REALTIME_LANGUAGES` | `es` | Idiomas esperados (ISO 639-1, separados por coma) |
| `OPENAI_REALTIME_DELAY` | `low` | Latencia vs. precisión: `minimal`, `low`, `medium`, `high`, `xhigh` |
| `OPENAI_NOISE_REDUCTION` | (vacío) | `near_field` (auriculares) o `far_field` (micrófono del portátil) |

## Cómo funciona

- `audio-processor.js` (AudioWorklet): pasa el audio a mono, lo remuestrea a 16 kHz y envía bloques de 32 ms con su energía RMS.
- `index.html`: el VAD por energía RMS abre un segmento cuando detecta voz. Mientras la persona habla, transcribe el segmento cada ~1 s (texto **interino**, opacidad 0,6). Tras 700 ms de silencio lo transcribe por última vez (texto **final**). Fuerza el cierre de los segmentos que superan 15 s. Si el worker está ocupado, descarta los interinos y solo encola los finales.
- `realtime.js` (motor OpenAI): pide un token temporal a `serve.py` y abre un WebSocket con OpenAI. Remuestrea el audio a 24 kHz PCM16 y lo envía solo mientras hay voz; al final de cada frase envía `commit`. Los fragmentos de texto que llegan mientras se habla se muestran como interinos, y la transcripción completa de cada frase como final.
- `worker.js`: el **encoder** de Whisper corre en **WebGPU (fp16)**. Es un único pase pesado sobre una ventana de 30 s, ideal para la GPU. El **decoder** corre en **WASM q8 multihilo** (CPU). Genera un token por paso, y en la GPU cada paso arrastra un costo fijo que domina el tiempo total. Si no hay WebGPU, todo corre en WASM.

## Rendimiento medido (Apple Silicon, frase de 5,8 s)

Comparación de configuraciones en el motor de Safari (WebKit):

| whisper-base | Latencia |
|---|---|
| Todo en WebGPU (encoder fp32 + decoder q4) | 1910 ms |
| Todo en WebGPU (fp16 + fp16) | 1392 ms |
| **Encoder WebGPU fp16 + decoder WASM q8, 4 hilos (elegida)** | **879 ms** |
| La misma, sin multihilo (`http.server`) | 1191 ms |

En la app, con la configuración elegida:

| Modelo | Navegador | Latencia por inferencia |
|---|---|---|
| whisper-base | Safari (WebKit) | ~0,45–1,3 s |
| whisper-base | Firefox 157 | ~0,6–1,6 s |
| whisper-small | Safari (WebKit) | ~2–3 s |

## Limitaciones conocidas

- **El motor OpenAI depende de internet.** Si se corta la conexión, los subtítulos se detienen; la app intenta reconectar con la siguiente frase, esperando entre 2 y 30 s entre intentos. Para usarlo, la app debe servirse con `serve.py`.
- **El motor OpenAI se probó con una conexión simulada**, no contra la API real, porque no había clave disponible. La configuración sigue la documentación oficial vigente (octubre de 2026).
- **El motor local no es streaming real.** Whisper procesa ventanas de 30 s, así que cada transcripción interina vuelve a procesar todo el segmento. El texto interino puede cambiar hasta que se vuelve final.
- **whisper-small es lento para usarlo en vivo**: tarda 2–3 s por inferencia y muchos interinos se descartan. Conviene usarlo solo si la calidad importa más que la latencia.
- **Alucinaciones.** El filtro descarta las frases típicas («Amara.org», «Gracias por ver el video», «Suscríbete»…), las anotaciones como [Música] y las repeticiones del segmento anterior, pero no lo atrapa todo. Si un «Gracias» real dura menos de ~0,5 s, puede descartarse.
- **VAD por energía.** No usa Silero. Con ruido de fondo constante, ajusta la «Sensibilidad VAD» guiándote por el medidor: la marca blanca es el umbral.
- Sin WebGPU (o en una GPU sin `shader-f16`), todo corre en CPU y la latencia sube a ~3–4 s.
- Hay que abrir la app desde `http://localhost`, no desde `file://`, porque el navegador exige un contexto seguro para usar la cámara y el micrófono.
- Las pruebas automáticas usaron el motor WebKit de Playwright (equivalente a Safari 27) y Firefox 157 real, con voz sintetizada. Falta probarla con un micrófono y una cámara reales.
