#!/usr/bin/env bash
# =============================================================================
#  download-models.sh — descarga UNA sola vez la librería y los modelos.
#  Después de esto la app funciona 100 % offline.
#
#  Uso:
#    ./download-models.sh             # whisper-base + whisper-small (~900 MB)
#    ./download-models.sh --solo-base # solo whisper-base (~250 MB)
#
#  Es re-ejecutable: los archivos ya descargados se saltan.
# =============================================================================
set -euo pipefail

TJS_VERSION="4.3.1"   # @huggingface/transformers (estable más reciente verificada)
HF="https://huggingface.co"
NPM="https://registry.npmjs.org"
JSDELIVR="https://cdn.jsdelivr.net/npm"

cd "$(dirname "$0")"
ROOT="$(pwd)"
LIB="$ROOT/lib"
MODELS="$ROOT/models"

MODEL_IDS=("onnx-community/whisper-base" "onnx-community/whisper-small")
if [[ "${1:-}" == "--solo-base" ]]; then
  MODEL_IDS=("onnx-community/whisper-base")
fi

# Archivos de configuración / tokenizer de cada modelo
CONFIG_FILES=(
  config.json generation_config.json preprocessor_config.json
  tokenizer.json tokenizer_config.json vocab.json merges.txt
  added_tokens.json normalizer.json special_tokens_map.json
)
# Pesos ONNX:
#   WebGPU -> encoder fp32 + decoder q4   (configuración probada para Whisper en WebGPU)
#   WASM   -> encoder q8   + decoder q4   (fallback)
ONNX_FILES=(
  onnx/encoder_model.onnx
  onnx/encoder_model_quantized.onnx
  onnx/decoder_model_merged_q4.onnx
)

# --- utilidades --------------------------------------------------------------
command -v curl >/dev/null || { echo "ERROR: se necesita curl"; exit 1; }
command -v tar  >/dev/null || { echo "ERROR: se necesita tar";  exit 1; }

# fetch URL DEST : descarga a DEST.part y renombra al terminar (salta si existe)
fetch() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "  ✓ ya existe  $(basename "$dest")"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  ↓ $(basename "$dest")"
  curl -fL --retry 3 --retry-delay 2 -C - --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

# --- 1. Transformers.js + runtime ONNX (WASM/WebGPU) -------------------------
echo "==> Transformers.js $TJS_VERSION"
mkdir -p "$LIB"
if [[ ! -s "$LIB/transformers.min.js" ]]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  curl -fsSL "$NPM/@huggingface/transformers/-/transformers-$TJS_VERSION.tgz" | tar xz -C "$TMP"
  cp "$TMP/package/dist/transformers.min.js" "$LIB/"
  # Versión exacta de onnxruntime-web contra la que se compiló esta release
  ORT_VERSION="$(sed -n 's/.*"onnxruntime-web": *"\([^"]*\)".*/\1/p' "$TMP/package/package.json" | head -1)"
  echo "$ORT_VERSION" > "$LIB/ORT_VERSION"
else
  echo "  ✓ ya existe  transformers.min.js"
fi
ORT_VERSION="$(cat "$LIB/ORT_VERSION")"

echo "==> onnxruntime-web $ORT_VERSION (binarios WASM, sin CDN en runtime)"
for f in ort-wasm-simd-threaded.asyncify.mjs ort-wasm-simd-threaded.asyncify.wasm; do
  fetch "$JSDELIVR/onnxruntime-web@$ORT_VERSION/dist/$f" "$LIB/$f"
done

# --- 2. Modelos Whisper multilingües ------------------------------------------
for id in "${MODEL_IDS[@]}"; do
  echo "==> $id"
  for f in "${CONFIG_FILES[@]}" "${ONNX_FILES[@]}"; do
    fetch "$HF/$id/resolve/main/$f" "$MODELS/$id/$f"
  done
done

echo
echo "Listo. Levanta la app con:"
echo "  cd \"$ROOT\" && python3 -m http.server 8000"
echo "y abre http://localhost:8000 en Chrome."
