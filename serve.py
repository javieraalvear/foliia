#!/usr/bin/env python3
"""Servidor local de la app de subtítulos.

- Sirve los archivos con aislamiento de origen (COOP/COEP) para habilitar
  WebAssembly multihilo (modo local).
- Lee OPENAI_API_KEY desde .env y entrega al navegador tokens temporales de la
  Realtime API (modo OpenAI). La clave real nunca sale de este proceso.

Uso: python3 serve.py [puerto]
Solo usa la librería estándar de Python.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Configuración (.env + variables de entorno, que tienen prioridad)
# -----------------------------------------------------------------------------
def load_env(path):
    env = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


ENV = {**load_env(ROOT / '.env'), **os.environ}
API_KEY = ENV.get('OPENAI_API_KEY', '')
HAS_KEY = API_KEY.startswith('sk-')


def realtime_session_config():
    """Sesión de transcripción en vivo (ver docs: guides/realtime-transcription)."""
    transcription = {
        'model': ENV.get('OPENAI_REALTIME_MODEL') or 'gpt-live-transcribe',
        'languages': [c.strip() for c in (ENV.get('OPENAI_REALTIME_LANGUAGES') or 'es').split(',') if c.strip()],
        'delay': ENV.get('OPENAI_REALTIME_DELAY') or 'low',
    }
    noise = ENV.get('OPENAI_NOISE_REDUCTION') or ''
    return {
        'type': 'transcription',
        'audio': {
            'input': {
                'format': {'type': 'audio/pcm', 'rate': 24000},
                'transcription': transcription,
                'turn_detection': None,  # el fin de frase lo decide el VAD del navegador
                'noise_reduction': {'type': noise} if noise in ('near_field', 'far_field') else None,
            }
        },
    }


def create_client_secret():
    """Pide a OpenAI un token temporal (10 min) ligado a la sesión de transcripción."""
    session = realtime_session_config()
    body = json.dumps({
        'expires_after': {'anchor': 'created_at', 'seconds': 600},
        'session': session,
    }).encode()
    req = urllib.request.Request(
        'https://api.openai.com/v1/realtime/client_secrets',
        data=body, method='POST',
        headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    return {'value': data['value'], 'expires_at': data.get('expires_at'), 'session': session}


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------
class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      '.js': 'text/javascript', '.mjs': 'text/javascript',
                      '.wasm': 'application/wasm', '.onnx': 'application/octet-stream'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # No servir secretos ni archivos ocultos (.env, .git…), tampoco codificados (%2e)
        path = urllib.parse.unquote(self.path.split('?')[0])
        if any(part.startswith('.') for part in path.split('/')):
            self.send_error(404)
            return
        if self.path == '/api/config':
            self.send_json(200, {'realtime': HAS_KEY})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != '/api/realtime-token':
            self.send_error(404)
            return
        # Solo peticiones de la propia app (no de otras páginas abiertas en el navegador)
        if self.headers.get('Sec-Fetch-Site', 'same-origin') != 'same-origin':
            self.send_json(403, {'error': 'Origen no permitido'})
            return
        if not HAS_KEY:
            self.send_json(503, {'error': 'Falta OPENAI_API_KEY en el archivo .env'})
            return
        try:
            self.send_json(200, create_client_secret())
        except urllib.error.HTTPError as e:
            try:
                msg = json.load(e).get('error', {}).get('message', '')
            except Exception:
                msg = ''
            self.send_json(e.code, {'error': f'OpenAI respondió {e.code}: {msg or e.reason}'})
        except Exception as e:  # sin internet, timeout, etc.
            self.send_json(502, {'error': f'No se pudo contactar a OpenAI: {e}'})

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f'Sirviendo en http://localhost:{port}  (Ctrl+C para salir)', flush=True)
    print('OpenAI Realtime: ' + ('clave cargada desde .env' if HAS_KEY else 'desactivado (sin OPENAI_API_KEY en .env)'), flush=True)
    try:
        ThreadingHTTPServer(('localhost', port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
