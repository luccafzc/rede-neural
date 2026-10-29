"""Interface de apresentação local: python interface.py (Ctrl+C para encerrar)."""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
from modelo_v2 import predict_bundle

ROOT = Path(__file__).resolve().parent
ASSETS = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/app.css': ('app.css', 'text/css; charset=utf-8'),
    '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
}


def make_server(port=8765, model_path=None):
    model_path = Path(model_path) if model_path else ROOT/'resultados_v2/modelo_selecionado.joblib'
    bundle = joblib.load(model_path)
    if bundle.get('schema_version') != 2:
        raise ValueError('A interface precisa do modelo revisado v2.')
    for filename, _ in ASSETS.values():
        if not (ROOT/'interface_web'/filename).is_file():
            raise FileNotFoundError(f'Arquivo da interface ausente: {filename}')
    model_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Não registrar textos, requisições nem resultados em disco.

        def respond(self, status, payload, content_type='application/json; charset=utf-8'):
            if isinstance(payload, dict):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # O usuário pode cancelar uma análise ou fechar a aba.

        def allowed_host(self):
            return self.headers.get('Host') in {
                f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.allowed_host():
                return self.respond(403, {'error': 'Acesso disponível apenas pelo endereço local.'})
            if self.path == '/api/config':
                return self.respond(200, {'version': 2, 'model': bundle['model_name'],
                    'policy': bundle['policy'], 'scope': bundle['scope']})
            if self.path in ASSETS:
                filename, mime = ASSETS[self.path]
                return self.respond(200, (ROOT/'interface_web'/filename).read_bytes(), mime)
            if self.path == '/favicon.ico':
                return self.respond(204, b'')
            self.respond(404, {'error': 'Página não encontrada.'})

        def do_POST(self):
            if not self.allowed_host():
                return self.respond(403, {'error': 'Endereço local inválido.'})
            origin = self.headers.get('Origin')
            expected = f'http://{self.headers.get("Host")}'
            if (origin is not None and origin != expected) or self.headers.get('X-Local-UI') != '1':
                return self.respond(403, {'error': 'Abra a interface local para analisar o texto.'})
            if self.path != '/api/analyze':
                return self.respond(404, {'error': 'Operação não encontrada.'})
            if self.headers.get_content_type() != 'application/json':
                return self.respond(415, {'error': 'Envie o texto em JSON.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                return self.respond(400, {'error': 'Tamanho de requisição inválido.'})
            if not 0 < length <= 650000:
                return self.respond(413, {'error': 'O texto enviado é grande demais.'})
            self.connection.settimeout(15)
            try:
                data = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(data, dict) or not isinstance(data.get('text'), str):
                    raise ValueError('Informe um texto para analisar.')
                with model_lock:
                    result = predict_bundle(bundle, data['text'])
                self.respond(200, result)
            except (ValueError, UnicodeDecodeError):
                self.respond(400, {'error': 'Informe um texto não vazio, com até 100.000 caracteres.'})
            except TimeoutError:
                self.respond(408, {'error': 'O envio demorou demais. Tente novamente.'})
            except Exception:
                self.respond(500, {'error': 'Não foi possível concluir a análise. Reinicie a interface e tente novamente.'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Escolha uma porta entre 1 e 65535.')
    print('Carregando o modelo local...', flush=True)
    try:
        server = make_server(args.port)
    except OSError as exc:
        parser.exit(1, f'Não foi possível iniciar: {exc}\nSe a porta estiver ocupada, use --port 8766.\n')
    except ValueError as exc:
        parser.exit(1, f'Modelo incompatível: {exc}\n')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Interface disponível: {url}\nMantenha este terminal aberto. Ctrl+C encerra a interface.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nInterface encerrada.', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
