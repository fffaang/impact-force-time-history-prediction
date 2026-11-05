"""Loopback-only offline prediction application and native Windows launcher."""
from __future__ import annotations
import argparse
import csv
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
from urllib.parse import urlsplit
import webbrowser
from engine import Predictor, InputError, ROOT


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    server_version = 'ImpactCurve/1.1'

    def log_message(self, message, *args):
        logging.info(message, *args)

    def send(self, code, data, content_type='application/json; charset=utf-8'):
        if isinstance(data, (dict,list)): data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
        elif isinstance(data, str): data=data.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','DENY')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get('Host') in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']

    def do_GET(self):
        if not self.valid_host(): return self.send(403,dict(error='Invalid host'))
        path = urlsplit(self.path).path
        if path=='/favicon.ico': return self.send(204,b'','image/x-icon')
        if path=='/api/health': return self.send(200,dict(status='ready',model='C',partition=20261102))
        if path=='/api/meta': return self.send(200,self.server.predictor.metadata()|dict(token=self.server.token))
        if path=='/api/template':
            return self.send(200,'\ufeff'+self.server.predictor.template_csv(),'text/csv; charset=utf-8')
        mapping={'/':'index.html','/index.html':'index.html','/static/app.js':'app.js',
                 '/static/style.css':'style.css','/static/plotly.min.js':'vendor/plotly.min.js',
                 '/static/lucide.min.js':'vendor/lucide.min.js','/favicon.ico':'favicon.ico'}
        if path not in mapping: return self.send(404,dict(error='Not found'))
        file=ROOT/'static'/mapping[path]
        if not file.exists(): return self.send(404,dict(error='Not found'))
        kind=mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
        if file.suffix=='.js':kind='application/javascript'
        self.send(200,file.read_bytes(),kind+'; charset=utf-8')

    def do_POST(self):
        if not self.valid_host(): return self.send(403,dict(error='Invalid host'))
        origin=self.headers.get('Origin')
        allowed=[f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']
        if origin and origin not in allowed: return self.send(403,dict(error='Origin refused'))
        if not secrets.compare_digest(self.headers.get('X-Session-Token',''),self.server.token):
            return self.send(403,dict(error='Session token required'))
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=2_000_000: return self.send(413,dict(error='Request too large or empty'))
            body=self.rfile.read(length).decode('utf-8-sig')
            path=urlsplit(self.path).path
            if path=='/api/predict':
                return self.send(200,self.server.predictor.predict(json.loads(body)))
            if path=='/api/batch':
                return self.send(200,self.server.predictor.batch_csv(body))
            if path=='/api/shutdown':
                self.send(200,dict(status='closed'))
                threading.Thread(target=self.server.shutdown,daemon=True).start()
                return
            return self.send(404,dict(error='Not found'))
        except InputError as e:
            return self.send(422,dict(error=str(e),details=e.details))
        except (ValueError,TypeError,UnicodeError,csv.Error):
            return self.send(400,dict(error='Invalid parameter format. Check the input.'))
        except Exception:
            logging.exception('Prediction request failed')
            return self.send(500,dict(error='Prediction failed. Check the local log or restart the tool.'))


def main():
    parser=argparse.ArgumentParser(description='ImpactCurve local saved-model prediction tool')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--state-file',type=Path)
    args=parser.parse_args()
    if not 0<=args.port<=65535:parser.error('Invalid port')
    executable_dir=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).parent
    state=args.state_file or executable_dir/'.runtime/instance.json'
    try: state.parent.mkdir(parents=True,exist_ok=True)
    except PermissionError:
        state=Path(tempfile.gettempdir())/'ImpactCurve/instance.json'
        state.parent.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=state.parent/'impactcurve.log',level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',encoding='utf-8')
    try:
        predictor=Predictor()
        ports=[0] if args.port==0 else range(args.port,min(args.port+20,65536))
        server=None
        for port in ports:
            try:
                server=Server(('127.0.0.1',port),Handler)
                break
            except OSError:continue
        if server is None:raise RuntimeError('No available loopback port')
        server.predictor=predictor
        server.token=secrets.token_urlsafe(32)
        url='http://127.0.0.1:'+str(server.server_port)
        state.write_text(json.dumps(dict(url=url,pid=os.getpid(),model='C',partition=20261102)),encoding='utf-8')
        if not args.no_browser: webbrowser.open(url)
        if sys.stdout:print(url,flush=True)
        try:server.serve_forever(poll_interval=.2)
        finally:server.server_close()
    except Exception as e:
        logging.exception('Startup failed')
        if getattr(sys,'frozen',False) and os.name=='nt':
            import ctypes
            ctypes.windll.user32.MessageBoxW(0,'Startup failed: '+str(e)+'\nLog: '+str(state.parent/'impactcurve.log'),'ImpactCurve',16)
        raise


if __name__=='__main__': main()
