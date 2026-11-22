#!/usr/bin/env python3
"""serve.py — 本地静态服务器 (手机/局域网也可访问)。

用法:  python web/serve.py [--port 8000] [--host 127.0.0.1]
然后浏览器打开 http://127.0.0.1:8000/
数据文件放在 web/data/demo.json (由 tools/export_demo.py 生成)。
"""
import argparse, functools, http.server, os, socketserver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
    with socketserver.TCPServer((a.host, a.port), handler) as httpd:
        httpd.allow_reuse_address = True
        print('serving %s on http://%s:%d/' % (root, a.host, a.port))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
