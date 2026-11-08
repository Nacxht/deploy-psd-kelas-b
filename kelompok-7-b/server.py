import os
import sys
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = int(os.environ.get("PORT", 3000))
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# Pastikan MIME types terdaftar dengan benar
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/mpeg", ".mp3")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/svg+xml", ".svg")

class WebAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Tambahkan header CORS dan no-cache agar model JSON termuat sempurna
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        sys.stdout.write(f"[{self.log_date_time_string()}] {format % args}\n")
        sys.stdout.flush()

def run():
    server_address = ("0.0.0.0", PORT)
    httpd = HTTPServer(server_address, WebAppHandler)
    print("=" * 60)
    print("  🚀 SISTEM DETEKSI SUARA KEBOCORAN (LEAKSOUND)")
    print("     Tugas Besar Pengolahan Sinyal Digital (PSD)")
    print("=" * 60)
    print(f"  Aplikasi berjalan di port : {PORT}")
    print(f"  Direktori web             : {WEB_DIR}")
    print("=" * 60)
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")
        httpd.server_close()

if __name__ == "__main__":
    run()

