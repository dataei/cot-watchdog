import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from memory import list_all_tasks, load_task_timeline
from openshell.policy_enforcer import read_audit_log
PORT = 8000
HTML_PATH = Path(__file__).parent / "timeline.html"
class TimelineHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/tasks":
            self._serve_json(list_all_tasks())
        elif self.path.startswith("/api/timeline/"):
            task_id = self.path.split("/")[-1]
            timeline = load_task_timeline(task_id)
            if timeline is None:
                self._serve_404()
            else:
                self._serve_json(timeline)
        elif self.path == "/api/audit":
            self._serve_json(read_audit_log())
        else:
            self._serve_404()
    def _serve_html(self):
        try:
            html = HTML_PATH.read_text()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        except FileNotFoundError:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"timeline.html not found")
    def _serve_json(self, data):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def _serve_404(self):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Not Found")
    def log_message(self, format, *args):
        # Quiet down the default request logging
        pass
def main():
    print(f"\n  CoT Watchdog Timeline UI")
    print(f"  http://localhost:{PORT}\n")
    httpd = HTTPServer(("localhost", PORT), TimelineHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
if __name__ == "__main__":
    main()
 