"""
Lightweight Standalone Web Application and REST API for SMS Spam Detection.
Uses Python standard library http.server + scikit-learn model bundle.
Requires zero extra web framework installations.
"""

import os
import sys
import json
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from pathlib import Path
import joblib

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.features import normalize

# Global model state & security limits
MODEL_BUNDLE = None
DEFAULT_PORT = 5000
DEFAULT_HOST = os.environ.get("HOST", "127.0.0.1")
MAX_CONTENT_LENGTH = 64 * 1024  # 64 KB maximum payload for SMS messages
WEB_DIR = (Path(ROOT_DIR) / "web").resolve()


def load_model():
    global MODEL_BUNDLE
    model_path = os.path.join(ROOT_DIR, "models", "spam_clf.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run 'python run_pipeline.py' first.")
    MODEL_BUNDLE = joblib.load(model_path)
    print(f"[OK] Loaded spam classifier model from {model_path}")
    print(f"[OK] Default calibrated decision threshold: {MODEL_BUNDLE.get('threshold', 0.709):.4f}")


def analyze_message(text: str, custom_threshold: float = None) -> dict:
    if MODEL_BUNDLE is None:
        load_model()

    model = MODEL_BUNDLE["model"]
    default_threshold = float(MODEL_BUNDLE.get("threshold", 0.709))
    threshold = float(custom_threshold) if custom_threshold is not None else default_threshold

    # Clean text string
    clean_text = text if isinstance(text, str) else ""
    
    if not clean_text.strip():
        return {
            "text": "",
            "spam_probability": 0.0,
            "is_spam": False,
            "threshold": threshold,
            "normalized_text": "",
            "extracted_tokens": [],
        }

    # Model inference
    proba = float(model.predict_proba([clean_text])[0, 1])
    is_spam = bool(proba >= threshold)

    # Normalization & Token inspection
    normalized = normalize(clean_text)
    tokens = []
    if "urltoken" in normalized:
        tokens.append({"token": "urltoken", "label": "Web URL / Hyperlink", "type": "url"})
    if "moneytoken" in normalized:
        tokens.append({"token": "moneytoken", "label": "Currency Amount (£, $, €)", "type": "money"})
    if "longnumtoken" in normalized:
        tokens.append({"token": "longnumtoken", "label": "Phone Number / Shortcode (5+ digits)", "type": "number"})

    return {
        "text": clean_text,
        "spam_probability": round(proba, 4),
        "spam_percentage": round(proba * 100, 1),
        "is_spam": is_spam,
        "verdict": "SPAM" if is_spam else "HAM",
        "threshold": round(threshold, 4),
        "normalized_text": normalized,
        "extracted_tokens": tokens,
        "model_architecture": "TF-IDF (1-gram) + LogisticRegression (C=30, Balanced)",
    }


class SpamDetectionServer(BaseHTTPRequestHandler):
    def end_headers(self):
        # Security headers to mitigate MIME-sniffing and clickjacking
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        # Enable CORS for API consumers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "healthy",
                "model_loaded": MODEL_BUNDLE is not None,
                "calibrated_threshold": MODEL_BUNDLE.get("threshold", 0.709) if MODEL_BUNDLE else None
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        # Serve static web frontend
        if path == "/" or path == "/index.html":
            target_path = (WEB_DIR / "index.html").resolve()
        else:
            # Unquote URL and strip leading slash / backslash to prevent traversal
            rel_path = urllib.parse.unquote(path).lstrip("/\\")
            target_path = (WEB_DIR / rel_path).resolve()

        # Strict security check to prevent directory traversal
        try:
            if not target_path.is_relative_to(WEB_DIR):
                self.send_error(403, "Access Denied")
                return
        except (ValueError, RuntimeError):
            self.send_error(400, "Bad Request")
            return

        if target_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(target_path))
            if mime_type is None:
                mime_type = "application/octet-stream"

            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(target_path.stat().st_size))
            self.end_headers()
            with open(target_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            try:
                raw_len = self.headers.get("Content-Length")
                if raw_len is None or not raw_len.strip().isdigit():
                    self.send_error(400, "Invalid or missing Content-Length header")
                    return

                content_len = int(raw_len)
                if content_len < 0:
                    self.send_error(400, "Content-Length must be non-negative")
                    return
                if content_len > MAX_CONTENT_LENGTH:
                    self.send_error(413, f"Payload Too Large (max {MAX_CONTENT_LENGTH} bytes)")
                    return

                post_body = self.rfile.read(content_len).decode("utf-8", errors="replace")
                data = json.loads(post_body) if post_body else {}
                if not isinstance(data, dict):
                    self.send_error(400, "Malformed request body: JSON object required")
                    return

                text = data.get("text", "")
                threshold = data.get("threshold", None)
                if threshold is not None:
                    threshold = float(threshold)

                result = analyze_message(text, threshold)

                resp_bytes = json.dumps(result, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON in request body")
            except Exception as e:
                # Log detailed error internally without leaking system info to client
                sys.stderr.write(f"[Internal Server Error] {e}\n")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Internal server error occurred."}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint Not Found")

    def log_message(self, format, *args):
        # Clean standard logging format supporting both requests and errors
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    # Ensure stdout handles UTF-8 on Windows
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    load_model()
    server_address = (host, port)
    httpd = HTTPServer(server_address, SpamDetectionServer)
    print("=" * 60)
    print(f"[*] SMS Spam Detection Web App is active!")
    print(f"[*] Local Web UI:   http://{host}:{port}")
    print(f"[*] API Endpoint:   http://{host}:{port}/api/predict")
    print(f"[*] Health Check:   http://{host}:{port}/api/health")
    print("Press Ctrl+C to terminate.")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    host_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_HOST
    run(host=host_arg, port=port_arg)

