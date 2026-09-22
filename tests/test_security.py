"""
Security verification tests for SMS Spam Detection applications.
Tests:
- Directory traversal defenses in web_app.py HTTP server
- Payload size and malformed input limits (DoS prevention)
- Security headers presence (MIME-sniffing, Clickjacking)
- HTML injection / XSS neutralization in Streamlit view
"""

import os
import sys
import json
import html
import threading
import time
import urllib.request
import urllib.error
from http.server import HTTPServer

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app import SpamDetectionServer, load_model, MAX_CONTENT_LENGTH
from src.features import normalize


def run_tests():
    print("=" * 60)
    print("RUNNING SECURITY INTEGRITY TESTS")
    print("=" * 60)

    # 1. Test XSS escaping logic used in app.py
    print("[1] Testing XSS sanitization...")
    raw_xss = '<script>alert("pwned")</script><img src=x onerror=alert(1)>'
    norm = normalize(raw_xss)
    escaped = html.escape(norm)
    assert "<script>" not in escaped, "Failed: script tag not escaped"
    assert "<img" not in escaped, "Failed: img tag not escaped"
    assert "&lt;script&gt;" in escaped, "Failed: script entity missing"
    print("    [PASS] XSS payload successfully neutralized via html.escape.")

    # 2. Spin up test server on ephemeral port
    print("\n[2] Starting isolated test HTTP server...")
    load_model()
    test_port = 5599
    server = HTTPServer(("127.0.0.1", test_port), SpamDetectionServer)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{test_port}"

    try:
        # 3. Test Static File Serving & Security Headers
        print("\n[3] Testing legitimate static file serving & security headers...")
        req = urllib.request.Request(f"{base_url}/index.html")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            headers = dict(resp.headers)
            assert headers.get("X-Content-Type-Options") == "nosniff", "Missing X-Content-Type-Options"
            assert headers.get("X-Frame-Options") == "DENY", "Missing X-Frame-Options"
            content = resp.read()
            assert b"SMS Spam Shield" in content, "Failed to read index.html content"
        print("    [PASS] Legitimate file served with nosniff and DENY headers.")

        # 4. Test Directory Traversal Defenses
        print("\n[4] Testing Directory Traversal Defenses...")
        traversal_urls = [
            f"{base_url}/../web_app.py",
            f"{base_url}/..%2fweb_app.py",
            f"{base_url}/../../requirements.txt",
            f"{base_url}/..\\..\\app.py",
            f"{base_url}/nonexistent.html",
        ]

        for url in traversal_urls:
            try:
                urllib.request.urlopen(url)
                raise AssertionError(f"VULNERABILITY DETECTED: Path traversal succeeded for {url}")
            except urllib.error.HTTPError as e:
                # 403 Forbidden or 404 Not Found are safe responses
                assert e.code in (400, 403, 404), f"Unexpected response code {e.code} for {url}"
                print(f"    [BLOCKED] Traversal attempt {url} -> HTTP {e.code}")

        # 5. Test Health Check Endpoint
        print("\n[5] Testing Health Check endpoint...")
        with urllib.request.urlopen(f"{base_url}/api/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
        print("    [PASS] /api/health returned healthy.")

        # 6. Test Valid API Prediction
        print("\n[6] Testing Valid API Prediction...")
        valid_payload = json.dumps({"text": "WINNER! You won £1000 cash.", "threshold": 0.709}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/predict",
            data=valid_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["is_spam"] is True
            assert res["verdict"] == "SPAM"
        print(f"    [PASS] /api/predict succeeded: prob={res['spam_probability']}, verdict={res['verdict']}")

        # 7. Test Payload Too Large (DoS Protection)
        print("\n[7] Testing DoS / Payload Size Protection (> 64 KB)...")
        oversized_data = b"A" * (MAX_CONTENT_LENGTH + 500)
        req = urllib.request.Request(
            f"{base_url}/api/predict",
            data=oversized_data,
            headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req)
            raise AssertionError("Failed: Oversized payload was accepted without 413 error")
        except urllib.error.HTTPError as e:
            assert e.code == 413, f"Expected HTTP 413 Payload Too Large, got {e.code}"
            print(f"    [PASS] Oversized payload rejected with HTTP {e.code}.")

        # 8. Test Malformed JSON Handling
        print("\n[8] Testing Malformed JSON handling...")
        bad_json = b"{not-a-valid-json}"
        req = urllib.request.Request(
            f"{base_url}/api/predict",
            data=bad_json,
            headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req)
            raise AssertionError("Failed: Malformed JSON was accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 400, f"Expected HTTP 400 Bad Request, got {e.code}"
            print(f"    [PASS] Malformed JSON rejected with HTTP {e.code}.")

    finally:
        server.shutdown()
        server.server_close()
        print("\n[OK] Test server shutdown cleanly.")

    print("\n" + "=" * 60)
    print("ALL SECURITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
