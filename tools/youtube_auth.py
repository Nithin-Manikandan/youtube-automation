"""One-time: get a YouTube refresh token.  python tools/youtube_auth.py CLIENT_ID CLIENT_SECRET
Create an OAuth client of type "Desktop app" in Google Cloud Console first (README explains)."""
import http.server
import sys
import threading
import urllib.parse
import webbrowser

import requests

cid, secret = sys.argv[1], sys.argv[2]
PORT = 8765
redirect = f"http://127.0.0.1:{PORT}"
scope = "https://www.googleapis.com/auth/youtube.upload"
code = {}


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        code["v"] = q.get("code", [""])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Done. You can close this tab and go back to the terminal.")
        threading.Thread(target=self.server.shutdown).start()

    def log_message(self, *a):
        pass


url = ("https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
    "client_id": cid, "redirect_uri": redirect, "response_type": "code", "scope": scope,
    "access_type": "offline", "prompt": "consent"}))
print("Opening browser. If it does not open, visit:\n", url)
webbrowser.open(url)
http.server.HTTPServer(("127.0.0.1", PORT), H).serve_forever()
r = requests.post("https://oauth2.googleapis.com/token", data={
    "code": code["v"], "client_id": cid, "client_secret": secret, "redirect_uri": redirect,
    "grant_type": "authorization_code"})
print("\nYT_REFRESH_TOKEN =", r.json().get("refresh_token", r.text))
