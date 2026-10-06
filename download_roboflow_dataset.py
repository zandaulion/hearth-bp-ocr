# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""
Download Roboflow Open Dataset for Blood Pressure Monitors
Supports two authentication methods:
1. Roboflow API Key:
   python download_roboflow_dataset.py --api-key <YOUR_KEY>
   or setting ROBOFLOW_API_KEY environment variable.

2. Browser OAuth 2.0 (Dynamic Client Registration + PKCE):
   python download_roboflow_dataset.py --oauth
"""

import argparse
import base64
import hashlib
import json
import os
import secrets
import sys
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import zipfile

WORKSPACE_ID = "final-project-cwtfb"
PROJECT_ID = "blood-pressure-monitor-display"
VERSION_NUM = 1
EXPORT_FORMAT = "yolov8"
OUTPUT_DIR = Path(__file__).resolve().parent / "dataset" / "roboflow_bp_display"

AUTH_SERVER = "https://app.roboflow.com"
REDIRECT_URI = "http://localhost:8080/callback"


def get_api_key_from_env_or_files() -> str | None:
    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ.get("ROBOFLOW_API_KEY")
    for candidate in [
        Path(__file__).resolve().parent / ".env",
        Path.home() / ".env",
        Path.home() / ".roboflow" / "config.json",
    ]:
        if candidate.exists():
            try:
                if candidate.suffix == ".json":
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    if "api_key" in data or "ROBOFLOW_API_KEY" in data:
                        return data.get("api_key") or data.get("ROBOFLOW_API_KEY")
                else:
                    for line in candidate.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line.startswith("ROBOFLOW_API_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
    return None


def download_with_api_key(api_key: str, export_format: str = EXPORT_FORMAT):
    print(f"[*] Authenticating with Roboflow API key...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        print(f"[*] Connecting to workspace '{WORKSPACE_ID}', project '{PROJECT_ID}'...")
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        version = project.version(VERSION_NUM)
        print(f"[*] Downloading dataset version {VERSION_NUM} in format '{export_format}' to {OUTPUT_DIR}...")
        dataset = version.download(export_format, location=str(OUTPUT_DIR), overwrite=True)
        print(f"[+] Download complete! Location: {dataset.location}")

        # Verify files
        image_files = list(OUTPUT_DIR.rglob("*.jpg")) + list(OUTPUT_DIR.rglob("*.jpeg")) + list(OUTPUT_DIR.rglob("*.png"))
        print(f"[+] Total images downloaded: {len(image_files)}")
        label_files = list(OUTPUT_DIR.rglob("*.txt"))
        print(f"[+] Total label files: {len(label_files)}")
        return True
    except Exception as e:
        print(f"[-] Error downloading with Roboflow SDK: {e}")
        return False


def run_oauth_flow():
    print("[*] Registering OAuth client dynamically with Roboflow...")
    reg_req = urllib.request.Request(
        f"{AUTH_SERVER}/oauth/register",
        data=json.dumps({
            "client_name": "Antigravity Blood Pressure OCR",
            "redirect_uris": [REDIRECT_URI],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Antigravity/1.0"},
    )
    with urllib.request.urlopen(reg_req) as resp:
        reg_data = json.loads(resp.read().decode("utf-8"))
    client_id = reg_data["client_id"]
    print(f"[+] Client registered (ID: {client_id})")

    # Generate PKCE verifier and challenge
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("utf-8")).digest()).decode("utf-8").rstrip("=")

    auth_params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid profile email workspace:read project:read version:read",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    authorize_url = f"{AUTH_SERVER}/oauth/authorize?" + urllib.parse.urlencode(auth_params)

    callback_code = None

    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal callback_code
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if "code" in query:
                callback_code = query["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Authentication successful!</h1><p>You can close this tab and return to the terminal.</p>")
            else:
                err = query.get("error_description", ["Unknown error"])[0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"<h1>Authentication failed: {err}</h1>".encode("utf-8"))

        def log_message(self, format, *args):
            return  # Suppress logging

    server = HTTPServer(("localhost", 8080), OAuthCallbackHandler)
    print("\n" + "=" * 60)
    print("Please open the following authorization URL in your browser:")
    print(authorize_url)
    print("=" * 60 + "\n")
    try:
        webbrowser.open(authorize_url)
    except Exception:
        pass

    print("[*] Waiting for browser authorization on http://localhost:8080/callback...")
    server.handle_request()

    if not callback_code:
        print("[-] Authorization failed or canceled.")
        return False

    print("[*] Authorization code received. Exchanging for access token...")
    token_req = urllib.request.Request(
        f"{AUTH_SERVER}/oauth/token",
        data=urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": callback_code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": code_verifier,
        }).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Antigravity/1.0"},
    )
    with urllib.request.urlopen(token_req) as resp:
        token_data = json.loads(resp.read().decode("utf-8"))

    access_token = token_data.get("access_token")
    print(f"[+] Successfully obtained access token!")

    # Use access token with Roboflow REST API to export/download
    print(f"[*] Requesting export URL for {WORKSPACE_ID}/{PROJECT_ID} version {VERSION_NUM}...")
    export_url = f"https://api.roboflow.com/{WORKSPACE_ID}/{PROJECT_ID}/{VERSION_NUM}/{EXPORT_FORMAT}"
    req = urllib.request.Request(export_url, headers={"Authorization": f"Bearer {access_token}"})
    with urllib.request.urlopen(req) as resp:
        export_resp = json.loads(resp.read().decode("utf-8"))

    zip_url = export_resp.get("export", {}).get("link")
    if not zip_url:
        print(f"[-] Could not get export download link: {export_resp}")
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUT_DIR / "dataset.zip"
    print(f"[*] Downloading dataset archive from {zip_url}...")
    urllib.request.urlretrieve(zip_url, zip_path)
    print(f"[*] Extracting archive to {OUTPUT_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(OUTPUT_DIR)
    zip_path.unlink()
    print(f"[+] Dataset successfully downloaded and extracted to {OUTPUT_DIR}!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Roboflow BP monitor dataset")
    parser.add_argument("--api-key", type=str, default=get_api_key_from_env_or_files(), help="Roboflow API Key")
    parser.add_argument("--oauth", action="store_true", help="Authenticate using browser OAuth")
    parser.add_argument("--format", type=str, default=EXPORT_FORMAT, help="Export format (e.g. yolov8, voc, coco)")
    args = parser.parse_args()

    if args.api_key:
        download_with_api_key(args.api_key, args.format)
    elif args.oauth:
        run_oauth_flow()
    else:
        print("[!] No API key or --oauth flag specified.")
        print("[!] Options:")
        print("    1. Pass --api-key <KEY> (Find yours at https://app.roboflow.com > Settings > API Keys)")
        print("    2. Pass --oauth to log in via browser")
        print("\nAttempting browser OAuth flow now...")
        run_oauth_flow()


if __name__ == "__main__":
    main()
