#!/usr/bin/env python3
# ruff: noqa: E501  (embedded CSS, JS and HTML)
"""Build the public project page (build/page/) from the private source page (site/).

Sections listed with --lock are encrypted with AES-256-GCM (key from PBKDF2-SHA256 of the
password) and replaced by a blurred preview plus an unlock form; the browser decrypts them with
the Web Crypto API. Images inside locked sections are embedded in the ciphertext. Every file of
site/assets/demo/ (clips, overlays, manifest) is encrypted on its own into assets/enc/<hash>.bin
with the same key; its name map lives inside the encrypted vault, and the demo decrypts the files
after unlock. Nothing from a locked section or the demo is published in clear.

Usage (ball3d env; needs google-chrome for the blurred previews):
  PAGE_PASSWORD='...' scripts/py tools/build_page.py --lock results,qualitative,demo
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "site", ROOT / "build" / "page"
DEMO = SRC / "assets" / "demo"
ITERATIONS = 600_000
SECTION_RE = r'<section id="{}"[^>]*>.*?</section>'

LOCK_CSS = """
.locked .lock { position: relative; min-height: 360px; border: 1px solid var(--rule); border-radius: 12px; overflow: hidden; background: var(--card); }
.locked .lock-preview { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: top; opacity: .6; }
.locked .lock-panel { position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; min-height: 360px; padding: 36px 20px; text-align: center; background: color-mix(in srgb, var(--bg) 62%, transparent); }
.lock-panel svg { width: 28px; height: 28px; color: var(--accent); }
.lock-title { font: 600 17px/1.3 var(--sans); margin: 0; color: var(--fg); }
.lock-text { font: 14px/1.55 var(--sans); color: var(--muted); margin: 0; max-width: 34em; text-align: center !important; }
.lock-form { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 6px; }
.lock-form input { font: 14px/1 var(--sans); padding: 10px 14px; min-width: 220px; border: 1px solid var(--rule); border-radius: 999px; background: var(--card); color: var(--fg); }
.lock-form input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.lock-form button { font: 600 14px/1 var(--sans); padding: 10px 18px; border: 1px solid var(--fg); border-radius: 999px; background: var(--fg); color: var(--bg); cursor: pointer; }
.lock-msg { font: 13px/1.4 var(--sans); color: var(--accent); min-height: 1.4em; margin: 0; text-align: center !important; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
"""

LOCK_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true"><rect x="4" y="11" width="16" height="10" rx="2"/>'
    '<path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>'
)

UNLOCK_JS = """
<script>
(function () {
  var KEY = 'unlock:' + location.pathname;
  function b64(s) { return Uint8Array.from(atob(s), function (c) { return c.charCodeAt(0); }); }
  async function derive(pw, v) {
    var base = await crypto.subtle.importKey('raw', new TextEncoder().encode(pw), 'PBKDF2', false, ['deriveKey']);
    return crypto.subtle.deriveKey({ name: 'PBKDF2', salt: b64(v.salt), iterations: v.iter, hash: 'SHA-256' },
      base, { name: 'AES-GCM', length: 256 }, false, ['decrypt']);
  }
  async function decrypt(pw) {
    var v = JSON.parse(document.getElementById('vault').textContent);
    var key = await derive(pw, v);
    var plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64(v.iv) }, key, b64(v.ct));
    return { key: key, data: JSON.parse(new TextDecoder().decode(plain)) };
  }
  function reveal(out) {
    var sections = out.data.sections;
    Object.keys(sections).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) { el.innerHTML = sections[id]; el.classList.remove('locked'); }
    });
    window.__pageVault = { key: out.key, assets: out.data.assets };
    window.dispatchEvent(new Event('page-unlocked'));
  }
  async function attempt(pw) {
    var msgs = document.querySelectorAll('.lock-msg');
    msgs.forEach(function (m) { m.textContent = 'Unlocking…'; });
    try {
      reveal(await decrypt(pw));
      try { sessionStorage.setItem(KEY, pw); } catch (e) {}
    } catch (e) {
      msgs.forEach(function (m) { m.textContent = 'That password did not work.'; });
      try { sessionStorage.removeItem(KEY); } catch (e2) {}
    }
  }
  window.pageUnlock = function (form) { attempt(form.querySelector('input').value); return false; };
  try { var saved = sessionStorage.getItem(KEY); if (saved) { window.addEventListener('load', function () { attempt(saved); }); } } catch (e) {}
})();
</script>
"""


def inline_images(html: str, base: Path) -> str:
    def repl(m):
        attr, path = m.group(1), m.group(2)
        f = base / path
        if path.startswith(("http", "data:")) or not f.is_file():
            return m.group(0)
        mime = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        return f'{attr}="data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}"'

    return re.sub(r'(src|href)="(assets/[^"]+)"', repl, html)


def preview_image(head: str, section_html: str, name: str, chrome: str) -> str:
    """Render the section alone, then blur it so nothing is legible."""
    tmp = SRC / f".preview-{name}.html"
    tmp.write_text(f'{head}<body><main class="wrap">{section_html}</main></body></html>')
    png = Path(tempfile.gettempdir()) / f"preview-{name}-{os.getpid()}.png"
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--virtual-time-budget=8000",
            "--window-size=1280,2400",
            f"--screenshot={png}",
            tmp.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=True,
    )
    tmp.unlink()
    im = Image.open(png).convert("RGB")
    png.unlink()
    im = im.crop((110, 0, 1170, min(im.height, 1100)))
    small = im.resize((im.width // 6, im.height // 6), Image.BILINEAR).filter(
        ImageFilter.GaussianBlur(2.5)
    )
    out = small.resize(im.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
    dst = OUT / "assets" / f"locked-{name}.jpg"
    out.save(dst, quality=60, optimize=True)
    return dst.name


def heading_of(section_html: str) -> str:
    m = re.search(r"<h2>(.*?)</h2>", section_html, re.S)
    return m.group(1) if m else "Section"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", required=True, help="comma-separated section ids to encrypt")
    ap.add_argument("--chrome", default=shutil.which("google-chrome") or "google-chrome")
    a = ap.parse_args()
    pw = os.environ.get("PAGE_PASSWORD") or ap.error("set PAGE_PASSWORD")
    html = (SRC / "index.html").read_text()
    head = html[: html.index("<body>")]
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "assets" / "enc").mkdir(parents=True)

    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(
        pw.encode()
    )
    aes = AESGCM(key)

    locked, used_in_locked = {}, set()
    for sid in [s.strip() for s in a.lock.split(",") if s.strip()]:
        m = re.search(SECTION_RE.format(re.escape(sid)), html, re.S)
        if not m:
            ap.error(f"section #{sid} not found")
        sec = m.group(0)
        inner = sec[sec.index(">") + 1 : sec.rindex("</section>")]
        used_in_locked |= set(re.findall(r'(?:src|href)="assets/([^"]+)"', inner))
        locked[sid] = inline_images(inner, SRC)
        prev = preview_image(head, sec, sid, a.chrome)
        stub = (
            f'<section id="{sid}" class="locked">\n  <h2>{heading_of(inner)}</h2>\n  <div class="lock">\n'
            f'    <img class="lock-preview" src="assets/{prev}" alt="" aria-hidden="true">\n'
            f'    <div class="lock-panel">{LOCK_ICON}\n      <p class="lock-title">Available on Request</p>\n'
            f'      <p class="lock-text">This section is shared with reviewers and collaborators. Enter the access password to view it.</p>\n'
            f'      <form class="lock-form" onsubmit="return pageUnlock(this)"><label class="sr-only" for="pw-{sid}">Access password</label>'
            f'<input id="pw-{sid}" type="password" placeholder="Access password" autocomplete="current-password" required>'
            f'<button type="submit">Unlock</button></form>\n      <p class="lock-msg" role="status" aria-live="polite"></p>\n'
            f"    </div>\n  </div>\n</section>"
        )
        html = html.replace(sec, stub)

    assets = {}
    demo_files = sorted(p for p in DEMO.iterdir() if p.is_file()) if DEMO.is_dir() else []
    for f in demo_files:
        nonce = secrets.token_bytes(12)
        blob = nonce + aes.encrypt(nonce, f.read_bytes(), None)
        # Content-addressed names: a new build never reuses a name a browser may have cached.
        path = f"assets/enc/{hashlib.sha256(blob).hexdigest()[:24]}.bin"
        (OUT / path).write_bytes(blob)
        assets[f.name] = path

    payload = json.dumps({"sections": locked, "assets": assets}).encode()
    ct = aes.encrypt(iv, payload, None)
    enc = lambda b: base64.b64encode(b).decode()  # noqa: E731
    vault = json.dumps({"salt": enc(salt), "iv": enc(iv), "iter": ITERATIONS, "ct": enc(ct)})
    html = html.replace("</style>", LOCK_CSS + "</style>", 1)
    html = html.replace(
        "</body>",
        f'<script type="application/json" id="vault">{vault}</script>\n{UNLOCK_JS}</body>',
        1,
    )

    public = set(re.findall(r'(?:src|href|content)="(?:[^"]*/)?assets/([^"]+)"', html))
    leaked = (public & used_in_locked) - {n for n in public if n.startswith(("locked-", "enc/"))}
    if leaked:
        ap.error(f"locked assets would be published: {sorted(leaked)}")
    for name in public:
        if (SRC / "assets" / name).is_file():
            shutil.copy2(SRC / "assets" / name, OUT / "assets" / name)
    # Version the page's own scripts so a cached copy from an older build is never mixed in.
    build_id = hashlib.sha256(ct).hexdigest()[:10]
    html = re.sub(r'src="assets/([\w.-]+\.js)"', rf'src="assets/\1?v={build_id}"', html)
    (OUT / "index.html").write_text(html)
    (OUT / ".nojekyll").touch()
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(
        f"locked: {', '.join(locked)} | demo files: {len(assets)} | public assets: "
        f"{sorted(n for n in public if (OUT / 'assets' / n).exists())} | total {size:.1f} MB"
    )


if __name__ == "__main__":
    main()
