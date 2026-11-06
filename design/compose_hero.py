"""Composes assets/hero.png from the real dark window render (rendered on CI by --render-assets)."""
import base64, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent.parent
window = Path(sys.argv[1])
icon = root / "assets/icon.png"
b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{margin:0;box-sizing:border-box}}
body{{width:1500px;height:1170px;font-family:-apple-system,"SF Pro Display","Inter","Helvetica Neue",Arial,sans-serif;
background:radial-gradient(1200px 700px at 50% 0%, #2A3170 0%, #161A3A 45%, #0E1024 100%);color:#fff;overflow:hidden;position:relative}}
.grid{{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:48px 48px;mask-image:radial-gradient(900px 500px at 50% 20%,#000,transparent)}}
.head{{position:absolute;top:58px;left:0;right:0;display:flex;flex-direction:column;align-items:center}}
.brand{{display:flex;align-items:center;gap:18px}}
.brand img{{width:84px;height:84px}}
.brand h1{{font-size:56px;font-weight:700;letter-spacing:-1px}}
.tag{{margin-top:12px;font-size:23px;color:#C3C8F0;letter-spacing:-.1px}}
.win{{position:absolute;top:228px;left:130px;width:1240px;border-radius:12px;box-shadow:0 40px 100px rgba(0,0,0,.55),0 0 0 1px rgba(255,255,255,.06)}}
</style></head><body><div class="grid"></div>
<div class="head"><div class="brand"><img src="data:image/png;base64,{b64(icon)}"><h1>ModelFit</h1></div>
<div class="tag">Can your Mac run this AI model? GGUF, safetensors and MLX, analyzed from the header in a second.</div></div>
<img class="win" src="data:image/png;base64,{b64(window)}">
</body></html>"""
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1170}, device_scale_factor=2)
    pg.set_content(html); pg.wait_for_timeout(400)
    pg.screenshot(path=str(root / "assets/hero.png"))
    b.close()
print("wrote assets/hero.png")
