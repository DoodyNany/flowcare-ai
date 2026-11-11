"""Builds the single-file dashboard: web/flowcare_ai.html (models + data embedded)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "export"
load = lambda n: json.load(open(E / n))
data = {
    "forecast": load("forecast_model.json"),
    "risk": load("risk_model.json"),
    "xray": load("xray_model.json"),
    "samples": load("xray_samples.json"),
    "ct": load("ct_model.json"),
    "ctSamples": load("ct_samples.json"),
    "results": {"forecast": load("forecast_results.json"), "risk": load("risk_results.json"),
                "xray": load("xray_results.json"), "ct": load("ct_results.json")},
}
html = (ROOT / "web" / "template.html").read_text()
html = html.replace("/*__IMAGING_JS__*/", (ROOT / "web" / "imaging_js.js").read_text())
html = html.replace("/*__MODELS_JS__*/", (ROOT / "web" / "models.js").read_text())
html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
logo = (ROOT / "brand" / "logo.svg").read_text().strip()
import urllib.parse
html = html.replace("/*__LOGO_SVG__*/", logo.replace(' width="120" height="120"', '', 1))
html = html.replace("/*__LOGO_URI__*/", urllib.parse.quote(logo))
out = ROOT / "web" / "flowcare_ai.html"
out.write_text(html)
print(out, round(out.stat().st_size / 1e6, 2), "MB")
