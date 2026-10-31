"""Validate and render the seven supplement PDFs; write only under qa/."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import fitz
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve()
OUTROOT = HERE.parents[1]
ROOT = HERE.parents[3]
FIGURES = OUTROOT / "figures"
QA = OUTROOT / "qa"
RENDERED = QA / "rendered"
MANIFEST = OUTROOT / "manifest.json"
EXPECTED = [
    "figS01_joint_spectrum_2d.pdf",
    "figS02_heldout_daily_scores.pdf",
    "figS03_coherent_response_train_test.pdf",
    "figS04_anisotropy_distributions.pdf",
    "figS05_matched_capacity.pdf",
    "figS06_heldout_reconstruction.pdf",
    "figS07_nonlinear_exploratory.pdf",
]
REQUIRED_CITY_TEXT = ("Porto", "San Francisco", "Chengdu")
FORBIDDEN_LEGACY_CITIES = ("Rome", "Chongqing", "Hangzhou", "Jilin", "Shanghai", "Yantai")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-reviewed", action="store_true",
                        help="record that the current 200-dpi renders were visually inspected")
    args = parser.parse_args()
    RENDERED.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures, warnings = [], []
    for rel, expected in manifest["inputs_sha256"].items():
        actual = sha256(ROOT / rel)
        if actual != expected:
            failures.append(f"input hash changed: {rel}")
    actual_files = sorted(p.name for p in FIGURES.glob("*.pdf"))
    if actual_files != EXPECTED:
        failures.append(f"PDF set mismatch: {actual_files}")
    records = {}
    for name in EXPECTED:
        path = FIGURES / name
        rec = {"file": str(path), "bytes": path.stat().st_size}
        if sha256(path) != manifest["outputs_sha256"].get(name):
            failures.append(f"output hash differs from manifest: {name}")
        doc = fitz.open(path)
        rec["pages"] = len(doc)
        if len(doc) != 1:
            failures.append(f"{name}: expected one page, got {len(doc)}")
        page = doc[0]
        rec["page_points"] = [float(page.rect.width), float(page.rect.height)]
        rec["metadata"] = {k: doc.metadata.get(k, "") for k in ("title", "author", "creator", "subject")}
        for key in ("title", "author", "creator", "subject"):
            if not rec["metadata"][key]:
                failures.append(f"{name}: missing PDF metadata {key}")
        text = page.get_text()
        rec["text_characters"] = len(text)
        rec["city_labels"] = {city: city in text for city in REQUIRED_CITY_TEXT}
        required_cities = ("Chengdu",) if name.startswith("figS07_") else REQUIRED_CITY_TEXT
        for city in required_cities:
            if city not in text:
                failures.append(f"{name}: missing city label {city}")
        if name.startswith("figS03_"):
            if "not reported" in text or "18-hour DFT bin m" not in text:
                failures.append(f"{name}: Chengdu ten-bin label/stale not-reported region check failed")
        if name.startswith("figS06_"):
            if "D + TRAIN climatology" in text or "raw TEST inputs/targets" not in text or "P-frame" not in text:
                failures.append(f"{name}: uniform P-frame raw-TEST caption check failed")
        if name.startswith("figS07_"):
            for phrase in ("exploratory; 3 TEST dates; underpowered", "full-FACE", "not inside-POD"):
                if phrase not in text:
                    failures.append(f"{name}: missing exploratory/full-FACE caveat: {phrase}")
        for city in FORBIDDEN_LEGACY_CITIES:
            if city in text:
                failures.append(f"{name}: legacy city label {city}")
        if "�" in text or "□" in text:
            failures.append(f"{name}: replacement/tofu glyph in extracted text")
        sizes, fonts, outside = [], set(), []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if not span.get("text", "").strip():
                        continue
                    size = float(span["size"])
                    sizes.append(size)
                    fonts.add(span["font"])
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < -0.5 or y0 < -0.5 or x1 > page.rect.width + 0.5 or y1 > page.rect.height + 0.5:
                        outside.append({"text": span["text"], "bbox": span["bbox"]})
        rec["font_span_count"] = len(sizes)
        rec["minimum_text_size_pt"] = min(sizes) if sizes else None
        rec["fonts"] = sorted(fonts)
        rec["text_spans_outside_page"] = outside
        if not sizes:
            failures.append(f"{name}: no selectable vector text")
        elif min(sizes) < 6.0 - 1e-6:
            failures.append(f"{name}: text below 6 pt ({min(sizes):.4f})")
        if outside:
            failures.append(f"{name}: {len(outside)} text spans outside page")
        embedded = []
        for item in page.get_fonts(full=True):
            xref = int(item[0])
            basefont = item[3]
            try:
                extracted = doc.extract_font(xref)
                payload = extracted[3] if len(extracted) > 3 else b""
                embedded.append({"font": basefont, "embedded_bytes": len(payload)})
                if not payload:
                    warnings.append(f"{name}: font has no extracted embedded payload: {basefont}")
            except Exception as exc:
                embedded.append({"font": basefont, "error": str(exc)})
                warnings.append(f"{name}: font embedding check failed: {basefont}: {exc}")
        rec["font_embedding"] = embedded
        zoom = 200.0 / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png = RENDERED / f"{path.stem}.png"
        pix.save(png)
        rec["rendered_png"] = str(png)
        im = Image.open(png).convert("L")
        ink = np.asarray(im) < 248
        ys, xs = np.where(ink)
        if not len(xs):
            failures.append(f"{name}: blank 200-dpi render")
        else:
            margins = [int(xs.min()), int(ys.min()), int(im.width - 1 - xs.max()), int(im.height - 1 - ys.max())]
            rec["render_pixels"] = [im.width, im.height]
            rec["ink_margins_pixels_left_top_right_bottom"] = margins
            rec["ink_bbox_area_fraction"] = float(((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) /
                                                  (im.width * im.height))
            if min(margins) < 3:
                failures.append(f"{name}: rendered content touches page edge; margins={margins}")
        doc.close()
        records[name] = rec
    result = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "failure_count": len(failures), "warning_count": len(warnings),
        "failures": failures, "warnings": warnings,
        "input_hash_count": len(manifest["inputs_sha256"]),
        "data_assertion_count": manifest["check_count"],
        "checks": ["input SHA-256 unchanged", "exact seven-PDF set", "one page per PDF",
                   "required metadata", "all three city labels", "no legacy city labels",
                   "selectable text", "all extracted text >= 6 pt", "no text outside page",
                   "nonblank 200-dpi render", "rendered content inside page"],
        "visual_review": {
            "status": "passed_by_rendered_png_inspection" if args.visual_reviewed else "pending",
            "resolution_dpi": 200,
            "scope": "all seven current rendered pages" if args.visual_reviewed else None,
        },
        "files": records,
    }
    (QA / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": failures, "warnings": warnings,
                      "minimum_fonts": {k: v["minimum_text_size_pt"] for k, v in records.items()}}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
