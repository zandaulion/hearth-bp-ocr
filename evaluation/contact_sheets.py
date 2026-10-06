# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Build local contact sheets to inspect samples and annotation semantics."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)


def sheet(items, target, cols=4, tile=(320, 320)):
    canvas = Image.new("RGB", (cols * tile[0], ((len(items) + cols - 1) // cols) * tile[1]), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for i, (path, caption, annotated) in enumerate(items):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if annotated:
            d = ImageDraw.Draw(im)
            for line in path.with_name(path.stem + ".txt").parent.parent.joinpath("labels", path.stem + ".txt").read_text().splitlines():
                cid, cx, cy, w, h = map(float, line.split())
                names = ["0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9"]
                xy = ((cx-w/2)*im.width, (cy-h/2)*im.height, (cx+w/2)*im.width, (cy+h/2)*im.height)
                color = "red" if cid == 2 else "blue"
                d.rectangle(xy, outline=color, width=2)
                d.text((xy[0], xy[1]), names[int(cid)], fill=color)
        im.thumbnail((tile[0] - 8, tile[1] - 55))
        x, y = (i % cols) * tile[0], (i // cols) * tile[1]
        canvas.paste(im, (x + (tile[0] - im.width) // 2, y + 5))
        draw.text((x + 6, y + tile[1] - 47), caption, fill="black")
    canvas.save(target)


truth = json.loads((ROOT / "dataset/ground_truth.json").read_text())
sheet([(ROOT / r["local_file"], f'{r["id"]}\n{r["sys"]}/{r["dia"]} pulse {r["pulse"]}', False) for r in truth], OUT / "curated_contact.jpg")
paths = sorted((ROOT / "dataset/roboflow_bp_display/valid/images").glob("*.jpg"))[:16]
sheet([(p, p.name.split(".rf.")[0], True) for p in paths], OUT / "annotated_contact.jpg")
print(OUT)
