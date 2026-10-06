# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Render the prototype's simple pulse mark at standard PWA icon sizes."""
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parents[1] / "web"
for size in (192, 512):
    scale = size * 4 / 128
    image = Image.new("RGB", (size * 4, size * 4), "#164b43")
    draw = ImageDraw.Draw(image)
    points = [(int(x * scale), int(y * scale)) for x, y in
              ((25, 66), (45, 66), (54, 43), (69, 86), (79, 66), (103, 66))]
    width = round(7 * scale)
    draw.line(points, fill="#e0edcc", width=width, joint="curve")
    for x, y in points:
        radius = width / 2
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="#e0edcc")
    image.resize((size, size), Image.Resampling.LANCZOS).save(out / f"icon-{size}.png")
