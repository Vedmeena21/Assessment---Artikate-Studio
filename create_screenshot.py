import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    return pretty


def render_text_to_image(text: str, out_path: Path, padding=16, bg=(255, 255, 255), fg=(18, 18, 18)):
    # try to use a monospace system font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 14)
    except Exception:
        font = ImageFont.load_default()


    # estimate size using a temporary draw context
    lines = text.splitlines()
    tmp_img = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)
    widths = []
    heights = []
    for line in lines:
        bbox = tmp_draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        widths.append(w)
        heights.append(h)

    max_width = max(widths) if widths else 0
    line_height = max(heights) if heights else font.getmetrics()[1]
    img_w = max_width + padding * 2
    img_h = line_height * len(lines) + padding * 2

    img = Image.new("RGB", (img_w, img_h), color=bg)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=fg)
        y += line_height

    img.save(out_path)


def main():
    base = Path(__file__).resolve().parent
    json_path = base / "sample_output.json"
    out_path = base / "sample_output.png"
    pretty = load_json(json_path)
    render_text_to_image(pretty, out_path)
    print(f"Wrote screenshot to: {out_path}")


if __name__ == "__main__":
    main()
