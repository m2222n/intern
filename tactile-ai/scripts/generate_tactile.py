# tactile-ai/scripts/generate_tactile.py
import argparse, os, sys, textwrap
from PIL import Image, ImageDraw, ImageFont

def find_korean_font():
    # 후보 경로: Windows(맑은 고딕), Noto Sans CJK 등
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",           # 맑은 고딕 Regular
        r"C:\Windows\Fonts\malgunbd.ttf",         # 맑은 고딕 Bold
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux Noto
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None  # 없으면 PIL 기본 폰트 사용(한글 깨질 수 있음)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="생성할 개념 (예: 피부의 단면)")
    parser.add_argument("--out", default="tactile-ai/data/sample.png", help="저장 경로")
    parser.add_argument("--size", type=int, default=36, help="폰트 크기")
    args = parser.parse_args()

    # 캔버스
    W, H = 1000, 500
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # 폰트 로드
    font_path = find_korean_font()
    if font_path:
        font = ImageFont.truetype(font_path, args.size)
    else:
        # 경고: 한글이 깨질 수 있음
        font = ImageFont.load_default()

    # 텍스트 준비 + 줄바꿈
    header = "[DEMO OUTPUT]"
    body = f"Prompt: {args.prompt}"
    wrapped = textwrap.fill(body, width=28)  # 적당히 줄바꿈
    text = f"{header}\n{wrapped}"

    # 중앙 정렬 배치
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="left")
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = (W - tw) // 2
    y = (H - th) // 2

    draw.multiline_text((x, y), text, font=font, fill="black", spacing=8, align="left")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img.save(args.out)
    print(f"✅ Saved: {args.out}  (font: {font_path or 'PIL default'})")

if __name__ == "__main__":
    main()

