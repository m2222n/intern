# tactile-ai/evaluation/evaluate_tactile.py
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

def evaluate_ssim(img1_path, img2_path):
    """
    두 이미지 간의 SSIM(Structural Similarity Index)을 계산.
    1에 가까울수록 유사도가 높다.
    """
    img1 = np.array(Image.open(img1_path).convert("L"))
    img2 = np.array(Image.open(img2_path).convert("L"))

    score, diff = ssim(img1, img2, full=True)
    print(f"✅ SSIM 유사도 점수: {score:.4f}")
    return score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="기준 이미지 경로")
    parser.add_argument("--gen", required=True, help="생성된 이미지 경로")
    args = parser.parse_args()

    evaluate_ssim(args.ref, args.gen)
