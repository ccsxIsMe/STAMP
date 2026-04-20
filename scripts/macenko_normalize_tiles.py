"""
macenko_normalize_tiles.py
--------------------------
Apply Macenko stain normalization to cached JPG tiles before feature extraction.

Strategy: read tiles from existing cache (ourdata or tcga), write normalized
tiles to a new cache directory, then run STAMP preprocess with the new cache.

Usage:
    python scripts/macenko_normalize_tiles.py \
        --src_cache /data3/chensx/STAMP/outputs/cache/ourdata \
        --dst_cache /data3/chensx/STAMP/outputs/cache/ourdata_macenko \
        --target_img /data3/chensx/STAMP/scripts/macenko_target.jpg \
        --workers 8

The target image should be a representative H&E tile from the REFERENCE domain
(e.g. a TCGA tile) so that Ourdata tiles are normalized toward TCGA style,
reducing the stain domain gap.

Requirements:  pip install staintools  (or torchstain)
"""

import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image


def _get_macenko_normalizer(target_path: Path):
    """Return a fitted Macenko normalizer (staintools or torchstain)."""
    try:
        import staintools
        target = staintools.read_image(str(target_path))
        normalizer = staintools.StainNormalizer(method="macenko")
        normalizer.fit(target)
        return normalizer, "staintools"
    except ImportError:
        pass

    try:
        import torchstain
        import torch
        from torchvision import transforms
        target_img = Image.open(target_path).convert("RGB")
        T = transforms.Compose([transforms.ToTensor(), lambda x: x * 255])
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        normalizer.fit(T(target_img))
        return normalizer, "torchstain"
    except ImportError:
        pass

    raise ImportError(
        "Neither staintools nor torchstain is installed.\n"
        "Run: pip install staintools   or   pip install torchstain"
    )


def normalize_tile(src: Path, dst: Path, normalizer, backend: str) -> bool:
    """Normalize one tile; return True on success, False if skipped."""
    try:
        if backend == "staintools":
            import staintools
            img = staintools.read_image(str(src))
            norm = normalizer.transform(img)
            Image.fromarray(norm).save(dst)
        else:
            import torch
            from torchvision import transforms
            img = Image.open(src).convert("RGB")
            T = transforms.Compose([transforms.ToTensor(), lambda x: x * 255])
            norm_t, _, _ = normalizer.normalize(T(img), stains=False)
            norm_img = (norm_t.permute(1, 2, 0).clamp(0, 255).byte().numpy())
            Image.fromarray(norm_img).save(dst)
        return True
    except Exception:
        # Copy original tile if normalization fails (e.g. low-contrast background tile)
        shutil.copy2(src, dst)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_cache", type=Path, required=True)
    parser.add_argument("--dst_cache", type=Path, required=True)
    parser.add_argument("--target_img", type=Path, required=True,
                        help="Reference H&E tile to normalize toward (TCGA style)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    normalizer, backend = _get_macenko_normalizer(args.target_img)
    print(f"Using {backend} Macenko normalizer")

    tiles = list(args.src_cache.rglob("*.jpg")) + list(args.src_cache.rglob("*.png"))
    print(f"Found {len(tiles)} tiles in {args.src_cache}")

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for src in tiles:
            dst = args.dst_cache / src.relative_to(args.src_cache)
            dst.parent.mkdir(parents=True, exist_ok=True)
            futures[pool.submit(normalize_tile, src, dst, normalizer, backend)] = src

        for i, fut in enumerate(as_completed(futures), 1):
            if not fut.result():
                failed += 1
            if i % 10000 == 0:
                print(f"  {i}/{len(tiles)} done  ({failed} fallback copies)")

    print(f"\nDone. {len(tiles)-failed}/{len(tiles)} normalized, "
          f"{failed} copied as-is (normalization failed).")
    print(f"Output: {args.dst_cache}")


if __name__ == "__main__":
    main()
