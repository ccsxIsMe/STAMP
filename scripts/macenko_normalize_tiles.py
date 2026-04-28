"""
macenko_normalize_tiles.py
--------------------------
Apply Macenko stain normalization to cached tiles before feature extraction.

Supports two cache layouts:
1. Plain image files under a directory tree.
2. STAMP tile-cache zip files produced by preprocess caching.

Typical usage with STAMP zip caches:
    python scripts/macenko_normalize_tiles.py \
        --src_cache /data3/chensx/STAMP/outputs/cache/tcga \
        --dst_cache /data3/chensx/STAMP/outputs/cache/tcga_macenko \
        --target_img /data3/chensx/STAMP/scripts/macenko_target.jpg \
        --workers 8

Then run STAMP preprocess again, pointing `cache_dir` to `tcga_macenko`, so
feature extraction reuses the normalized cached tiles instead of raw tiles.

Requirements:
    pip install staintools
or:
    pip install torchstain
"""

from __future__ import annotations

import argparse
import io
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image


def _get_macenko_normalizer(target_path: Path):
    """Return a fitted Macenko normalizer plus backend label."""
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
        from torchvision import transforms

        target_img = Image.open(target_path).convert("RGB")
        to_tensor255 = transforms.Compose([transforms.ToTensor(), lambda x: x * 255])
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        normalizer.fit(to_tensor255(target_img))
        return normalizer, "torchstain"
    except ImportError:
        pass

    raise ImportError(
        "Neither staintools nor torchstain is installed.\n"
        "Run: pip install staintools   or   pip install torchstain"
    )


def _normalize_pil_image(img: Image.Image, normalizer, backend: str) -> np.ndarray:
    if backend == "staintools":
        import staintools

        arr = np.array(img.convert("RGB"))
        standardized = staintools.LuminosityStandardizer.standardize(arr)
        return normalizer.transform(standardized)

    from torchvision import transforms

    to_tensor255 = transforms.Compose([transforms.ToTensor(), lambda x: x * 255])
    norm_t, _, _ = normalizer.normalize(to_tensor255(img.convert("RGB")), stains=False)
    return norm_t.permute(1, 2, 0).clamp(0, 255).byte().numpy()


def normalize_tile_file(src: Path, dst: Path, normalizer, backend: str) -> bool:
    """Normalize one plain image file; return True on success."""
    try:
        img = Image.open(src).convert("RGB")
        norm = _normalize_pil_image(img, normalizer, backend)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(norm).save(dst)
        return True
    except Exception:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return False


def normalize_cache_zip(src_zip: Path, dst_zip: Path, normalizer, backend: str) -> tuple[int, int]:
    """
    Normalize one STAMP tile-cache zip.

    Returns:
        (num_normalized, num_fallback_copies)
    """
    normalized = 0
    fallback = 0
    dst_zip.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(src_zip, "r") as zin, ZipFile(dst_zip, "w", compression=ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            lower_name = info.filename.lower()

            if info.is_dir():
                zout.writestr(info, data)
                continue

            if lower_name.endswith((".jpg", ".jpeg", ".png")):
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    norm = _normalize_pil_image(img, normalizer, backend)
                    out_buf = io.BytesIO()
                    fmt = "PNG" if lower_name.endswith(".png") else "JPEG"
                    Image.fromarray(norm).save(out_buf, format=fmt)
                    zout.writestr(info.filename, out_buf.getvalue())
                    normalized += 1
                except Exception:
                    zout.writestr(info.filename, data)
                    fallback += 1
            else:
                # Preserve metadata files inside the STAMP cache zip unchanged.
                zout.writestr(info.filename, data)

    return normalized, fallback


def _find_plain_tiles(src_cache: Path) -> list[Path]:
    return list(src_cache.rglob("*.jpg")) + list(src_cache.rglob("*.jpeg")) + list(src_cache.rglob("*.png"))


def _find_cache_zips(src_cache: Path) -> list[Path]:
    return list(src_cache.rglob("*.zip"))


def _run_plain_mode(src_cache: Path, dst_cache: Path, normalizer, backend: str, workers: int) -> None:
    tiles = _find_plain_tiles(src_cache)
    print(f"Detected plain tile cache with {len(tiles)} image files")

    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for src in tiles:
            dst = dst_cache / src.relative_to(src_cache)
            futures[pool.submit(normalize_tile_file, src, dst, normalizer, backend)] = src

        for i, fut in enumerate(as_completed(futures), 1):
            if not fut.result():
                failed += 1
            if i % 10000 == 0:
                print(f"  {i}/{len(tiles)} done  ({failed} fallback copies)")

    print(
        f"\nDone. {len(tiles) - failed}/{len(tiles)} normalized, "
        f"{failed} copied as-is (normalization failed)."
    )
    print(f"Output: {dst_cache}")


def _run_zip_mode(src_cache: Path, dst_cache: Path, normalizer, backend: str, workers: int) -> None:
    cache_zips = _find_cache_zips(src_cache)
    print(f"Detected STAMP zip cache with {len(cache_zips)} slide archives")

    total_normalized = 0
    total_fallback = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for src_zip in cache_zips:
            dst_zip = dst_cache / src_zip.relative_to(src_cache)
            futures[pool.submit(normalize_cache_zip, src_zip, dst_zip, normalizer, backend)] = src_zip

        for i, fut in enumerate(as_completed(futures), 1):
            normalized, fallback = fut.result()
            total_normalized += normalized
            total_fallback += fallback
            if i % 100 == 0:
                print(
                    f"  {i}/{len(cache_zips)} archives done  "
                    f"({total_normalized} normalized tiles, {total_fallback} fallback copies)"
                )

    print(
        f"\nDone. {total_normalized} tiles normalized, "
        f"{total_fallback} copied as-is (normalization failed)."
    )
    print(f"Output: {dst_cache}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_cache", type=Path, required=True)
    parser.add_argument("--dst_cache", type=Path, required=True)
    parser.add_argument(
        "--target_img",
        type=Path,
        required=True,
        help="Reference H&E tile to normalize toward (TCGA style)",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    normalizer, backend = _get_macenko_normalizer(args.target_img)
    print(f"Using {backend} Macenko normalizer")

    cache_zips = _find_cache_zips(args.src_cache)
    plain_tiles = _find_plain_tiles(args.src_cache) if not cache_zips else []

    if cache_zips:
        _run_zip_mode(
            src_cache=args.src_cache,
            dst_cache=args.dst_cache,
            normalizer=normalizer,
            backend=backend,
            workers=args.workers,
        )
    elif plain_tiles:
        _run_plain_mode(
            src_cache=args.src_cache,
            dst_cache=args.dst_cache,
            normalizer=normalizer,
            backend=backend,
            workers=args.workers,
        )
    else:
        raise FileNotFoundError(
            f"No supported cache content found under {args.src_cache}. "
            "Expected either .zip archives or image tiles."
        )


if __name__ == "__main__":
    main()
