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
from threading import Lock
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image


_FIRST_ERROR_LOCK = Lock()
_FIRST_ERROR_MESSAGE: str | None = None
_FIRST_ERROR_PRINTED = False


def _record_first_error(message: str) -> None:
    global _FIRST_ERROR_MESSAGE, _FIRST_ERROR_PRINTED
    with _FIRST_ERROR_LOCK:
        if _FIRST_ERROR_MESSAGE is None:
            _FIRST_ERROR_MESSAGE = message
        if not _FIRST_ERROR_PRINTED:
            print(f"First normalization error: {_FIRST_ERROR_MESSAGE}", flush=True)
            _FIRST_ERROR_PRINTED = True


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


def _extract_auto_target_image(src_cache: Path) -> tuple[Image.Image, str]:
    """
    Pick one tile from cache as the normalization reference.

    Returns:
        (image, description)
    """
    cache_zips = _find_cache_zips(src_cache)
    if cache_zips:
        for zip_path in sorted(cache_zips):
            with ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    lower_name = name.lower()
                    if lower_name.endswith((".jpg", ".jpeg", ".png")):
                        data = zf.read(name)
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        return img, f"{zip_path.name}:{name}"

    plain_tiles = _find_plain_tiles(src_cache)
    if plain_tiles:
        tile_path = sorted(plain_tiles)[0]
        return Image.open(tile_path).convert("RGB"), str(tile_path)

    raise FileNotFoundError(
        f"No usable cache tile found under {src_cache}. "
        "Expected either STAMP .zip caches or image tiles."
    )


def _get_macenko_normalizer_from_image(target_img: Image.Image):
    """Return a fitted Macenko normalizer from an in-memory PIL image."""
    try:
        import staintools

        target = np.array(target_img.convert("RGB"))
        target = staintools.LuminosityStandardizer.standardize(target)
        normalizer = staintools.StainNormalizer(method="macenko")
        normalizer.fit(target)
        return normalizer, "staintools"
    except ImportError:
        pass

    try:
        import torchstain
        from torchvision import transforms

        to_tensor255 = transforms.Compose([transforms.ToTensor(), lambda x: x * 255])
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        normalizer.fit(to_tensor255(target_img.convert("RGB")))
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
        return _coerce_image_array(normalizer.transform(standardized))

    from torchvision import transforms

    to_tensor255 = transforms.Compose([transforms.ToTensor(), lambda x: x * 255])
    input_tensor = to_tensor255(img.convert("RGB"))

    # torchstain versions differ in normalize() return signatures.
    try:
        normalized = normalizer.normalize(input_tensor, stains=False)
    except TypeError:
        normalized = normalizer.normalize(input_tensor)

    if isinstance(normalized, tuple):
        norm_t = normalized[0]
    else:
        norm_t = normalized

    return _coerce_image_array(norm_t)


def _coerce_image_array(arr_like) -> np.ndarray:
    """
    Convert torchstain/staintools output to a PIL-safe H x W x 3 uint8 array.

    Handles common variants:
    - torch.Tensor or np.ndarray
    - batched or unbatched output
    - channel-first or channel-last layout
    - grayscale fallback
    """
    try:
        import torch

        if isinstance(arr_like, torch.Tensor):
            arr = arr_like.detach().cpu().numpy()
        else:
            arr = np.asarray(arr_like)
    except Exception:
        arr = np.asarray(arr_like)

    arr = np.squeeze(arr)

    if arr.ndim == 0:
        raise TypeError(f"Unexpected scalar image output: shape={arr.shape}")

    if arr.ndim == 1:
        raise TypeError(f"Unexpected 1D image output: shape={arr.shape}")

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        # Channel-first cases: CxHxW
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        # Still single-channel after transpose/squeeze.
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        # Rare case: HxWx224 etc. Try channel-first reinterpretation if last dim is invalid.
        elif arr.shape[-1] not in (3, 4) and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
    else:
        raise TypeError(f"Unexpected image rank: shape={arr.shape}")

    if arr.ndim != 3:
        raise TypeError(f"Failed to coerce image to rank-3 array: shape={arr.shape}")

    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] != 3:
        raise TypeError(f"Failed to coerce image to 3 channels: shape={arr.shape}, dtype={arr.dtype}")

    return arr


def normalize_tile_file(src: Path, dst: Path, normalizer, backend: str) -> bool:
    """Normalize one plain image file; return True on success."""
    try:
        img = Image.open(src).convert("RGB")
        norm = _normalize_pil_image(img, normalizer, backend)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(norm).save(dst)
        return True
    except Exception as e:
        _record_first_error(f"{src}: {type(e).__name__}: {e}")
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
                except Exception as e:
                    _record_first_error(
                        f"{src_zip}:{info.filename}: {type(e).__name__}: {e}"
                    )
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
    if _FIRST_ERROR_MESSAGE is not None:
        print(f"First normalization error: {_FIRST_ERROR_MESSAGE}")
    print(f"Output: {dst_cache}")


def _run_zip_mode(
    src_cache: Path,
    dst_cache: Path,
    normalizer,
    backend: str,
    workers: int,
    max_archives: int | None = None,
) -> None:
    cache_zips = _find_cache_zips(src_cache)
    if max_archives is not None:
        cache_zips = cache_zips[:max_archives]
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
    if _FIRST_ERROR_MESSAGE is not None:
        print(f"First normalization error: {_FIRST_ERROR_MESSAGE}")
    print(f"Output: {dst_cache}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_cache", type=Path, required=True)
    parser.add_argument("--dst_cache", type=Path, required=True)
    parser.add_argument(
        "--target_img",
        type=Path,
        default=None,
        help="Optional reference H&E tile to normalize toward. If omitted or missing, auto-pick one tile from src_cache.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max_archives",
        type=int,
        default=None,
        help="Optional debug limit: only process the first N cache zip archives.",
    )
    args = parser.parse_args()

    if args.target_img is not None and args.target_img.exists():
        print(f"Using explicit target image: {args.target_img}")
        normalizer, backend = _get_macenko_normalizer(args.target_img)
    else:
        if args.target_img is not None:
            print(f"Target image not found: {args.target_img}")
        target_img, target_desc = _extract_auto_target_image(args.src_cache)
        print(f"Auto-selected target tile: {target_desc}")
        normalizer, backend = _get_macenko_normalizer_from_image(target_img)
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
            max_archives=args.max_archives,
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
