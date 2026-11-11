#!/usr/bin/env python3
"""Restore a SAPC SH-VQ Gaussian map to point_cloud.ply.

Supports extreme_saving.zip (including nested directory entries produced by
`zip -r`) and the original extreme_saving/ directory from vectree.py.

Only the SH attributes of non-salient Gaussians are vector-quantized. Latent
localization features are stored directly. No CUDA, PyTorch, or VQ model is
needed for decompression.

Requires: numpy.
Only run on map archives you trust: original metadata.npz contains a pickled
Python dictionary that NumPy must deserialize.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
import zipfile

import numpy as np

REQUIRED = (
    "metadata.npz", "xyz.npz", "other_attribute.npz", "non_vq_mask.npz",
    "non_vq_feats.npz", "codebook.npz", "vq_indexs.npz", "loc_feature.npz",
)


class MapReader:
    """Read NPZ arrays by name from a directory or ZIP, without extracting ZIP."""

    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.zf = None
        if input_path.is_file() and zipfile.is_zipfile(input_path):
            self.zf = zipfile.ZipFile(input_path, "r")
            # Match regardless of ZIP prefix, e.g., results/chess/extreme_saving/.
            names = {}
            for entry in self.zf.infolist():
                if entry.is_dir():
                    continue
                base = Path(entry.filename).name
                if base in REQUIRED:
                    if base in names:
                        raise ValueError(f"Duplicate ZIP member for {base}")
                    names[base] = entry.filename
            self.names = names
        elif input_path.is_dir():
            if all((input_path / name).is_file() for name in REQUIRED):
                self.dir = input_path
            elif all((input_path / "extreme_saving" / name).is_file() for name in REQUIRED):
                self.dir = input_path / "extreme_saving"
            else:
                raise FileNotFoundError(
                    f"Cannot find SAPC NPZ files under {input_path}"
                )
            self.names = {name: str(self.dir / name) for name in REQUIRED}
        else:
            raise FileNotFoundError(f"Not a ZIP archive or directory: {input_path}")

        missing = set(REQUIRED) - set(self.names)
        if missing:
            raise FileNotFoundError(f"Missing SAPC files: {', '.join(sorted(missing))}")

    def load(self, filename: str):
        if self.zf is None:
            source = self.names[filename]
        else:
            source = io.BytesIO(self.zf.read(self.names[filename]))
        # The original vectree.py stores metadata as a pickled dict.
        with np.load(source, allow_pickle=(filename == "metadata.npz")) as npz:
            key = "metadata" if filename == "metadata.npz" else "arr_0"
            if key not in npz:
                raise ValueError(f"{filename} is missing NPZ key {key!r}")
            array = npz[key]
            return array.item() if filename == "metadata.npz" else array

    def close(self):
        if self.zf is not None:
            self.zf.close()


def _decode_indices(packed: np.ndarray, count: int, num_codes: int) -> np.ndarray:
    """Undo np.packbits(dec2bin(indices, log2(codebook_size)))."""
    if num_codes < 2 or num_codes & (num_codes - 1):
        raise ValueError(f"Codebook size must be a power of two, got {num_codes}")
    bits_per_index = num_codes.bit_length() - 1
    bitstream = np.unpackbits(np.asarray(packed, dtype=np.uint8).ravel())
    required_bits = count * bits_per_index
    if bitstream.size < required_bits:
        raise ValueError(
            f"VQ index stream too short: {bitstream.size} bits, "
            f"expected at least {required_bits} bits"
        )
    # Up to 7 bits of padding are permitted at the end of np.packbits output.
    if bitstream.size != math.ceil(required_bits / 8) * 8:
        raise ValueError("Unexpected packed VQ index length")
    if count == 0:
        return np.empty((0,), dtype=np.int64)
    rows = bitstream[:required_bits].reshape(count, bits_per_index).astype(np.int64)
    weights = 1 << np.arange(bits_per_index - 1, -1, -1, dtype=np.int64)
    indices = rows @ weights
    if np.any(indices >= num_codes):
        raise ValueError("VQ index exceeds codebook size")
    return indices


def _load_gaussian_arrays(reader: MapReader):
    metadata = reader.load("metadata.npz")
    if not isinstance(metadata, dict):
        raise ValueError("metadata.npz must contain a metadata dictionary")
    n = int(metadata["input_pc_num"])
    sh_dim = int(metadata["codebook_dim"])
    codebook_size = int(metadata["codebook_size"])
    if n <= 0 or sh_dim < 3 or (sh_dim - 3) % 3:
        raise ValueError(f"Unexpected Gaussian count or SH dimension: N={n}, SH={sh_dim}")

    xyz = np.asarray(reader.load("xyz.npz"), dtype=np.float32)
    other = np.asarray(reader.load("other_attribute.npz"), dtype=np.float32)
    loc = np.asarray(reader.load("loc_feature.npz"), dtype=np.float32)
    if xyz.shape != (n, 3) or other.shape != (n, 8):
        raise ValueError(f"Invalid xyz/other shape: {xyz.shape}, {other.shape}")
    if loc.ndim != 2 or loc.shape[0] != n:
        raise ValueError(f"Invalid loc_feature shape: {loc.shape}")
    stored_dim = metadata.get("input_pc_dim")
    expected_dim = 6 + sh_dim + 8 + loc.shape[1]
    if stored_dim is not None and int(stored_dim) != expected_dim:
        raise ValueError(
            f"Input dimension mismatch: metadata={stored_dim}, reconstructed={expected_dim}"
        )

    mask_packed = np.asarray(reader.load("non_vq_mask.npz"), dtype=np.uint8).ravel()
    if mask_packed.size != math.ceil(n / 8):
        raise ValueError("Unexpected packed salient-mask size")
    keep_mask = np.unpackbits(mask_packed)[:n].astype(bool)
    n_keep = int(np.count_nonzero(keep_mask))
    n_vq = n - n_keep

    kept_sh = np.asarray(reader.load("non_vq_feats.npz"), dtype=np.float32)
    codebook = np.asarray(reader.load("codebook.npz"), dtype=np.float32)
    if codebook.ndim == 3 and codebook.shape[0] == 1:
        codebook = codebook[0]
    if kept_sh.shape != (n_keep, sh_dim):
        raise ValueError(f"Invalid preserved SH shape: {kept_sh.shape}")
    if codebook.shape != (codebook_size, sh_dim):
        raise ValueError(f"Invalid codebook shape: {codebook.shape}")

    indices = _decode_indices(reader.load("vq_indexs.npz"), n_vq, codebook_size)
    sh = np.empty((n, sh_dim), dtype=np.float32)
    sh[keep_mask] = kept_sh
    sh[~keep_mask] = codebook[indices]
    return xyz, sh, other, loc, n_keep, n_vq


def _attribute_names(sh_dim: int, loc_dim: int):
    # Match write_ply_data() from the supplied utils.py.
    return (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(sh_dim - 3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
        + [f"loc_{i}" for i in range(loc_dim)]
    )


def decompress_map(input_path: Path, save_path: Path, overwrite: bool = False) -> Path:
    input_path = Path(input_path)
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".ply":
        save_path = save_path / "point_cloud.ply"
    if save_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {save_path} (use --overwrite)")

    reader = MapReader(input_path)
    try:
        xyz, sh, other, loc, n_keep, n_vq = _load_gaussian_arrays(reader)
    finally:
        reader.close()
    n, sh_dim = sh.shape
    names = _attribute_names(sh_dim, loc.shape[1])
    vertex = np.empty(n, dtype=[(name, "<f4") for name in names])

    # Original utils.py reconstructs normals as zeros (they are not in SAPC).
    src = [
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    ]
    src.extend(sh[:, j] for j in range(sh_dim))
    src.extend(other[:, j] for j in range(8))
    src.extend(loc[:, j] for j in range(loc.shape[1]))
    for name, values in zip(names, src):
        vertex[name] = values

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Standard binary little-endian PLY, compatible with plyfile / 3DGS readers.
    header = [
        "ply", "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    header.extend(f"property float {name}" for name in names)
    header.append("end_header")
    with save_path.open("wb") as output:
        output.write(("\n".join(header) + "\n").encode("ascii"))
        vertex.tofile(output)
    print(f"Restored: {save_path}")
    print(f"Gaussians: {n:,} | salient preserved: {n_keep:,} | SH VQ: {n_vq:,}")
    print(f"SH dimension: {sh_dim} | localization feature dimension: {loc.shape[1]}")
    print(f"PLY size: {save_path.stat().st_size / (1024 ** 2):.2f} MiB")
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Decompress SAPC SH-VQ archive to point_cloud.ply"
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Path to extreme_saving.zip or an extreme_saving directory",
    )
    parser.add_argument(
        "--save_path", required=True,
        help="Output directory or an explicit point_cloud.ply path",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing PLY")
    args = parser.parse_args()
    decompress_map(Path(args.input_path), Path(args.save_path), args.overwrite)


if __name__ == "__main__":
    main()
