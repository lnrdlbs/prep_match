"""Restore the large data files that are stored gzip-compressed in this repository.

Usage (from the repository root):
    python decompress_data.py

Each <file>.gz is decompressed next to itself as <file> and verified against the
SHA-256 checksum of the original. Files that already exist with the correct
checksum are skipped. Only the Python standard library is used.
"""
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# SHA-256 checksums of the original (decompressed) files.
FILES = {
    "kb.csv": "51999d57503153f5ca46e38d39e741fbe18b1a2b504f8b93094da23c7c7a3656",
    "real_world_data/f_mnist_10k.pickle": "eac1d39030332e1fda714b434b15d09e9db7a408e0177453a85f85345bd04f5d",
    "real_world_data/f_mnist_20k.pickle": "9e49ab32ba2f65ee1967ecadca0daeb09fe5614f24454b015dc5c01c804190f2",
    "real_world_data/usps.pickle": "e98f01c3c3046df040df617321bee6f10ff728935299580a642bdd40a643a84d",
    "tpe_baseline/online_phase_prep4clust/f_mnist_10k.csv": "7562a627a15a348c4a302b81ffdffe2746ae1c8d55638146b57ff6df210ea464",
    "tpe_baseline/online_phase_prep4clust/f_mnist_20k.csv": "f4f76388b50b2b7ca4456593b0f2b9aa88516dd1cc96ebe0fbf1854c828973fe",
    "tpe_baseline/online_phase_prep4clust/usps.csv": "25e6dfcbd99248631a78c7cbd123493fe669bc382a8233aff7651d4da60e1a13",
    "tpot_baseline/own_datasets/f_mnist_10k.csv": "708f3477474355634eb0a97b407d566f14c061d8e0a64ba8cc6df0daba4a0dde",
    "tpot_baseline/own_datasets/f_mnist_20k.csv": "f7f6995526670b7aedc33cc098c5d899690cf82f9268c497c37312155df6ca5f",
    "tpot_baseline/own_datasets/usps.csv": "1a727bd3cc0d59b31e69739242b0db3f5ab72ee07f214241f1087148ae50ca3c",
    "tpot_baseline/poac_sv7.joblib": "d2954dc432df32bad1d5c9dfd51b15c82a9e5fc726cb35755b7ab7e0de449bbe",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore(rel: str, expected: str) -> bool:
    target = ROOT / rel
    archive = target.with_name(target.name + ".gz")

    if target.exists() and sha256(target) == expected:
        print(f"already present   {rel}")
        return True
    if not archive.exists():
        print(f"MISSING ARCHIVE   {rel}.gz")
        return False

    tmp = target.with_name(target.name + ".part")
    try:
        with gzip.open(archive, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        if sha256(tmp) != expected:
            print(f"CHECKSUM MISMATCH {rel}")
            return False
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)

    print(f"decompressed      {rel}")
    return True


def main() -> int:
    failed = [rel for rel, expected in FILES.items() if not restore(rel, expected)]
    if failed:
        print(f"\n{len(failed)} of {len(FILES)} files could not be restored.", file=sys.stderr)
        return 1
    print(f"\nAll {len(FILES)} files restored and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
