#!/usr/bin/env python3
"""
DEV-ONLY: extract ONE disk-bounded SHARD of the FULL DermNet balanced_dataset so
the whole 262k-image set can be covered without ever holding it all on disk.

Usage:  _shard_extract.py <K> <k>
  K = total number of shards, k = which shard (0..K-1) to materialise now.

Deterministic: every invocation rebuilds the identical global ordered list
(member, cos, refer, split) from the zip's namelist (no image bytes read for the
ordering), then materialises only the contiguous slice for shard k into
../_ds_cond/shard/  with a manifest.json = {items:[{path,cos,refer,split}]}.
The shard dir is CLEARED first, so disk only ever holds one shard (~1GB).
build_cond_full.html feature-extracts each shard and accumulates the 49-d
descriptors (tiny), so training covers 100% of images. Labels mirror
_extract_sample.py exactly.
"""
import zipfile, os, json, random, re, sys, shutil

ZIP = os.path.expanduser("~/Downloads/archive.zip")
SHARD_DIR = os.path.join(os.path.dirname(__file__), "..", "_ds_cond", "shard")
PREFIX = "balanced_dataset/balanced_dataset/"
SEED = 13
TEST_FRAC = 0.20
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# DermNet class -> (cos, refer); cos=-1 excluded from the softmax. Mirrors
# _extract_sample.py exactly.
LABELS = {
    "Heathy": (0, 0),
    "Acne And Rosacea Photos": (1, 0),
    "Fu Ringworm": (2, 1),
    "Fu Nail Fungus": (2, 1),
    "Fu Athlete Foot": (2, 1),
    "Nail Fungus And Other Nail Disease": (2, 1),
    "Tinea Ringworm Candidiasis And Other Fungal Infections": (2, 1),
    "Eczema Photos": (3, 1),
    "Atopic Dermatitis Photos": (3, 1),
    "Light Diseases And Disorders Of Pigmentation": (4, 0),
    "Psoriasis Pictures Lichen Planus And Related Diseases": (5, 1),
    "Melanoma Skin Cancer Nevi And Moles": (-1, 1),
    "Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions": (-1, 1),
    "Malignant": (-1, 1),
    "Benign": (-1, 0),
    "Seborrheic Keratoses And Other Benign Tumors": (-1, 0),
    "Bullous Disease Photos": (-1, 1),
    "Lupus And Other Connective Tissue Diseases": (-1, 1),
    "Vasculitis Photos": (-1, 1),
    "Vascular Tumors": (-1, 1),
    "Systemic Disease": (-1, 1),
    "Urticaria Hives": (-1, 1),
    "Hair Loss Photos Alopecia And Other Hair Diseases": (-1, 1),
    "Poison Ivy Photos And Other Contact Dermatitis": (-1, 1),
    "Cellulitis Impetigo And Other Bacterial Infections": (-1, 1),
    "Ba Impetigo": (-1, 1),
    "Ba  Cellulitis": (-1, 1),
    "Vi Chickenpox": (-1, 1),
    "Vi Shingles": (-1, 1),
    "Pa Cutaneous Larva Migrans": (-1, 1),
    "Herpes Hpv And Other Stds Photos": (-1, 1),
    "Warts Molluscum And Other Viral Infections": (-1, 1),
    "Scabies Lyme Disease And Other Infestations And Bites": (-1, 1),
    "Exanthems And Drug Eruptions": (-1, 1),
    "Rashes": (-1, 1),
}


def build_master(zf):
    """Deterministic global ordered list of (member, cos, refer, split)."""
    by_class = {}
    for n in zf.namelist():
        if not n.startswith(PREFIX) or n.endswith("/"):
            continue
        rest = n[len(PREFIX):]
        parts = rest.split("/")
        if len(parts) < 2:
            continue
        cls = parts[0]
        if cls not in LABELS or not n.lower().endswith(IMG_EXT):
            continue
        by_class.setdefault(cls, []).append(n)

    master = []
    for cls in sorted(by_class):
        cos, refer = LABELS[cls]
        members = sorted(by_class[cls])                 # stable base order
        random.Random(SEED).shuffle(members)            # per-class deterministic shuffle
        ntest = int(len(members) * TEST_FRAC)
        for i, m in enumerate(members):
            split = "test" if i < ntest else "train"
            master.append((m, cos, refer, split))
    return master


def main():
    if len(sys.argv) != 3:
        print("usage: _shard_extract.py <K> <k>"); sys.exit(1)
    K, k = int(sys.argv[1]), int(sys.argv[2])
    zf = zipfile.ZipFile(ZIP)
    master = build_master(zf)
    N = len(master)
    chunk = (N + K - 1) // K
    lo, hi = k * chunk, min(N, (k + 1) * chunk)

    # clear + recreate the single shard dir (bounds disk to one shard)
    if os.path.isdir(SHARD_DIR):
        shutil.rmtree(SHARD_DIR)
    os.makedirs(SHARD_DIR, exist_ok=True)

    items = []
    for gi in range(lo, hi):
        member, cos, refer, split = master[gi]
        try:
            data = zf.read(member)
        except Exception:
            continue
        fn = f"{gi}.jpg"
        with open(os.path.join(SHARD_DIR, fn), "wb") as fo:
            fo.write(data)
        items.append({"path": fn, "cos": cos, "refer": refer, "split": split})

    with open(os.path.join(SHARD_DIR, "manifest.json"), "w") as fo:
        json.dump({"shard": k, "K": K, "total": N, "items": items}, fo)
    print(f"shard {k}/{K}: images {lo}..{hi} ({len(items)} written) of {N}")


if __name__ == "__main__":
    main()
