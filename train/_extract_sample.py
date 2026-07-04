#!/usr/bin/env python3
"""
DEV-ONLY: build a disk-capped, labelled sample of the DermNet balanced_dataset
zip into ../_ds_cond/SkinDisease/SkinDisease/{train,test}/<slug>/ + manifest.json
for train/build_cond.html. Extracts only the images we pick (never the full zip).
Emits cosmetic (cos 0-5 / -1) + referral (medical 1 / cosmetic 0 / -1) labels.
"""
import zipfile, os, io, json, random, re, sys

ZIP = os.path.expanduser("~/Downloads/archive.zip")
BASE = os.path.join(os.path.dirname(__file__), "..", "_ds_cond", "SkinDisease", "SkinDisease")
PREFIX = "balanced_dataset/balanced_dataset/"
TRAIN_PER, TEST_PER = 300, 80
SEED = 13

# cosmetic classes (index) — must match COS in build_cond.html
# ["Clear","Acne","Fungal","Eczema","Pigmentation","Dryness"]
# label map: DermNet class -> (cos, refer). cos=-1 excluded from softmax.
LABELS = {
    "Heathy":                                                          (0, 0),
    "Acne And Rosacea Photos":                                         (1, 0),
    "Fu Ringworm":                                                     (2, 1),
    "Fu Nail Fungus":                                                  (2, 1),
    "Fu Athlete Foot":                                                 (2, 1),
    "Nail Fungus And Other Nail Disease":                              (2, 1),
    "Tinea Ringworm Candidiasis And Other Fungal Infections":          (2, 1),
    "Eczema Photos":                                                   (3, 1),
    "Atopic Dermatitis Photos":                                        (3, 1),
    "Light Diseases And Disorders Of Pigmentation":                    (4, 0),
    "Psoriasis Pictures Lichen Planus And Related Diseases":           (5, 1),
    "Melanoma Skin Cancer Nevi And Moles":                             (-1, 1),
    "Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions": (-1, 1),
    "Malignant":                                                       (-1, 1),
    "Benign":                                                          (-1, 0),
    "Seborrheic Keratoses And Other Benign Tumors":                    (-1, 0),
    "Bullous Disease Photos":                                          (-1, 1),
    "Lupus And Other Connective Tissue Diseases":                      (-1, 1),
    "Vasculitis Photos":                                               (-1, 1),
    "Vascular Tumors":                                                 (-1, 1),
    "Systemic Disease":                                                (-1, 1),
    "Urticaria Hives":                                                 (-1, 1),
    "Hair Loss Photos Alopecia And Other Hair Diseases":              (-1, 1),
    "Poison Ivy Photos And Other Contact Dermatitis":                 (-1, 1),
    "Cellulitis Impetigo And Other Bacterial Infections":            (-1, 1),
    "Ba Impetigo":                                                     (-1, 1),
    "Ba  Cellulitis":                                                  (-1, 1),
    "Vi Chickenpox":                                                   (-1, 1),
    "Vi Shingles":                                                     (-1, 1),
    "Pa Cutaneous Larva Migrans":                                      (-1, 1),
    "Herpes Hpv And Other Stds Photos":                               (-1, 1),
    "Warts Molluscum And Other Viral Infections":                     (-1, 1),
    "Scabies Lyme Disease And Other Infestations And Bites":          (-1, 1),
    "Exanthems And Drug Eruptions":                                   (-1, 1),
    "Rashes":                                                          (-1, 1),
}

def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def main():
    random.seed(SEED)
    zf = zipfile.ZipFile(ZIP)
    # group image members by class
    by_class = {}
    for n in zf.namelist():
        if not n.startswith(PREFIX) or n.endswith("/"):
            continue
        rest = n[len(PREFIX):]
        parts = rest.split("/")
        if len(parts) < 2:
            continue
        cls = parts[0]
        if cls not in LABELS:
            continue
        if not n.lower().endswith(IMG_EXT):
            continue
        by_class.setdefault(cls, []).append(n)

    os.makedirs(BASE, exist_ok=True)
    # back up any previous manifest
    mpath = os.path.join(BASE, "manifest.json")
    if os.path.exists(mpath):
        os.replace(mpath, os.path.join(BASE, "manifest_prev.json"))

    train, test = [], []
    summary = []
    for cls, members in sorted(by_class.items()):
        cos, refer = LABELS[cls]
        random.shuffle(members)
        pick = members[: TRAIN_PER + TEST_PER]
        sg = slug(cls)
        for split, lo, hi in (("train", 0, TRAIN_PER), ("test", TRAIN_PER, TRAIN_PER + TEST_PER)):
            outdir = os.path.join(BASE, split, sg)
            os.makedirs(outdir, exist_ok=True)
            i = 0
            for m in pick[lo:hi]:
                try:
                    data = zf.read(m)
                except Exception:
                    continue
                fn = f"{i}.jpg"
                with open(os.path.join(outdir, fn), "wb") as fo:
                    fo.write(data)
                rec = {"path": f"{split}/{sg}/{fn}", "cos": cos, "refer": refer}
                (train if split == "train" else test).append(rec)
                i += 1
        summary.append((cls, cos, refer, len(pick)))
        print(f"  {cls:52s} cos={cos:2d} refer={refer} n={len(pick)}")

    manifest = {"train": train, "test": test,
                "classes": ["Clear","Acne","Fungal","Eczema","Pigmentation","Dryness"]}
    with open(mpath, "w") as fo:
        json.dump(manifest, fo)
    print(f"\nTOTAL train={len(train)} test={len(test)} classes={len(by_class)}")

if __name__ == "__main__":
    main()
