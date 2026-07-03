# TwachaSense — AI Skin Intelligence for Indian Skin

A camera-first **skin-analysis engine** that reads a face/skin photo and returns a
structured skin profile — tone, undertone, oiliness, redness, pigmentation,
texture — plus a **learned condition screen** and a personalised routine mapped to
a brand's own products.

> **Screening & cosmetic guidance only — not a medical diagnosis.** Persistent,
> spreading, painful or bleeding conditions should be seen by a dermatologist.

## Why this is defensible (the moat)

Western pretrained dermatology/beauty models are trained on light skin and
**degrade badly on Indian skin tones**. TwachaSense's core is two things that are
hard to copy:

1. **Calibrated CV metrics** that are skin-tone-aware (YCbCr skin mask, Lab-space
   tone/undertone, erythema & specular indices) — see `js/skin_metrics.js`.
2. **Few-shot, on-your-own-data learning.** A brand teaches the condition model on
   *its own* Indian-skin customer photos, in the browser, no Python/GPU/cloud —
   see `js/classifier.js`. Every scan a brand collects makes its model better and
   less clonable. That data flywheel is the long-term moat.

## The business — sell to D2C beauty brands

Not a consumer app: an **embeddable skin-intelligence widget** brands license.

- **Personalisation → conversion.** Customer scans → skin profile → the brand's own
  SKUs recommended for AM/PM. Personalised routines lift AOV & conversion.
- **White-label.** A brand supplies a catalog JSON mapping ingredient slots to its
  real products (`Recommend.SAMPLE_CATALOG` in `js/recommend.js`); the widget
  instantly recommends *their* range.
- **Data flywheel.** Opt-in scans → a proprietary Indian-skin dataset per brand →
  a better model competitors can't match.
- **Revenue:** SaaS license per brand + usage tier; later, aggregated
  (anonymised) skin-trend insights for R&D.

## Run

Pure static site — no build step.

```bash
cd twachasense
python3 -m http.server 8000        # any static server
# open http://localhost:8000
```

Camera needs `localhost` or HTTPS. If offline, the ML model won't load and the app
gracefully falls back to **metrics + recommendations only** (still fully useful).

### Try it
1. **Start camera** → **Capture** (or **Upload** a photo). You get a skin profile +
   routine immediately (pure CV, no ML needed).
2. **Seed Indian-skin baseline** (section 3) pre-trains the condition model on a
   tone-diverse synthetic dataset so screening works out-of-the-box. Then add your
   own real examples per class and it keeps improving. Model persists in
   `localStorage`.
3. **Copy shareable report** for the text summary.

### Tone-diverse baseline (`js/seed.js`)
Since real dermatology photos can't ship here, the baseline is **procedurally
synthesised across 50 realistic Indian skin tones × 6 conditions**, each rendered
in **2 severity + lighting variants** (600 patches). Tones are sampled with
**correlated RGB channels** (real skin: G≈0.72–0.80·R, B≈0.55–0.66·R) spanning
wheatish → deep brown with warm/neutral/cool undertones — independent random RGB
produces impossible colours that break the redness metric. Each patch is
additionally augmented with exposure + white-balance variation so the model is
robust to real-world capture lighting, not just the ideal studio shot.

On a held-out set drawn from the same realistic tone distribution the seeded KNN
scores **~98% (Clear / Eczema / Pigmentation / Dryness ~100%, Acne 98%, Fungal
93–98%)**, stable across independent test seeds. The old Acne↔Clear confusion is
resolved by two discriminative CV cues: a tone-invariant **localised-erythema**
signal (kept inside the skin gamut so mild acne no longer reads as clear) and a
**light-desaturated-patch** signal that isolates fungal patches. This is a
**synthetic baseline** — a brand replaces/augments it with real customer photos,
which is the real moat.

### Real-data face detectors (`js/real_models.js`)
The condition screen is synthetic; the **Real-skin estimates** row is not. Three
attribute detectors — **Clarity**, **Dark circles**, **Wrinkles** — are trained on
**~4,900 real Indian-face photos** (5,000 sampled; grayscale / no-skin dropped)
with human attribute labels. The pipeline is deliberately honest:

- A shared 24-d **face-level descriptor** (`js/face_features.js`) is computed over
  the whole photo — skin fraction, Lab tone, oiliness/shine, texture in **four
  regional bands** (brow / eye / cheek / chin), an under-eye dark-band + chroma
  signal, edge density, and per-region luma ratios. The **exact same function**
  runs at training and inference time, so the feature distribution can never
  drift between train and serve.
- Per attribute, a small **MLP** (24 → 12 tanh → sigmoid, class-weighted BCE, L2,
  full-batch GD) is trained in-browser via `train/build.html` against a local
  `_ds/` dataset. Only the learned weights ship in `js/real_models.js` — **never
  the photos**. Inference stays 100% offline / pure-JS / no CDN.

Held-out AUC on a ~970-image test split: **Clarity 0.78, Wrinkles 0.77, Dark
circles 0.71** — honest, moderate signals surfaced as *estimates*, not diagnoses.
The MLP + regional features lifted every attribute over the earlier linear
logistic baseline (0.71 / 0.70 / 0.68), which was *underfitting* — the gain came
from model capacity, not more rows. Oiliness trained to ~0.61 AUC (still weak) so
it stays the **direct CV specular measurement** in `skin_metrics.js` rather than a
learned model. To retrain: open `train/build.html` against a local `_ds/` dataset
and paste the new bundle.

### Real-data condition model + derm-referral nudge (`js/cond_models.js`)
The shipped Condition screen is now a **real-data model**, not the synthetic KNN.
Two honest heads, both fed the **shared 49-d `CondFeatures` descriptor** (finer
HSV/Lab colour moments, texture statistics, and a **3×3 spatial grid** so the
model sees *where* an abnormality sits — the same function runs at train and serve
time, so features can't drift), trained on **~11k real dermatology photos**
(`train/build_cond.html`, weights only — never the photos):

- **Cosmetic 6-class softmax MLP** (Clear/Acne/Fungal/Eczema/Pigmentation/Dryness).
  Held-out **balanced-accuracy ~0.50** (6-class chance 0.167 → ~3×). It's a
  *screen*, surfaced as an estimate. The richer descriptor + capacity tuning
  lifted this from **0.42 → 0.50** balanced (the reused 24-d skin vector was
  *feature-limited* — it couldn't fit even the training set).
- **Non-diagnostic referral head** — a binary MLP (medical vs cosmetic,
  held-out **AUC ~0.70, recall ~0.68**) that raises a calm *"consider seeing a
  dermatologist"* nudge when the skin pattern looks unlike ordinary cosmetic
  concerns. It **never names a disease** — a deliberate liability/regulatory
  choice. Recall is favoured over specificity so genuine medical patterns are
  caught (at the cost of some over-nudging on cosmetic cases).

The few-shot KNN (`js/classifier.js`) still ships: when a brand teaches it on its
**own** labelled photos, that personalised signal takes over the Condition screen.
To retrain the real model: open `train/build_cond.html` against a local `_ds_cond/`
dataset and paste the new bundle.

## How it works

```
photo ─┬─ centre ROI (224²) ─┬─▶ SkinMetrics.analyse()  → tone, undertone, scores  (offline CV)
       │                     ├─▶ CondFeatures (49-d) ─▶ real MLPs → cosmetic condition + derm-referral nudge
       │                     │        (colour moments + texture + 3×3 spatial grid, trained on ~11k derm photos)
       │                     ├─▶ 24-d skin vector ─▶ KNN few-shot classifier → personalised condition (brand's own data)
       │                     └─▶ Recommend.build() → ranked concerns → AM/PM routine → brand SKUs
       └─ whole photo (160²) ──▶ FaceFeatures (24-d) ─▶ real MLP detectors → Clarity / Dark circles / Wrinkles
```

| Piece | File | What |
|---|---|---|
| Skin CV metrics | `js/skin_metrics.js` | skin mask, Lab tone/undertone, oiliness/redness/pigment/texture/fungal-patch |
| Condition descriptor | `js/cond_features.js` | 49-d condition vector (colour moments + texture + 3×3 spatial grid), shared by trainer + inference |
| Real condition model | `js/cond_models.js` | cosmetic 6-class softmax MLP + non-diagnostic derm-referral MLP, trained on ~11k real derm photos (weights only) |
| Condition trainer | `train/build_cond.html` | dev-only in-browser trainer over a local `_ds_cond/` dataset → paste-ready weight bundle |
| Few-shot personaliser | `js/classifier.js` | 24-d skin feature vector + trainable weighted KNN, saved to `localStorage`; overrides the Condition screen when a brand trains on its own photos |
| Face-level features | `js/face_features.js` | 24-d whole-photo descriptor (regional darkness/texture), shared by trainer + inference for train/serve parity |
| Real-data detectors | `js/real_models.js` | MLP Clarity/Dark-circles/Wrinkles trained on ~4,900 real Indian faces (weights only) |
| Detector trainer | `train/build.html` | dev-only in-browser trainer over a local `_ds/` dataset → paste-ready weight bundle |
| D2C recommender | `js/recommend.js` | concern ranking + ingredient→brand-SKU routine, white-label catalog |
| App / UI | `js/app.js`, `index.html`, `css/style.css` | camera, ROI, training UI, report |

## Scale path — swap in a real trained CNN

The few-shot KNN is the demo/bootstrap. For production, train a proper CNN on a
brand's accumulated labelled data and export to TF.js — the app is unchanged
because it only calls `Classifier.predict()`. A starter trainer lives in
`train/` (`train.py`): transfer-learning on MobileNet/EfficientNet →
`tensorflowjs_converter` → drop the model into the browser.

## Roadmap
- [ ] Multi-ROI (forehead T-zone vs cheek) for combination-skin precision
- [ ] Lighting/white-balance normalisation via a reference card for cross-photo consistency
- [ ] Progress tracking over time (before/after per concern)
- [ ] Real CNN trained per brand + `tfjs` export (see `train/`)
- [ ] Ingredient-conflict checker (e.g. retinol + acids) in the routine builder
