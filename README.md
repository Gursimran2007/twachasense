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

## How it works

```
photo ─▶ centre ROI (224²) ─┬─▶ SkinMetrics.analyse()  → tone, undertone, scores  (offline CV)
                            └─▶ 24-d skin feature vector ─▶ KNN few-shot classifier → condition (ML)
                                        │   (HSV histogram + CV metrics + Lab tone, all in-browser)
                                        └─▶ Recommend.build() → ranked concerns → AM/PM routine → brand SKUs
```

| Piece | File | What |
|---|---|---|
| Skin CV metrics | `js/skin_metrics.js` | skin mask, Lab tone/undertone, oiliness/redness/pigment/texture/fungal-patch |
| ML condition model | `js/classifier.js` | 24-d skin feature vector + trainable weighted KNN, saved to `localStorage` (100% offline, no CDN) |
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
