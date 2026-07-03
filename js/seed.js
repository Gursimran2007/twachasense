/*
 * seed.js — procedural, tone-diverse baseline dataset for the condition model.
 *
 * We cannot ship real dermatology photos, so this synthesises training patches
 * that span INDIAN SKIN DIVERSITY (Monk ~3 wheatish → ~10 deep) across warm /
 * neutral / cool undertones, and stamps each condition's visual signature on
 * top. It gives the KNN a tone-robust prior so the app works out-of-the-box.
 *
 * This is a synthetic BASELINE, not ground truth — a brand replaces/augments it
 * with real customer photos (that labelled real-skin data is the actual moat).
 */
(function (global) {
  "use strict";

  function clamp(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
  function mulberry(seed) { // tiny deterministic PRNG
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Realistic Indian skin-tone sampler. Channels are CORRELATED (real skin has
  // G≈0.72–0.80·R, B≈0.55–0.66·R) — independent random RGB produces impossible
  // colours that break the redness metric. Spans wheatish (light) → deep brown,
  // across warm / neutral / cool undertones. Shared by seeding AND evaluation so
  // the train/test tone distributions match.
  function sampleTone(rng) {
    const R = 90 + rng() * 150;              // ~90 (deep) .. 240 (wheatish)
    const gRatio = 0.72 + rng() * 0.08;      // green fraction of red
    const bRatio = 0.55 + rng() * 0.11;      // blue fraction of red
    const under = rng();                     // undertone tilt
    const warm = (under - 0.5) * 16;         // >0 warm(golden), <0 cool(pink)
    return [
      clamp(R + Math.max(0, -warm) * 0.3),
      clamp(R * gRatio + warm * 0.2),
      clamp(R * bRatio - warm),
    ];
  }

  function disc(d, W, cx, cy, r, fn) {
    for (let y = Math.max(0, cy - r); y < Math.min(W, cy + r); y++)
      for (let x = Math.max(0, cx - r); x < Math.min(W, cx + r); x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 > r * r) continue;
        const i = (y * W + x) * 4;
        fn(i);
      }
  }

  // Simulate varied capture lighting: exposure gain + mild warm/cool white
  // balance. Trains the model to be robust to real-world lighting, not just the
  // ideal studio shot. Kept mild so it never flips a condition's label.
  function applyLighting(d, rng) {
    const gain = 0.82 + rng() * 0.36;        // exposure 0.82 .. 1.18
    const rMul = 1 + (rng() - 0.5) * 0.05;   // ±2.5% warm/cool WB tilt (mild: must
    const bMul = 1 + (rng() - 0.5) * 0.05;   // not flip Clear→Acne via false redness)
    for (let i = 0; i < d.length; i += 4) {
      d[i]     = clamp(d[i]     * gain * rMul);
      d[i + 1] = clamp(d[i + 1] * gain);
      d[i + 2] = clamp(d[i + 2] * gain * bMul);
    }
  }

  // draw one 224² patch for a given base colour + condition.
  // sev ∈ [0..1] scales condition severity (mild → severe) for generalisation.
  function makePatch(base, condition, rng, sev) {
    if (sev == null) sev = 0.55 + rng() * 0.45;      // default: moderate..severe
    const W = 224;
    const c = document.createElement("canvas"); c.width = W; c.height = W;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(W, W); const d = img.data;

    // base fill + fine skin noise (texture level is itself a condition cue)
    const grain = condition === "Dryness" ? 14 + 12 * sev : condition === "Eczema" ? 8 + 6 * sev : 5;
    for (let i = 0; i < d.length; i += 4) {
      const n = (rng() - 0.5) * grain;
      d[i] = clamp(base[0] + n);
      d[i + 1] = clamp(base[1] + n * 0.9);
      d[i + 2] = clamp(base[2] + n * 0.8);
      d[i + 3] = 255;
    }
    const R = (i) => d[i], put = (i, r, g, b) => { d[i] = clamp(r); d[i + 1] = clamp(g); d[i + 2] = clamp(b); };

    // condition signatures (kept within skin gamut so features still compute).
    // Counts/intensity scale with severity → the model learns mild→severe.
    if (condition === "Acne") {                    // many small RED bumps (localised redness)
      // Boost kept mild so bumps stay INSIDE the YCbCr skin gamut (cr<=180) — an
      // over-saturated red bump reads as "not skin" and the erythema detector
      // silently drops it, making mild acne look Clear. Real pimples are only
      // slightly redder than surrounding skin, so this is also more realistic.
      const n = (18 + sev * 40) | 0;
      const boost = 16 + sev * 14;
      for (let k = 0; k < n; k++) {
        const cx = rng() * W, cy = rng() * W, r = 3 + rng() * 4;
        disc(d, W, cx | 0, cy | 0, r | 0, (i) => put(i, R(i) + boost, R(i + 1) - boost * 0.5, R(i + 2) - boost * 0.38));
      }
    } else if (condition === "Fungal") {           // large LIGHT desaturated patches
      const n = (4 + sev * 7) | 0;
      for (let k = 0; k < n; k++) {
        const cx = rng() * W, cy = rng() * W, r = 16 + rng() * 20;
        disc(d, W, cx | 0, cy | 0, r | 0, (i) => {
          const mean = (R(i) + R(i + 1) + R(i + 2)) / 3;
          put(i, R(i) * 0.4 + mean * 0.6 + 24, R(i + 1) * 0.4 + mean * 0.6 + 24, R(i + 2) * 0.4 + mean * 0.6 + 22);
        });
      }
    } else if (condition === "Eczema") {           // strongly RED + rough (the red separates it from Dryness)
      const boost = 26 + sev * 22;
      for (let i = 0; i < d.length; i += 4) put(i, R(i) + boost, R(i + 1) - boost * 0.4, R(i + 2) - boost * 0.3);
    } else if (condition === "Pigmentation") {      // many DARK spots / melasma
      const n = (8 + sev * 16) | 0;
      const dark = 0.62 - sev * 0.16;              // deeper spots at higher severity
      for (let k = 0; k < n; k++) {
        const cx = rng() * W, cy = rng() * W, r = 6 + rng() * 10;
        disc(d, W, cx | 0, cy | 0, r | 0, (i) => put(i, R(i) * dark, R(i + 1) * dark, R(i + 2) * (dark + 0.02)));
      }
    } else if (condition === "Dryness") {           // LIGHTER + flaky, deliberately NOT red
      const amp = 10 + sev * 16;
      for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const flake = ((x * 7 + y * 3) % 5 < 2 ? 1 : -1) * (amp + rng() * 10);
        // green/blue raised more than red -> lowers a* so redness stays low
        put(i, R(i) + 10 + flake, R(i + 1) + 20 + flake, R(i + 2) + 18 + flake);
      }
    }
    // "Clear" = smooth base only

    applyLighting(d, rng);
    ctx.putImageData(img, 0, 0);
    return c;
  }

  const IDX = { Clear: 0, Acne: 1, Fungal: 2, Eczema: 3, Pigmentation: 4, Dryness: 5 };

  // Train the classifier across a dense sweep of realistic Indian tones ×
  // conditions × severities × lighting. onProgress(done, total) for UI feedback.
  const TONES_PER_CLASS = 50;   // tones
  const VARIANTS = 2;           // severity/lighting variants per tone
  function seed(Classifier, onProgress) {
    const conditions = Object.keys(IDX);
    const rng = mulberry(1234);
    const total = TONES_PER_CLASS * VARIANTS * conditions.length;
    let done = 0;
    for (let t = 0; t < TONES_PER_CLASS; t++)
      for (let v = 0; v < VARIANTS; v++)
        for (const cond of conditions) {
          const base = sampleTone(rng);
          const sev = 0.5 + rng() * 0.5;         // mild → severe coverage
          Classifier.addExample(makePatch(base, cond, rng, sev), IDX[cond]);
          done++;
          onProgress && onProgress(done, total);
        }
    return total;
  }

  global.SeedData = { seed, makePatch, sampleTone, mulberry, applyLighting };
})(window);
