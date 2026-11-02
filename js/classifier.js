/*
 * classifier.js — the learned ML core. FULLY OFFLINE (no network, no CDN, no
 * model download). This is deliberate: the earlier MobileNet-from-CDN version
 * broke with "Failed to fetch" whenever the network/CDN was unavailable (and on
 * file://). Everything now runs in pure JS in the browser.
 *
 * Pipeline: image patch -> hand-crafted skin feature vector (HSV colour
 * histogram over skin pixels + the CV skin metrics + Lab tone stats) -> KNN
 * few-shot classifier the brand trains in-browser on its OWN Indian-skin photos.
 *
 * That "few-shot on your own data, no cloud" loop is the moat. Scale path: swap
 * this vector for a brand-trained CNN's embedding — the interface is unchanged.
 *
 * Predictions are CALIBRATED: label smoothing + out-of-distribution flattening,
 * so a unanimous 7-neighbour vote reads as strong evidence, never 100%/0%.
 * Each example carries an origin tag ("real" brand photo vs synthetic "seed");
 * only real brand photos may override the shipped real-derm model (app.js).
 */
(function (global) {
  "use strict";

  const CLASSES = ["Clear", "Acne", "Fungal", "Eczema", "Pigmentation", "Dryness"];
  const STORE_KEY = "twacha_knn_v3";
  const K = 7;

  // Per-dimension distance weights so CONDITION drives classification, not skin
  // tone. 12 histogram dims (tone-ish) down-weighted, 6 CV-metric dims (redness,
  // pigment, texture… — the condition signal) up-weighted, 3 Lab-tone dims low.
  // Applied inside featurize so stored + query vectors stay consistent.
  const WEIGHTS = [
    ...new Array(12).fill(0.4),   // HSV histogram (tone-ish, low influence)
    2.0, 2.4, 2.6, 2.4, 2.0, 1.4, // Oiliness Redness Pigment Texture Dryness Radiance
    0.5, 0.5, 0.5,                // Lab L a b
    2.6, 2.6,                     // rednessLocal (acne) , rednessDiffuse (eczema)
    2.6,                          // fungalPatch (light desaturated patch)
  ];

  let examples = [];   // [{label, vec:[...], origin:"seed"|"real"}]
  let ready = false;
  let scaleCache = null; // typical nearest-neighbour distance (calibration ref)

  async function init(onState) {
    loadFromStorage();
    ready = true;
    onState && onState(examples.length ? `ML ready · ${examples.length} examples` : "ML ready (offline)");
    return ready;
  }

  // ---- feature extraction --------------------------------------------------
  function rgb2hsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), df = mx - mn;
    let h = 0;
    if (df !== 0) {
      if (mx === r) h = ((g - b) / df) % 6;
      else if (mx === g) h = (b - r) / df + 2;
      else h = (r - g) / df + 4;
      h *= 60; if (h < 0) h += 360;
    }
    return { h, s: mx === 0 ? 0 : df / mx, v: mx };
  }

  // canvas -> normalised, weighted feature vector (length 24)
  function featurize(canvas) {
    const ctx = canvas.getContext("2d");
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d = img.data;
    const hBins = new Array(6).fill(0), sBins = new Array(3).fill(0), vBins = new Array(3).fill(0);
    let skin = 0, total = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];
      const isSkin = SkinMetrics.isSkin(r, g, b);
      if (!isSkin) { total++; continue; }
      total++; skin++;
      const { h, s, v } = rgb2hsv(r, g, b);
      hBins[Math.min(5, Math.floor(h / 60))]++;
      sBins[Math.min(2, Math.floor(s * 3))]++;
      vBins[Math.min(2, Math.floor(v * 3))]++;
    }
    const norm = (arr, n) => arr.map((x) => (n > 0 ? x / n : 0));
    const hist = [...norm(hBins, skin), ...norm(sBins, skin), ...norm(vBins, skin)]; // 12

    // metric + tone part (reuse the CV engine)
    const prof = SkinMetrics.analyse(img);
    let metrics, tone, redSplit;
    if (prof.ok) {
      const s = prof.scores;
      metrics = [s.Oiliness, s.Redness, s.Pigmentation, s.Texture, s.Dryness, s.Radiance].map((x) => x / 100);
      tone = [prof.raw.meanL / 100, (prof.raw.meanA + 50) / 100, (prof.raw.meanB + 50) / 100];
      redSplit = [prof.raw.rednessLocal, prof.raw.rednessDiffuse, prof.raw.fungalPatch]; // acne / eczema / fungal cues
    } else {
      metrics = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
      tone = [0.5, 0.5, 0.5];
      redSplit = [0, 0, 0];
    }
    const vec = [...hist, ...metrics, ...tone, ...redSplit]; // 24-d
    return vec.map((x, i) => x * WEIGHTS[i]);   // weighted so condition > tone
  }

  function dist2(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return s;
  }

  // ---- KNN api (same signatures the app already calls) ---------------------
  // origin: "real" = a brand's own labelled photo (the personalisation moat),
  // "seed" = the synthetic tone prior. The app routes on this: only REAL brand
  // photos may override the shipped real-derm model.
  function addExample(canvas, classIdx, origin) {
    if (!ready) return false;
    examples.push({ label: CLASSES[classIdx], vec: featurize(canvas), origin: origin || "real" });
    scaleCache = null;
    saveToStorage();
    return true;
  }

  // Typical nearest-neighbour distance inside the training set (median over a
  // sample). Used as the calibration reference: a query whose nearest neighbour
  // sits far beyond this is out-of-distribution and must NOT be confident.
  function refScale() {
    if (scaleCache != null) return scaleCache;
    const N = examples.length;
    if (N < 8) return (scaleCache = 1);
    const S = Math.min(160, N), step = Math.max(1, Math.floor(N / S));
    const ds = [];
    for (let i = 0; i < N; i += step) {
      let best = Infinity;
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        const d = dist2(examples[i].vec, examples[j].vec);
        if (d < best) best = d;
      }
      ds.push(Math.sqrt(best));
    }
    ds.sort((a, b) => a - b);
    return (scaleCache = ds[Math.floor(ds.length / 2)] || 1);
  }

  const SMOOTH = 0.15; // label smoothing floor -> max confidence ~87%, never 100/0

  function predict(canvas) {
    if (!ready || examples.length === 0) return null;
    const v = featurize(canvas);
    const scored = examples
      .map((e) => ({ label: e.label, d: Math.sqrt(dist2(v, e.vec)) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, Math.min(K, examples.length));

    // distance-weighted vote
    const weight = {}; let wsum = 0;
    CLASSES.forEach((c) => (weight[c] = 0));
    scored.forEach((s) => { const w = 1 / (s.d + 1e-3); weight[s.label] += w; wsum += w; });

    const confidences = {};
    CLASSES.forEach((c) => (confidences[c] = wsum > 0 ? weight[c] / wsum : 0));

    // ---- calibration (fixes the 100%-one-class / 0%-everything-else collapse)
    // 1. reliability: how in-distribution is this query? 1 when its nearest
    //    neighbour is within the typical NN distance, decaying beyond it.
    // 2. smoothing: mix toward uniform — a 7-0 neighbour vote is evidence,
    //    not certainty. Far queries flatten hard toward "don't know".
    const D0 = refScale();
    const rel = Math.exp(-Math.max(0, scored[0].d - D0) / (D0 || 1));
    const lam = 1 - (1 - SMOOTH) * rel;
    CLASSES.forEach((c) => (confidences[c] = (1 - lam) * confidences[c] + lam / CLASSES.length));

    let label = CLASSES[0], best = -1;
    CLASSES.forEach((c) => { if (confidences[c] > best) { best = confidences[c]; label = c; } });

    return { label, confidence: Math.round(best * 100), confidences, reliability: rel };
  }

  function counts() {
    const out = {}; CLASSES.forEach((c) => (out[c] = 0));
    examples.forEach((e) => (out[e.label] = (out[e.label] || 0) + 1));
    return out;
  }
  function totalExamples() { return examples.length; }
  function realExamples() { return examples.reduce((n, e) => n + (e.origin === "real" ? 1 : 0), 0); }
  function isTrained() { return ready && examples.length > 0; }
  // true only when the brand has taught it on its OWN photos — the only case
  // where this KNN is allowed to override the shipped real-derm model.
  function hasRealExamples() { return ready && realExamples() > 0; }

  // ---- persistence ---------------------------------------------------------
  function saveToStorage() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(examples)); }
    catch (e) { console.warn("save failed", e); }
  }
  function loadFromStorage() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) examples = JSON.parse(raw);
      // examples stored before origin-tagging existed are of unknown provenance —
      // treat them as "seed" so data we can't vouch for never outranks the
      // shipped real-derm model. New brand photos get tagged "real" on add.
      examples.forEach((e) => { if (!e.origin) e.origin = "seed"; });
    }
    catch (e) { console.warn("load failed", e); examples = []; }
    scaleCache = null;
  }
  function reset() { examples = []; scaleCache = null; localStorage.removeItem(STORE_KEY); }

  global.Classifier = {
    CLASSES, init, addExample, predict, counts, totalExamples, realExamples,
    isTrained, hasRealExamples, reset, featurize, get ready() { return ready; },
  };
})(window);
