/*
 * real_models.js — REAL-DATA attribute detectors.
 *
 * Unlike the synthetic condition baseline (seed.js), these three detectors are
 * trained on ~4,900 REAL Indian-face photos (5,000 sampled, grayscale/no-skin
 * dropped) with human labels (clear skin, dark circles, wrinkles) via
 * train/build.html. Only the learned weights ship here — never the photos.
 * Inference is a standardised logistic regression over the FaceFeatures vector,
 * so it stays 100% offline / pure-JS / no CDN.
 *
 * Held-out AUC on a ~970-image test split: clear 0.71, wrinkle 0.70, dark 0.68.
 * These are honest, moderate signals — surfaced as *estimates*, not diagnoses.
 * (oily_skin trained to ~0.57 AUC ≈ chance, so oiliness stays the direct CV
 * specular measurement in skin_metrics.js instead of a weak learned model.)
 *
 * WEIGHTS come straight from the trainer's exported bundle. To retrain: open
 * train/build.html against a local ../_ds/ dataset and paste the new bundle.
 */
(function (global) {
  "use strict";

  const BUNDLE = {"names":["skinFrac","meanL","meanA","meanB","oiliness","texGlobal","texUpper","spotDensity","rednessDiffuse","darkBand","contrast","warmth","brightPct","p90luma","lowerShine","bandChroma"],"mean":[0.5879,0.4858,0.6629,0.7162,0.5495,0.7206,0.7283,0.9983,0.0781,0.901,0.8788,0.3633,0.6135,0.7255,0.5481,0.1063],"std":[0.1661,0.1012,0.0493,0.0725,0.4158,0.202,0.2019,0.0335,0.1396,0.2167,0.1441,0.1823,0.4234,0.1162,0.4182,0.1654],"models":{"clear":{"w":[0.2887,0.5978,0.2049,-0.2919,-0.1967,-0.2698,0.0107,-0.0097,-0.2116,0.1373,0.4064,0.1584,-0.2035,0.3024,-0.1092,-0.2957],"b":0.2219},"dark":{"w":[-0.0255,-0.4517,-0.0291,0.4616,0.115,0.4413,-0.7253,-0.0018,0.0539,0.157,0.0131,0.0486,-0.0034,-0.0845,-0.0352,-0.091],"b":-0.0283},"wrinkle":{"w":[-0.4901,-0.379,-0.0496,-0.0607,0.1603,0.1298,0.3575,-0.0172,0.0069,-0.1492,-0.4553,-0.1509,0.2268,-0.039,0.0912,0.3748],"b":-0.2579}},"trainedOn":4867,"date":"2026-07-03","testAUC":{"clear":0.71,"dark":0.68,"wrinkle":0.70}};

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  function prob(model, z) {
    let s = model.b;
    for (let j = 0; j < z.length; j++) s += model.w[j] * z[j];
    return sigmoid(s);
  }

  // score(imageData) -> { clarity, darkCircles, wrinkles } as 0..100.
  // clarity   = P(clear skin)          (higher is better)
  // darkCircles / wrinkles = P(present) (higher is more concern)
  function score(imageData) {
    if (!global.FaceFeatures) return null;
    const f = FaceFeatures.extract(imageData);
    if (!f.ok) return null;
    const z = f.vec.map((v, j) => (v - BUNDLE.mean[j]) / BUNDLE.std[j]);
    return {
      clarity: Math.round(prob(BUNDLE.models.clear, z) * 100),
      darkCircles: Math.round(prob(BUNDLE.models.dark, z) * 100),
      wrinkles: Math.round(prob(BUNDLE.models.wrinkle, z) * 100),
    };
  }

  global.RealModels = { score, BUNDLE };
})(window);
