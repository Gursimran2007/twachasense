/*
 * face_features.js — face-level feature extractor for the REAL-DATA detectors.
 *
 * The condition screen (classifier.js) works on a centre cheek ROI. The four
 * attributes we have REAL Indian-skin labels for — clear skin, oily skin, dark
 * circles, wrinkles — are whole-FACE properties (dark circles live under the
 * eyes, wrinkles on the forehead), so they need a face-level descriptor computed
 * over the entire photo, not just the cheek patch.
 *
 * CRITICAL: this exact function is used for BOTH training (train/build.html over
 * the labelled dataset) AND in-browser inference (app.js). Sharing one code path
 * guarantees the feature distribution the model was trained on matches what it
 * sees at run time — the single biggest failure mode of a separate train/serve
 * pipeline. Everything is offline, pure JS, no network.
 */
(function (global) {
  "use strict";

  const FEATURE_NAMES = [
    "skinFrac", "meanL", "meanA", "meanB", "oiliness",
    "texGlobal", "texUpper", "spotDensity", "rednessDiffuse",
    "darkBand", "contrast", "warmth",
    "brightPct", "p90luma", "lowerShine", "bandChroma",
  ];

  function rgb2lab(r, g, b) {
    let R = r / 255, G = g / 255, B = b / 255;
    const f = (c) => (c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92);
    R = f(R); G = f(G); B = f(B);
    let X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047;
    let Y = R * 0.2126 + G * 0.7152 + B * 0.0722;
    let Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;
    const g2 = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    X = g2(X); Y = g2(Y); Z = g2(Z);
    return { L: 116 * Y - 16, a: 500 * (X - Y), b: 200 * (Y - Z) };
  }
  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

  // Extract from an ImageData (any size; ~160² is plenty). Returns a fixed-length
  // vector aligned with FEATURE_NAMES. Values are pre-normalised to ~[0,1] ranges
  // (the model additionally standardises with its own mean/std at load time).
  function extract(img) {
    const d = img.data, W = img.width, H = img.height;
    const isSkin = SkinMetrics.isSkin;
    const luma = new Float32Array(W * H);

    let n = 0, sumL = 0, sumA = 0, sumB = 0, sumLuma = 0, sumLuma2 = 0, shine = 0;
    let bright = 0, lowerShine = 0, lowerN = 0;
    const rowLumaSum = new Float64Array(H), rowLumaN = new Float64Array(H);
    const rowChromaSum = new Float64Array(H);      // Σ(|a|+|b|) per row for under-eye chroma
    const skinLuma = [];                            // for a robust luma percentile
    const lowerT = (H * 0.45) | 0;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const r = d[i], g = d[i + 1], b = d[i + 2];
        const lm = 0.299 * r + 0.587 * g + 0.114 * b;
        luma[y * W + x] = lm;
        if (!isSkin(r, g, b)) continue;
        n++;
        const lab = rgb2lab(r, g, b);
        sumL += lab.L; sumA += lab.a; sumB += lab.b;
        sumLuma += lm; sumLuma2 += lm * lm;
        rowLumaSum[y] += lm; rowLumaN[y] += 1;
        rowChromaSum[y] += Math.abs(lab.a) + Math.abs(lab.b);
        skinLuma.push(lm);
        if (lm > 205) bright++;
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const sat = mx === 0 ? 0 : (mx - mn) / mx;
        const isShine = lm > 195 && sat < 0.28;
        if (isShine) shine++;
        if (y >= lowerT) { lowerN++; if (isShine) lowerShine++; }
      }
    }

    const total = W * H;
    if (n < total * 0.05) {                 // barely any skin — return neutral vec
      return { names: FEATURE_NAMES, vec: new Array(FEATURE_NAMES.length).fill(0.5), ok: false };
    }

    const meanL = sumL / n, meanA = sumA / n, meanB = sumB / n;
    const meanLuma = sumLuma / n;
    const variance = Math.max(0, sumLuma2 / n - meanLuma * meanLuma);
    const contrast = Math.sqrt(variance);

    // pass 2: texture (Laplacian) global + upper band, and local dark spots.
    // Upper band (rows 0.15–0.50 H) ≈ forehead/eye zone for a roughly centred
    // face → wrinkle signal. Lower influence bands are captured by the global.
    let texAcc = 0, texN = 0, texUp = 0, texUpN = 0, darkSpots = 0;
    const upT = (H * 0.15) | 0, upB = (H * 0.5) | 0;
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = (y * W + x) * 4;
        if (!isSkin(d[i], d[i + 1], d[i + 2])) continue;
        const c = luma[y * W + x];
        if (c < meanLuma * 0.72) darkSpots++;
        const lap = Math.abs(4 * c - luma[(y - 1) * W + x] - luma[(y + 1) * W + x]
          - luma[y * W + (x - 1)] - luma[y * W + (x + 1)]);
        texAcc += lap; texN++;
        if (y >= upT && y < upB) { texUp += lap; texUpN++; }
      }
    }

    // dark-circle proxy: darkest smoothed skin-row in the eye zone vs face mean.
    // A pronounced dark band under the eyes drops a run of rows well below mean;
    // we also read that band's chroma (under-eye circles are often more saturated
    // / bluish-purple than the cheek).
    let minBand = meanLuma, K = 3, bandY = upT;
    for (let y = upT; y < upB; y++) {
      let s = 0, cnt = 0;
      for (let k = -K; k <= K; k++) {
        const yy = y + k; if (yy < 0 || yy >= H) continue;
        s += rowLumaSum[yy]; cnt += rowLumaN[yy];
      }
      if (cnt > 0) { const m = s / cnt; if (m < minBand) { minBand = m; bandY = y; } }
    }
    const darkBand = clamp01(((meanLuma - minBand) / (meanLuma + 1e-3)) * 4);
    // mean chroma over the darkest band (±K rows) vs the whole-face mean chroma
    let bandChr = 0, bandChrN = 0;
    for (let k = -K; k <= K; k++) {
      const yy = bandY + k; if (yy < 0 || yy >= H) continue;
      bandChr += rowChromaSum[yy]; bandChrN += rowLumaN[yy];
    }
    let faceChr = 0; for (let y = 0; y < H; y++) faceChr += rowChromaSum[y];
    const meanChr = faceChr / n;
    const bandChroma = clamp01(((bandChrN ? bandChr / bandChrN : meanChr) / (meanChr + 1e-3) - 0.7) / 0.9);

    // 90th-percentile skin luma — a robust "how bright do the shiny bits get"
    // signal for oiliness that a simple mean washes out.
    skinLuma.sort((a, b) => a - b);
    const p90luma = skinLuma[(skinLuma.length * 0.9) | 0] || meanLuma;

    const rednessDiffuse = clamp01((meanA - 0.5 * meanB - 5) / 18);

    const vec = [
      clamp01(n / total),                         // skinFrac
      clamp01(meanL / 100),                        // meanL
      clamp01((meanA + 40) / 80),                  // meanA
      clamp01((meanB + 40) / 80),                  // meanB
      clamp01((shine / n) / 0.06),                 // oiliness
      clamp01((texAcc / Math.max(1, texN)) / 20),  // texGlobal
      clamp01((texUp / Math.max(1, texUpN)) / 20), // texUpper (wrinkle)
      clamp01((darkSpots / n) / 0.08),             // spotDensity (clarity inv)
      rednessDiffuse,                              // rednessDiffuse
      darkBand,                                    // darkBand (dark circles)
      clamp01(contrast / 55),                      // contrast
      clamp01((meanB - 0.5 * meanA) / 30),         // warmth
      clamp01((bright / n) / 0.04),                // brightPct (strong highlights)
      clamp01(p90luma / 255),                      // p90luma
      clamp01((lowerShine / Math.max(1, lowerN)) / 0.05), // lowerShine (T-zone)
      bandChroma,                                  // bandChroma (under-eye)
    ];
    return { names: FEATURE_NAMES, vec, ok: true };
  }

  global.FaceFeatures = { extract, FEATURE_NAMES };
})(window);
