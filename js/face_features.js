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
    // spatial / regional signals (added so a non-linear model can exploit WHERE
    // darkness & texture sit, not just face-wide averages)
    "upperLuma", "eyeLuma", "cheekLuma", "lowerLuma",
    "texMid", "texLower", "edgeDensity", "chromaRowVar",
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

    // pass 2: texture (Laplacian) global + per-region bands, local dark spots,
    // and edge density. Bands (fractions of H) for a roughly centred face:
    //   upper 0.15–0.50 ≈ forehead/eye (wrinkles), mid 0.45–0.68 ≈ cheek/nose,
    //   lower 0.68–0.95 ≈ mouth/chin (nasolabial lines). A non-linear model can
    // then tell forehead-lines from smile-lines instead of one blurred average.
    let texAcc = 0, texN = 0, texUp = 0, texUpN = 0, darkSpots = 0;
    let texMidAcc = 0, texMidN = 0, texLowAcc = 0, texLowN = 0, edgeCount = 0;
    const upT = (H * 0.15) | 0, upB = (H * 0.5) | 0;
    const midT = (H * 0.45) | 0, midB = (H * 0.68) | 0;
    const lowLT = (H * 0.68) | 0, lowLB = (H * 0.95) | 0;
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = (y * W + x) * 4;
        if (!isSkin(d[i], d[i + 1], d[i + 2])) continue;
        const c = luma[y * W + x];
        if (c < meanLuma * 0.72) darkSpots++;
        const lap = Math.abs(4 * c - luma[(y - 1) * W + x] - luma[(y + 1) * W + x]
          - luma[y * W + (x - 1)] - luma[y * W + (x + 1)]);
        texAcc += lap; texN++;
        if (lap > 12) edgeCount++;
        if (y >= upT && y < upB) { texUp += lap; texUpN++; }
        if (y >= midT && y < midB) { texMidAcc += lap; texMidN++; }
        if (y >= lowLT && y < lowLB) { texLowAcc += lap; texLowN++; }
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

    // regional luma means (skin-only) as ratios to the whole-face mean, so they
    // are tone-invariant: a dark under-eye band reads the same on fair or deep
    // skin. Bands: upper 0.10–0.35 (brow), eye 0.28–0.45 (dark circles),
    // cheek 0.45–0.65 (clarity reference), lower 0.68–0.95 (chin).
    function bandLumaRatio(y0f, y1f) {
      const y0 = (H * y0f) | 0, y1 = (H * y1f) | 0;
      let s = 0, c = 0;
      for (let y = y0; y < y1; y++) { s += rowLumaSum[y]; c += rowLumaN[y]; }
      return c > 0 ? (s / c) / (meanLuma + 1e-3) : 1;
    }
    const upperLuma = bandLumaRatio(0.10, 0.35);
    const eyeLuma = bandLumaRatio(0.28, 0.45);
    const cheekLuma = bandLumaRatio(0.45, 0.65);
    const lowerLuma = bandLumaRatio(0.68, 0.95);
    // variability of per-row chroma across skin rows — uneven pigmentation /
    // under-eye discoloration raises it vs an even complexion.
    let cN = 0, cMean = 0, cM2 = 0;
    for (let y = 0; y < H; y++) {
      if (rowLumaN[y] < 3) continue;
      const rc = rowChromaSum[y] / rowLumaN[y];
      cN++; const dlt = rc - cMean; cMean += dlt / cN; cM2 += dlt * (rc - cMean);
    }
    const chromaRowVar = cN > 1 ? Math.sqrt(cM2 / cN) : 0;

    // map a luma ratio (~0.7–1.2) into ~[0,1]; model standardises after.
    const ratioNorm = (r) => clamp01((r - 0.7) / 0.55);

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
      ratioNorm(upperLuma),                        // upperLuma (brow band)
      ratioNorm(eyeLuma),                          // eyeLuma (dark circles)
      ratioNorm(cheekLuma),                        // cheekLuma (clarity ref)
      ratioNorm(lowerLuma),                        // lowerLuma (chin band)
      clamp01((texMidAcc / Math.max(1, texMidN)) / 20), // texMid (cheek smoothness)
      clamp01((texLowAcc / Math.max(1, texLowN)) / 20), // texLower (smile lines)
      clamp01((edgeCount / Math.max(1, texN)) / 0.15),  // edgeDensity (wrinkle lines)
      clamp01(chromaRowVar / 6),                   // chromaRowVar (uneven tone)
    ];
    return { names: FEATURE_NAMES, vec, ok: true };
  }

  global.FaceFeatures = { extract, FEATURE_NAMES };
})(window);
