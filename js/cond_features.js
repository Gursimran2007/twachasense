/*
 * cond_features.js — condition-specific image descriptor. SHARED by the dev
 * trainer (train/build_cond.html) and the shipped inference model
 * (js/cond_models.js), so the feature distribution is byte-identical at train
 * and serve time — the single most important property of an honest ML pipeline.
 *
 * Unlike Classifier.featurize (tuned for the synthetic tone-diverse baseline and
 * masked to skin pixels), this descriptor is built for REAL dermatology-style
 * close-ups: it runs over ALL pixels (a lesion often fails the YCbCr skin mask),
 * uses finer colour histograms, richer texture statistics, and a 3x3 SPATIAL
 * grid so the model can see WHERE the abnormality sits, not just its average.
 *
 * extract(imageData) -> { ok, vec:[...], names:[...] }
 *   ok=false for grayscale / near-chroma-free images (they add only label noise).
 * Values are left roughly raw; the trainer standardises every column, and the
 * shipped model reuses that same mean/std — so absolute scaling is irrelevant as
 * long as it is consistent, which it is because both call THIS function.
 */
(function (global) {
  "use strict";

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
  const clamp01 = (x) => Math.max(0, Math.min(1, x));

  // 8 hue + 4 sat + 4 val = 16 histogram bins
  const HBINS = 8, SBINS = 4, VBINS = 4;
  const OBINS = 6;                       // gradient-orientation histogram bins

  const FEATURE_NAMES = [
    // global Lab colour moments (6)
    "meanL", "meanA", "meanB", "stdL", "stdA", "stdB",
    // HSV histogram (16)
    "h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7",
    "s0", "s1", "s2", "s3", "v0", "v1", "v2", "v3",
    // erythema / redness (4)
    "meanRG", "rednessLocal", "rednessDiffuse", "redSpotFrac",
    // texture (3)
    "texMean", "texStd", "edgeDensity",
    // patch anomalies (2)
    "lightDesatFrac", "darkSpotFrac",
    // 3x3 spatial luma ratio grid (9)
    "gL0", "gL1", "gL2", "gL3", "gL4", "gL5", "gL6", "gL7", "gL8",
    // 3x3 spatial a* (redness) grid (9)
    "gA0", "gA1", "gA2", "gA3", "gA4", "gA5", "gA6", "gA7", "gA8",
    // --- v2 additions (23) : richer signal to break the 49-d ceiling ---
    // 3x3 spatial b* (yellow/brown crust) grid (9)
    "gB0", "gB1", "gB2", "gB3", "gB4", "gB5", "gB6", "gB7", "gB8",
    // colour heterogeneity (2): circular hue dispersion + chroma spread
    "hueDisp", "stdChroma",
    // redness/yellow distribution shape (2): spotty (acne) vs diffuse
    "aSkew", "bSkew",
    // coarse (multi-scale) texture (2): flaking / scaliness
    "coarseTexMean", "coarseTexStd",
    // gradient-orientation histogram (6): directional scale-edges / streaks
    "o0", "o1", "o2", "o3", "o4", "o5",
    // colour-region fractions (2): yellow crust, blue/purple vascular
    "yellowCrustFrac", "bluePurpleFrac",
  ];

  function extract(img) {
    const d = img.data, W = img.width, H = img.height, N = W * H;
    const luma = new Float32Array(N);
    const hBins = new Array(HBINS).fill(0);
    const sBins = new Array(SBINS).fill(0);
    const vBins = new Array(VBINS).fill(0);
    let sumL = 0, sumA = 0, sumB = 0, sumL2 = 0, sumA2 = 0, sumB2 = 0;
    let sumRG = 0, sumLuma = 0, sumChroma = 0, sumChroma2 = 0;
    let sumA3 = 0, sumB3 = 0;                 // 3rd raw moments -> skew
    let sumSinH = 0, sumCosH = 0, sumSat = 0; // sat-weighted circular hue
    let yellowCrust = 0, bluePurple = 0;      // colour-region fractions
    // 3x3 grid accumulators
    const gLuma = new Float64Array(9), gA = new Float64Array(9),
          gB = new Float64Array(9), gN = new Float64Array(9);

    for (let y = 0; y < H; y++) {
      const gy = y < H / 3 ? 0 : y < (2 * H) / 3 ? 1 : 2;
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const r = d[i], g = d[i + 1], b = d[i + 2];
        const lu = 0.299 * r + 0.587 * g + 0.114 * b;
        luma[y * W + x] = lu;
        const lab = rgb2lab(r, g, b);
        sumL += lab.L; sumA += lab.a; sumB += lab.b;
        sumL2 += lab.L * lab.L; sumA2 += lab.a * lab.a; sumB2 += lab.b * lab.b;
        sumA3 += lab.a * lab.a * lab.a; sumB3 += lab.b * lab.b * lab.b;
        sumRG += Math.max(0, r - g) / 255;
        sumLuma += lu;
        const chroma = Math.sqrt(lab.a * lab.a + lab.b * lab.b);
        sumChroma += chroma; sumChroma2 += chroma * chroma;
        // yellow crust (high b*, mid lightness) vs blue/purple (negative b*)
        if (lab.b > 20 && lab.L > 30 && lab.L < 82) yellowCrust++;
        if (lab.b < -8) bluePurple++;
        const { h, s, v } = rgb2hsv(r, g, b);
        const rad = (h * Math.PI) / 180;
        sumSinH += s * Math.sin(rad); sumCosH += s * Math.cos(rad); sumSat += s;
        hBins[Math.min(HBINS - 1, Math.floor((h / 360) * HBINS))]++;
        sBins[Math.min(SBINS - 1, Math.floor(s * SBINS))]++;
        vBins[Math.min(VBINS - 1, Math.floor(v * VBINS))]++;
        const gx = x < W / 3 ? 0 : x < (2 * W) / 3 ? 1 : 2;
        const cell = gy * 3 + gx;
        gLuma[cell] += lu; gA[cell] += lab.a; gB[cell] += lab.b; gN[cell]++;
      }
    }

    const meanL = sumL / N, meanA = sumA / N, meanB = sumB / N;
    const meanLuma = sumLuma / N || 1;
    const meanRG = sumRG / N;
    const meanChroma = sumChroma / N;
    // grayscale / chroma-free image -> useless for a condition model, skip it
    if (meanChroma < 4) return { ok: false, reason: "grayscale / no chroma" };
    const stdL = Math.sqrt(Math.max(0, sumL2 / N - meanL * meanL));
    const stdA = Math.sqrt(Math.max(0, sumA2 / N - meanA * meanA));
    const stdB = Math.sqrt(Math.max(0, sumB2 / N - meanB * meanB));
    const stdChroma = Math.sqrt(Math.max(0, sumChroma2 / N - meanChroma * meanChroma));
    // sat-weighted circular hue dispersion: 0 = one dominant hue, 1 = spread out
    const hueR = Math.sqrt(sumSinH * sumSinH + sumCosH * sumCosH) / (sumSat || 1);
    const hueDisp = clamp01(1 - hueR);
    // Lab a*/b* skew (central 3rd moment / sigma^3): spotty vs diffuse
    const m3A = sumA3 / N - 3 * meanA * (sumA2 / N) + 2 * meanA * meanA * meanA;
    const m3B = sumB3 / N - 3 * meanB * (sumB2 / N) + 2 * meanB * meanB * meanB;
    const aSkew = m3A / (Math.pow(stdA, 3) || 1);
    const bSkew = m3B / (Math.pow(stdB, 3) || 1);

    // pass 2: texture (Laplacian) + local red/light/dark patch fractions +
    // coarse (dilated) Laplacian for scaliness + gradient-orientation histogram
    let texAcc = 0, texN = 0, texSq = 0, edge = 0, redSpots = 0, lightDesat = 0, darkSpots = 0;
    let coarseAcc = 0, coarseSq = 0, coarseN = 0, totalMag = 0;
    const oBins = new Float64Array(OBINS);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = (y * W + x) * 4;
        const r = d[i], g = d[i + 1], b = d[i + 2];
        const c = luma[y * W + x];
        const rg = Math.max(0, r - g) / 255;
        if (rg > meanRG + 0.06) redSpots++;
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const sat = mx === 0 ? 0 : (mx - mn) / mx;
        if (c > meanLuma * 1.04 && sat < 0.28) lightDesat++;
        if (c < meanLuma * 0.72) darkSpots++;
        const up = luma[(y - 1) * W + x], dn = luma[(y + 1) * W + x];
        const lf = luma[y * W + (x - 1)], rt = luma[y * W + (x + 1)];
        const lap = Math.abs(4 * c - up - dn - lf - rt);
        texAcc += lap; texSq += lap * lap; texN++;
        if (lap > 12) edge++;
        // gradient orientation (unsigned 0..pi), magnitude-weighted
        const gx = rt - lf, gyv = dn - up;
        const mag = Math.sqrt(gx * gx + gyv * gyv);
        if (mag > 6) {
          let ori = Math.atan2(gyv, gx); if (ori < 0) ori += Math.PI;
          let ob = Math.floor((ori / Math.PI) * OBINS); if (ob >= OBINS) ob = OBINS - 1;
          oBins[ob] += mag; totalMag += mag;
        }
        // coarse Laplacian on a 2px-dilated stencil -> larger-scale flaking
        if (y >= 2 && y < H - 2 && x >= 2 && x < W - 2) {
          const clap = Math.abs(
            4 * c - luma[(y - 2) * W + x] - luma[(y + 2) * W + x] -
            luma[y * W + (x - 2)] - luma[y * W + (x + 2)]
          );
          coarseAcc += clap; coarseSq += clap * clap; coarseN++;
        }
      }
    }
    texN = texN || 1;
    const texMean = texAcc / texN;
    const texStd = Math.sqrt(Math.max(0, texSq / texN - texMean * texMean));
    coarseN = coarseN || 1;
    const coarseTexMean = coarseAcc / coarseN;
    const coarseTexStd = Math.sqrt(Math.max(0, coarseSq / coarseN - coarseTexMean * coarseTexMean));
    const magNorm = totalMag || 1;

    const rednessLocal = clamp01((redSpots / N) / 0.10);
    const rednessDiffuse = clamp01((meanA - 0.5 * meanB - 5) / 18);

    const vec = [
      meanL / 100, (meanA + 50) / 100, (meanB + 50) / 100,
      stdL / 50, stdA / 30, stdB / 30,
      ...hBins.map((x) => x / N), ...sBins.map((x) => x / N), ...vBins.map((x) => x / N),
      meanRG * 5, rednessLocal, rednessDiffuse, (redSpots / N) * 5,
      texMean / 22, texStd / 22, edge / texN,
      (lightDesat / N) * 8, (darkSpots / N) * 8,
    ];
    // 3x3 luma ratio grid (relative to overall mean luma)
    for (let c = 0; c < 9; c++) vec.push(gN[c] > 0 ? (gLuma[c] / gN[c]) / meanLuma : 1);
    // 3x3 a* grid (redness spatial distribution, relative to overall a*)
    for (let c = 0; c < 9; c++) vec.push(gN[c] > 0 ? (gA[c] / gN[c] - meanA) / 20 : 0);
    // --- v2 additions (23) ---
    // 3x3 b* grid (yellow/brown crust spatial distribution, relative to overall b*)
    for (let c = 0; c < 9; c++) vec.push(gN[c] > 0 ? (gB[c] / gN[c] - meanB) / 20 : 0);
    // colour heterogeneity
    vec.push(hueDisp, stdChroma / 40);
    // redness/yellow skew (clamped for outlier robustness)
    vec.push(clamp01((aSkew + 3) / 6) * 2 - 1, clamp01((bSkew + 3) / 6) * 2 - 1);
    // coarse texture
    vec.push(coarseTexMean / 30, coarseTexStd / 30);
    // gradient-orientation histogram (magnitude-normalised)
    for (let c = 0; c < OBINS; c++) vec.push(oBins[c] / magNorm);
    // colour-region fractions
    vec.push((yellowCrust / N) * 4, (bluePurple / N) * 4);

    return { ok: true, vec, names: FEATURE_NAMES };
  }

  global.CondFeatures = { extract, FEATURE_NAMES };
})(window);
