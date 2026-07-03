/*
 * skin_metrics.js — classical computer-vision skin analysis.
 * Fully offline (no network, no ML weights). Operates on an ImageData patch.
 *
 * Everything here is deterministic and explainable — the metrics beauty brands
 * actually merchandise on: tone, undertone, oiliness, redness, texture, pigmentation.
 * The learned condition classifier lives separately in classifier.js.
 */
(function (global) {
  "use strict";

  // ---- colour-space helpers ------------------------------------------------
  function rgb2ycbcr(r, g, b) {
    return {
      y: 0.299 * r + 0.587 * g + 0.114 * b,
      cb: 128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
      cr: 128 + 0.5 * r - 0.418688 * g - 0.081312 * b,
    };
  }
  function rgb2lab(r, g, b) {
    // sRGB -> linear -> XYZ -> Lab (D65)
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

  // ---- skin mask (YCbCr rule, robust across tones) -------------------------
  function isSkin(r, g, b) {
    const { cb, cr } = rgb2ycbcr(r, g, b);
    return cb >= 77 && cb <= 135 && cr >= 133 && cr <= 180;
  }

  // Monk Skin Tone scale anchor L* values (approx, light->deep)
  const MONK = [
    { n: 1, L: 94, hex: "#f6ede4" }, { n: 2, L: 89, hex: "#f3e7db" },
    { n: 3, L: 83, hex: "#f7ead0" }, { n: 4, L: 74, hex: "#eadaba" },
    { n: 5, L: 64, hex: "#d7bd96" }, { n: 6, L: 54, hex: "#a07e56" },
    { n: 7, L: 43, hex: "#825c43" }, { n: 8, L: 33, hex: "#604134" },
    { n: 9, L: 24, hex: "#3a312a" }, { n: 10, L: 16, hex: "#292420" },
  ];
  function monkFromL(L) {
    let best = MONK[0], d = 1e9;
    for (const m of MONK) { const dd = Math.abs(m.L - L); if (dd < d) { d = dd; best = m; } }
    return best;
  }

  function clamp01(x) { return Math.max(0, Math.min(1, x)); }
  function pct(x) { return Math.round(clamp01(x) * 100); }

  /**
   * analyse(imageData) -> profile object
   */
  function analyse(img) {
    const d = img.data, W = img.width, H = img.height;
    let n = 0;
    let sumL = 0, sumA = 0, sumB = 0;
    let sumR = 0, sumG = 0, sumBl = 0;
    let sumSat = 0;           // mean skin saturation (for fungal-patch detection)
    let redEx = 0;            // erythema (redness) accumulator
    let shine = 0;            // specular highlight pixels
    let darkSpots = 0;        // pigmentation pixels
    const lumaGrid = new Float32Array(W * H); // for texture

    // pass 1: per-pixel skin stats
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const r = d[i], g = d[i + 1], b = d[i + 2];
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;
        lumaGrid[y * W + x] = luma;
        if (!isSkin(r, g, b)) continue;
        n++;
        const lab = rgb2lab(r, g, b);
        sumL += lab.L; sumA += lab.a; sumB += lab.b;
        sumR += r; sumG += g; sumBl += b;
        // accumulate baseline redness (r-g) so redness can be measured RELATIVE
        // to the person's own tone — warm Indian skin must not read as "inflamed"
        redEx += Math.max(0, (r - g)) / 255;
        // specular: bright + low colour saturation => oily shine
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const sat = mx === 0 ? 0 : (mx - mn) / mx;
        sumSat += sat;
        if (luma > 200 && sat < 0.25) shine++;
      }
    }

    if (n < 200) {
      return { ok: false, reason: "Not enough skin detected — move closer / better light." };
    }

    const meanL = sumL / n, meanA = sumA / n, meanB = sumB / n;
    const meanLuma = (0.299 * sumR + 0.587 * sumG + 0.114 * sumBl) / n;
    const meanRG = redEx / n;                                // baseline warmth of THIS skin
    const meanSat = sumSat / n;                              // baseline colourfulness of THIS skin

    // pass 2: pigmentation (locally dark) + texture (local variance) + erythema
    // (redness EXCESS over the person's own baseline — tone-invariant) + fungal
    // (locally LIGHT + DESATURATED patches — the signature that separates the
    // hypopigmented fungal patch from both clear skin and inflamed conditions)
    let texAcc = 0, texN = 0, redSpots = 0, lightDesat = 0;
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = (y * W + x) * 4;
        const r = d[i], g = d[i + 1], b = d[i + 2];
        if (!isSkin(r, g, b)) continue;
        const rg = Math.max(0, (r - g)) / 255;
        if (rg > meanRG + 0.06) redSpots++;                 // locally inflamed patch
        const c = lumaGrid[y * W + x];
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        const sat = mx === 0 ? 0 : (mx - mn) / mx;
        // lighter than local skin AND noticeably less colourful => fungal/hypo patch
        if (c > meanLuma * 1.04 && sat < meanSat * 0.72) lightDesat++;
        if (c < meanLuma * 0.72) darkSpots++;               // dark blob vs local skin
        // Laplacian magnitude => high-frequency texture / pores
        const lap = Math.abs(
          4 * c - lumaGrid[(y - 1) * W + x] - lumaGrid[(y + 1) * W + x] -
          lumaGrid[y * W + (x - 1)] - lumaGrid[y * W + (x + 1)]
        );
        texAcc += lap; texN++;
      }
    }

    // ---- normalise metrics into 0..1 concern scores --------------------------
    const oiliness = clamp01(shine / n / 0.10);              // 10% shine => max
    // redness = max(localised inflamed patches, diffuse erythema). Diffuse index
    // a* - 0.5·b* rises with inflammation (pink) but stays low for golden warmth,
    // so naturally warm Indian skin does NOT read as inflamed.
    const rednessLocal   = clamp01((redSpots / n) / 0.10);
    const rednessDiffuse = clamp01((meanA - 0.5 * meanB - 5) / 18);
    const redness  = Math.max(rednessLocal, rednessDiffuse);
    const fungalPatch = clamp01((lightDesat / n) / 0.12);   // light desaturated area
    const pigment  = clamp01((darkSpots / n) / 0.12);
    const texture  = clamp01((texAcc / Math.max(1, texN)) / 22); // roughness
    const smoothness = 1 - texture;

    // dryness inferred from low oil + high texture (flaky) + duller luma
    const dryness = clamp01(0.6 * texture + 0.4 * (1 - oiliness) - 0.15);
    const radiance = clamp01((meanLuma / 200) * (1 - 0.4 * redness)); // glow proxy

    // undertone from Lab b* (yellow) vs a* (red)
    let undertone;
    const warmth = meanB - meanA * 0.5;
    if (warmth > 14) undertone = "Warm (golden)";
    else if (warmth < 6) undertone = "Cool (pink)";
    else undertone = "Neutral";

    // skin type
    let type;
    if (oiliness > 0.55 && dryness > 0.45) type = "Combination";
    else if (oiliness > 0.5) type = "Oily";
    else if (dryness > 0.5) type = "Dry";
    else type = "Normal / balanced";

    const monk = monkFromL(meanL);

    return {
      ok: true,
      skinPixels: n,
      type,
      tone: { monk: monk.n, hex: monk.hex, L: Math.round(meanL) },
      undertone,
      // scores 0..100
      scores: {
        Oiliness: pct(oiliness),
        Redness: pct(redness),
        Pigmentation: pct(pigment),
        Texture: pct(texture),
        Dryness: pct(dryness),
        Radiance: pct(radiance),
      },
      raw: { meanL, meanA, meanB, smoothness, rednessLocal, rednessDiffuse, fungalPatch },
    };
  }

  global.SkinMetrics = { analyse, monkFromL, isSkin };
})(window);
