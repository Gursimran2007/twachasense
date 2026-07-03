/*
 * recommend.js — the D2C monetisation layer.
 * Maps a skin profile (metrics + screened condition) to ranked concerns and a
 * concrete AM/PM routine, resolved against a BRAND CATALOG (white-label).
 *
 * A brand drops in its own catalog JSON (its real SKUs against each ingredient
 * slot) and TwachaSense becomes that brand's personalised recommender widget.
 */
(function (global) {
  "use strict";

  // Sample white-label catalog. Replace `product`/`url` with a brand's real SKUs.
  const SAMPLE_CATALOG = {
    brand: "Demo Skincare Co.",
    ingredients: {
      salicylic_acid: { product: "2% BHA Clarifying Toner", url: "#" },
      niacinamide:    { product: "10% Niacinamide Serum", url: "#" },
      centella:       { product: "Cica Soothing Gel", url: "#" },
      vitamin_c:      { product: "15% Vitamin C Booster", url: "#" },
      kojic_acid:     { product: "Kojic Bright Serum", url: "#" },
      hyaluronic:     { product: "Hyaluronic Hydra Serum", url: "#" },
      ceramide:       { product: "Ceramide Barrier Cream", url: "#" },
      spf:            { product: "Mattifying SPF 50 PA++++", url: "#" },
      retinol:        { product: "0.3% Retinol Night Serum", url: "#" },
      gentle_cleanser:{ product: "pH-Balanced Gel Cleanser", url: "#" },
    },
  };

  // ingredient -> which concerns it addresses (for AM/PM slotting)
  const INGREDIENT = {
    salicylic_acid: { for: ["Oiliness", "Texture"], when: "AM", note: "unclogs pores, controls oil" },
    niacinamide:    { for: ["Oiliness", "Redness", "Pigmentation"], when: "AM", note: "regulates sebum, calms" },
    centella:       { for: ["Redness"], when: "AM", note: "soothes inflammation" },
    vitamin_c:      { for: ["Pigmentation", "Radiance"], when: "AM", note: "brightens, evens tone" },
    kojic_acid:     { for: ["Pigmentation"], when: "PM", note: "fades dark spots" },
    hyaluronic:     { for: ["Dryness"], when: "AM", note: "deep hydration" },
    ceramide:       { for: ["Dryness"], when: "PM", note: "repairs barrier" },
    retinol:        { for: ["Texture", "Pigmentation"], when: "PM", note: "cell turnover (start 2x/week)" },
    gentle_cleanser:{ for: [], when: "AM", note: "base cleanse" },
    spf:            { for: [], when: "AM", note: "non-negotiable — prevents pigmentation" },
  };

  function rankConcerns(scores) {
    return Object.entries(scores)
      .filter(([k]) => k !== "Radiance")            // radiance is a positive, not a concern
      .map(([k, v]) => ({ concern: k, score: v }))
      .sort((a, b) => b.score - a.score);
  }

  // conditions the ML classifier may output -> extra ingredient pushes / cautions
  const CONDITION_RULES = {
    Acne:        { add: ["salicylic_acid", "niacinamide"], caution: null },
    Fungal:      { add: ["gentle_cleanser"], caution: "Looks like fungal acne — avoid heavy oils/esters; antifungal + see a doctor if it spreads." },
    Eczema:      { add: ["ceramide", "centella"], caution: "Barrier-compromised — skip acids/retinol; see a dermatologist for flares." },
    Pigmentation:{ add: ["vitamin_c", "kojic_acid"], caution: null },
    Dryness:     { add: ["hyaluronic", "ceramide"], caution: null },
    Clear:       { add: [], caution: null },
  };

  function build(profile, condition, catalog) {
    catalog = catalog || SAMPLE_CATALOG;
    const ranked = rankConcerns(profile.scores);
    const topConcerns = ranked.filter((r) => r.score >= 40).map((r) => r.concern);

    // collect target ingredients from concerns
    const picks = new Set(["gentle_cleanser", "spf"]);
    for (const key of Object.keys(INGREDIENT)) {
      const meta = INGREDIENT[key];
      if (meta.for.some((c) => topConcerns.includes(c))) picks.add(key);
    }

    // condition-driven additions / cautions
    let caution = null;
    if (condition && CONDITION_RULES[condition.label]) {
      const rule = CONDITION_RULES[condition.label];
      rule.add.forEach((k) => picks.add(k));
      caution = rule.caution;
    }

    // slot into AM / PM
    const am = [], pm = [];
    picks.forEach((k) => {
      const meta = INGREDIENT[k];
      const cat = (catalog.ingredients[k]) || { product: k, url: "#" };
      const item = {
        key: k, note: meta.note, product: cat.product, url: cat.url,
        // live-brand fields (present only when a real catalog is connected)
        price: cat.price != null ? cat.price : null,
        image: cat.image || null, id: cat.id || null, variantId: cat.variantId || null,
      };
      (meta.when === "PM" ? pm : am).push(item);
    });
    // ensure cleanser first, spf last in AM
    am.sort((a, b) => (a.key === "gentle_cleanser" ? -1 : b.key === "gentle_cleanser" ? 1 : a.key === "spf" ? 1 : b.key === "spf" ? -1 : 0));

    return { brand: catalog.brand, ranked, topConcerns, am, pm, caution };
  }

  function toText(profile, condition, plan) {
    const L = [];
    L.push(`— ${plan.brand} · Skin Report —`);
    L.push(`Type: ${profile.type} | Tone: Monk ${profile.tone.monk} | Undertone: ${profile.undertone}`);
    if (condition && condition.label && condition.label !== "Clear")
      L.push(`Screened concern: ${condition.label} (${condition.confidence}% conf.)`);
    L.push(`Top concerns: ${plan.topConcerns.join(", ") || "none — balanced skin"}`);
    if (plan.caution) L.push(`⚠ ${plan.caution}`);
    L.push(`AM: ${plan.am.map((x) => x.product).join(" → ")}`);
    L.push(`PM: ${plan.pm.map((x) => x.product).join(" → ") || "moisturise"}`);
    L.push(`(Cosmetic guidance, not a diagnosis.)`);
    return L.join("\n");
  }

  global.Recommend = { build, toText, SAMPLE_CATALOG };
})(window);
