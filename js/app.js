/*
 * app.js — orchestration: camera/upload -> ROI patch -> metrics + ML -> report.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const video = $("video"), frame = $("frame"), shot = $("shot");
  let stream = null;
  let lastPatch = null;   // canvas of the analysed ROI (for training too)
  let lastFull = null;    // 160² whole-photo canvas for the face-level real models

  // Whole-photo canvas, stretched to 160² to EXACTLY match how the real-model
  // trainer (train/build.html) rendered each image — feature parity is essential.
  function grabFull(sourceEl, sw, sh) {
    const c = document.createElement("canvas");
    c.width = 160; c.height = 160;
    c.getContext("2d", { willReadFrequently: true }).drawImage(sourceEl, 0, 0, sw, sh, 0, 0, 160, 160);
    return c;
  }

  // ---- camera --------------------------------------------------------------
  async function startCam() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 }, audio: false,
      });
      video.srcObject = stream;
      video.hidden = false; shot.hidden = true;
      $("btnSnap").disabled = false;
      $("btnCam").textContent = "Restart camera";
    } catch (e) {
      alert("Camera unavailable — use Upload instead.\n" + e.message);
    }
  }

  // Extract a centred circular-ish ROI, downscaled to 224 for the CNN.
  function grabPatch(sourceEl, sw, sh) {
    const size = Math.min(sw, sh) * 0.46;
    const sx = (sw - size) / 2, sy = (sh - size) / 2;
    const c = document.createElement("canvas");
    c.width = 224; c.height = 224;
    const ctx = c.getContext("2d");
    ctx.drawImage(sourceEl, sx, sy, size, size, 0, 0, 224, 224);
    return c;
  }

  function snap() {
    frame.width = video.videoWidth; frame.height = video.videoHeight;
    frame.getContext("2d").drawImage(video, 0, 0);
    shot.src = frame.toDataURL("image/jpeg");
    shot.hidden = false; video.hidden = true;
    lastPatch = grabPatch(frame, frame.width, frame.height);
    lastFull = grabFull(frame, frame.width, frame.height);
    analyse();
  }

  function onFile(ev) {
    const f = ev.target.files[0];
    if (!f) return;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.width; c.height = img.height;
      c.getContext("2d").drawImage(img, 0, 0);
      shot.src = c.toDataURL("image/jpeg");
      shot.hidden = false; video.hidden = true;
      lastPatch = grabPatch(c, img.width, img.height);
      lastFull = grabFull(c, img.width, img.height);
      analyse();
    };
    img.src = URL.createObjectURL(f);
  }

  // ---- analysis ------------------------------------------------------------
  let lastProfile = null, lastCondition = null, lastPlan = null, lastReal = null, lastRefer = null;

  // adapt CondModels.score() -> the {label, confidence, confidences} shape the
  // KNN classifier and the recommender already speak (confidences are 0..1).
  function condFromReal(r) {
    if (!r) return null;
    const confidences = {};
    Object.keys(r.probs).forEach((k) => (confidences[k] = r.probs[k] / 100));
    return { label: r.label, confidence: r.confidence, confidences, source: "real" };
  }

  async function analyse() {
    if (!lastPatch) return;
    const imgData = lastPatch.getContext("2d").getImageData(0, 0, 224, 224);
    const profile = SkinMetrics.analyse(imgData);
    if (!profile.ok) { $("empty").hidden = false; $("report").hidden = true; $("empty").textContent = profile.reason; return; }

    lastProfile = profile;
    // real-data condition model (trained on ~11k real derm photos) + non-diagnostic
    // derm-referral head. Always available — no seeding needed.
    const realCond = (window.CondModels && lastPatch) ? CondModels.score(lastPatch) : null;
    lastRefer = realCond;
    // If the brand has taught the few-shot KNN on its own photos, that personalised
    // signal wins; otherwise fall back to the shipped real-data model.
    lastCondition = Classifier.isTrained()
      ? await Classifier.predict(lastPatch)
      : condFromReal(realCond);
    // real-data face-level estimates from the whole photo (may be null on a
    // tight crop with too little skin — the UI hides the section in that case)
    lastReal = (lastFull && window.RealModels)
      ? RealModels.score(lastFull.getContext("2d").getImageData(0, 0, 160, 160)) : null;
    // If a brand store is connected, resolve the routine against ITS real catalog
    // (live SKUs + prices) instead of the demo catalog.
    const catalog = (window.Brand && Brand.isConnected())
      ? { brand: Brand.get().brand, ingredients: Brand.get().ingredients }
      : Recommend.SAMPLE_CATALOG;
    lastPlan = Recommend.build(profile, lastCondition, catalog);

    render(profile, lastCondition, lastPlan, lastReal);
  }

  // One cohesive brand colour for normal scores; the warm accent appears ONLY
  // when a concern crosses a warning / critical threshold, so the eye is drawn to
  // what actually needs attention instead of a rainbow. Radiance is a POSITIVE
  // metric (higher = better skin) so it never turns "warning".
  // plain-language callout per screened condition (what the user actually sees)
  const CALLOUT = {
    Acne: "Active acne / breakouts",
    Fungal: "Fungal-type acne pattern",
    Eczema: "Eczema-like irritation",
    Pigmentation: "Pigmentation / dark spots",
    Dryness: "Dry, dehydrated skin",
  };
  // confidence -> severity word + class for the coloured badge
  function severityFor(conf) {
    if (conf >= 70) return { word: "High signal", cls: "high" };
    if (conf >= 45) return { word: "Moderate", cls: "mod" };
    return { word: "Mild", cls: "low" };
  }

  const BAR_BRAND = "#8b93f8", BAR_WARN = "#fbbf24", BAR_CRIT = "#fb7185";
  function barColor(key, v) {
    if (key === "Radiance") return BAR_BRAND;   // high radiance is good, never a warning
    if (v >= 85) return BAR_CRIT;
    if (v >= 65) return BAR_WARN;
    return BAR_BRAND;
  }

  // draw a set of {label:value} into a .bars container. `positive` keys are the
  // ones where higher = better (Radiance, Clarity) so they never turn "warning".
  function drawBars(container, entries, positive) {
    container.innerHTML = "";
    entries.forEach(([k, v]) => {
      const isPos = positive && positive.has(k);
      const color = isPos ? BAR_BRAND : barColor(k, v);
      const el = document.createElement("div");
      el.className = "bar-item";
      el.innerHTML =
        `<span>${k}</span>
         <div class="bar-track"><div class="bar-fill" style="width:${v}%;background:${color}"></div></div>
         <span class="bar-val">${v}</span>`;
      container.appendChild(el);
    });
  }

  function render(p, cond, plan, real) {
    $("empty").hidden = true; $("report").hidden = false;

    $("mType").textContent = p.type;
    $("mToneTxt").textContent = "Monk " + p.tone.monk;
    $("toneChip").style.background = p.tone.hex;
    $("mUnder").textContent = p.undertone;

    // CV concern bars
    drawBars($("bars"), Object.entries(p.scores), new Set(["Radiance"]));

    // real-data face-level estimates (Clarity higher = better; Dark circles /
    // Wrinkles higher = more concern). Hidden when the photo has too little skin.
    if (real) {
      drawBars($("realBars"),
        [["Clarity", real.clarity], ["Dark circles", real.darkCircles], ["Wrinkles", real.wrinkles]],
        new Set(["Clarity"]));
      $("realBars").previousElementSibling.style.display = "";
      $("realBars").style.display = "";
    } else {
      $("realBars").innerHTML = "";
      $("realBars").style.display = "none";
      $("realBars").previousElementSibling.style.display = "none";
    }

    // condition — a detected concern (anything other than "Clear") is surfaced
    // PROMINENTLY: a coloured severity badge + a plain callout of what was seen,
    // so acne / fungal / eczema / pigmentation don't hide in a faint list.
    const co = $("conditionOut");
    if (cond) {
      const others = Object.entries(cond.confidences)
        .filter(([k]) => k !== cond.label)
        .sort((a, b) => b[1] - a[1]).slice(0, 3)
        .map(([k, v]) => `<li><span>${k}</span><span>${Math.round(v * 100)}%</span></li>`).join("");
      const detected = cond.label && cond.label !== "Clear";
      const sev = severityFor(cond.confidence);   // {word,cls}
      co.className = detected ? "condition flagged sev-" + sev.cls : "condition clear";
      const badge = detected
        ? `<span class="cond-badge sev-${sev.cls}">${sev.word}</span>`
        : `<span class="cond-badge clear">Looks clear</span>`;
      const headline = detected
        ? `${CALLOUT[cond.label] || cond.label} detected`
        : `No cosmetic concern flagged`;
      co.innerHTML =
        `<div class="cond-head">
           <div class="cond-top">${cond.label} · ${cond.confidence}%</div>${badge}
         </div>
         <p class="cond-callout">${headline}</p>
         <ul class="cond-list">${others}</ul>`;
      $("mlState").textContent = cond.source === "real"
        ? "real model · ~11k derm photos"
        : `model: ${Classifier.totalExamples()} examples (personalised)`;
    } else {
      co.className = "condition locked";
      co.innerHTML =
        `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
           <rect x="4" y="10" width="16" height="10" rx="2" stroke="currentColor" stroke-width="1.8"/>
           <path d="M8 10V7a4 4 0 0 1 8 0v3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
         </svg>
         <span>Seed the baseline or train the model below to unlock condition screening.</span>`;
      $("mlState").textContent = "model: not trained";
    }

    // NON-DIAGNOSTIC derm-referral nudge. The binary head (AUC ~0.70) fires when
    // the skin pattern looks unlike ordinary cosmetic concerns. It NEVER names a
    // disease — it only suggests a professional look. Shown above ~55% to keep it
    // a genuine signal, not a blanket disclaimer.
    const nudge = $("referNudge");
    if (nudge) {
      if (lastRefer && lastRefer.refer >= 55) {
        $("referTxt").textContent =
          "This skin pattern looks different from ordinary cosmetic concerns. " +
          "This is a cosmetic screen, not a diagnosis — consider having it looked at by a dermatologist.";
        nudge.hidden = false;
      } else {
        nudge.hidden = true;
      }
    }

    // routine — when a brand store is connected each item shows its real price
    // and an "Add" button that drops the SKU into the cart.
    const connected = window.Brand && Brand.isConnected();
    const cur = connected ? (Brand.get().currency || "INR") : "INR";
    const routine = $("routine"); routine.innerHTML = "";
    const line = (x) => {
      const priced = connected && x.price != null && x.price > 0;
      const buy = priced
        ? `<button class="add-cart" data-key="${x.key}">${money(x.price, cur)} · Add</button>`
        : "";
      return `<li><div class="rx-line"><span class="rx-name">${x.product}
                <span class="ing">— ${x.note}</span></span>${buy}</div></li>`;
    };
    const mk = (title, items) => {
      const li = items.map(line).join("");
      return `<div class="slot"><h4>${title}</h4><ul>${li || "<li>moisturise</li>"}</ul></div>`;
    };
    let html = mk("☀ AM", plan.am) + mk("🌙 PM", plan.pm);
    if (plan.caution)
      html += `<div class="slot" style="grid-column:1/-1;border-color:#f8717155">
                 <h4 style="color:#f87171">⚠ Caution</h4>
                 <p style="margin:0;font-size:13px;line-height:1.5">${plan.caution}</p></div>`;
    routine.innerHTML = html;

    // wire the Add buttons to the cart, resolving each key back to its plan item
    if (connected) {
      const byKey = {};
      plan.am.concat(plan.pm).forEach((x) => (byKey[x.key] = x));
      routine.querySelectorAll(".add-cart").forEach((b) =>
        b.addEventListener("click", () => {
          const x = byKey[b.dataset.key];
          if (!x) return;
          Cart.add({ id: x.id, variantId: x.variantId, title: x.product,
            price: x.price, image: x.image, url: x.url });
          b.textContent = "Added ✓";
          setTimeout(() => (b.textContent = money(x.price, cur) + " · Add"), 1200);
        }));
    }
  }

  const CURSYM = { INR: "₹", USD: "$", GBP: "£", EUR: "€" };
  function money(v, cur) {
    const s = CURSYM[cur] || (cur ? cur + " " : "");
    return s + (Math.round(Number(v) * 100) / 100);
  }

  // ---- training UI ---------------------------------------------------------
  function buildClassCards() {
    const row = $("classRow"); row.innerHTML = "";
    const c = Classifier.counts();
    Classifier.CLASSES.forEach((name, idx) => {
      const card = document.createElement("div");
      card.className = "class-card";
      card.innerHTML =
        `<div class="name">${name}</div>
         <div class="count" id="cnt-${idx}">${c[name] || 0} samples</div>
         <button class="btn add" data-idx="${idx}">+ Add current</button>`;
      row.appendChild(card);
    });
    row.querySelectorAll(".add").forEach((b) =>
      b.addEventListener("click", () => addSample(+b.dataset.idx)));
  }

  function addSample(idx) {
    if (!Classifier.ready) return alert("ML still loading…");
    if (!lastPatch) return alert("Capture or upload a photo first.");
    Classifier.addExample(lastPatch, idx);
    const c = Classifier.counts();
    $("cnt-" + idx).textContent = (c[Classifier.CLASSES[idx]] || 0) + " samples";
    $("btnTrain").disabled = false;
    $("trainStat").textContent =
      `${Classifier.totalExamples()} total examples across ${Object.values(c).filter((v) => v > 0).length} classes`;
  }

  async function retrain() {
    // KNN is instance-based: adding examples already "trains" it. This just
    // re-runs prediction on the current photo so the user sees the effect.
    $("trainStat").textContent = "Model updated ✓ — re-scoring current photo…";
    await analyse();
    $("trainStat").textContent = `Model live · ${Classifier.totalExamples()} examples`;
  }

  function seedBaseline() {
    if (!Classifier.ready) return alert("ML still loading…");
    if (Classifier.totalExamples() > 0 &&
        !confirm("Add the tone-diverse synthetic Indian-skin baseline on top of the current model?")) return;
    $("btnSeed").disabled = true;
    $("trainStat").textContent = "Seeding tone-diverse baseline…";
    // defer so the UI can paint the "seeding…" text before the heavy loop
    setTimeout(() => {
      const total = SeedData.seed(Classifier, (done, t) => {
        if (done % 24 === 0 || done === t) $("trainStat").textContent = `Seeding… ${done}/${t}`;
      });
      buildClassCards();
      $("btnTrain").disabled = false;
      $("btnSeed").disabled = false;
      $("trainStat").textContent =
        `Baseline seeded: ${total} synthetic patches across 50 Indian tones × 6 conditions (severity + lighting augmented).`;
      analyse();
    }, 30);
  }

  function resetModel() {
    if (!confirm("Erase the trained model from this browser?")) return;
    Classifier.reset();
    buildClassCards();
    $("btnTrain").disabled = true;
    $("trainStat").textContent = "Model reset.";
    analyse();
  }

  // ---- report --------------------------------------------------------------
  function copyReport() {
    if (!lastProfile) return;
    const txt = Recommend.toText(lastProfile, lastCondition, lastPlan);
    navigator.clipboard.writeText(txt).then(
      () => { $("btnReport").textContent = "Copied ✓"; setTimeout(() => ($("btnReport").textContent = "Copy shareable report"), 1500); },
      () => alert(txt)
    );
  }

  // ---- brand link-up (no visible UI) ---------------------------------------
  // A brand is linked on demand via js/brand_catalog.js (BRAND_CATALOG paste or
  // BRAND_SHOPIFY domain). When present, the routine resolves against the brand's
  // REAL SKUs; otherwise it stays on the neutral demo catalog. No connect panel.
  async function autoLinkBrand() {
    if (!window.Brand) return;
    try {
      if (window.BRAND_CATALOG) {
        const s = await Brand.connect({ mode: "json", value: JSON.stringify(window.BRAND_CATALOG) });
        $("brandBadge").textContent = s.brand;
      } else if (window.BRAND_SHOPIFY) {
        const s = await Brand.connect({ mode: "shopify", value: window.BRAND_SHOPIFY });
        $("brandBadge").textContent = s.brand;
      } else if (Brand.isConnected()) {
        $("brandBadge").textContent = Brand.get().brand;   // restored from a prior session
      }
    } catch (e) { /* fall back silently to the demo catalog */ }
    if (lastProfile) analyse();
  }

  // ---- cart ----------------------------------------------------------------
  function openCart() { $("cartOverlay").hidden = false; $("cartDrawer").hidden = false; }
  function closeCart() { $("cartOverlay").hidden = true; $("cartDrawer").hidden = true; }

  function renderCart(items) {
    const box = $("cartItems");
    const cur = (window.Brand && Brand.isConnected() && Brand.get().currency) || "INR";
    $("cartCount").textContent = Cart.count();
    $("cartCount").hidden = Cart.count() === 0;
    if (!items.length) {
      box.innerHTML = `<p class="cart-empty">Cart is empty. Add products from your routine.</p>`;
      $("cartTotal").textContent = "—";
      $("btnCheckout").disabled = true;
      return;
    }
    box.innerHTML = items.map((it) => {
      const key = it.id || it.title;
      const img = it.image
        ? `<img src="${it.image}" alt="" class="ci-img" />`
        : `<div class="ci-img ci-noimg"></div>`;
      return `<div class="cart-item">
        ${img}
        <div class="ci-body">
          <div class="ci-title">${it.title}</div>
          <div class="ci-price">${money(it.price, cur)}</div>
          <div class="ci-qty">
            <button class="qtybtn" data-key="${key}" data-d="-1">–</button>
            <span>${it.qty}</span>
            <button class="qtybtn" data-key="${key}" data-d="1">+</button>
            <button class="ci-remove" data-key="${key}">Remove</button>
          </div>
        </div></div>`;
    }).join("");
    $("cartTotal").textContent = money(Cart.total(), cur);
    $("btnCheckout").disabled = false;
    box.querySelectorAll(".qtybtn").forEach((b) => b.addEventListener("click", () => {
      const it = Cart.all().find((x) => (x.id || x.title) === b.dataset.key);
      if (it) Cart.setQty(b.dataset.key, it.qty + Number(b.dataset.d));
    }));
    box.querySelectorAll(".ci-remove").forEach((b) =>
      b.addEventListener("click", () => Cart.remove(b.dataset.key)));
  }

  function doCheckout() {
    const r = Cart.checkout();
    if (!r.ok) { $("btnCheckout").textContent = r.reason; setTimeout(() => ($("btnCheckout").textContent = "Checkout on brand store"), 2200); }
  }

  // ---- wire up -------------------------------------------------------------
  $("btnCart").addEventListener("click", openCart);
  $("btnCartClose").addEventListener("click", closeCart);
  $("cartOverlay").addEventListener("click", closeCart);
  $("btnCheckout").addEventListener("click", doCheckout);
  Cart.onChange(renderCart);
  autoLinkBrand();

  $("btnCam").addEventListener("click", startCam);
  $("btnSnap").addEventListener("click", snap);
  $("file").addEventListener("change", onFile);
  $("btnSeed").addEventListener("click", seedBaseline);
  $("btnTrain").addEventListener("click", retrain);
  $("btnClear").addEventListener("click", resetModel);
  $("btnReport").addEventListener("click", copyReport);

  buildClassCards();
  Classifier.init((s) => { $("tfState").textContent = s; })
    .then(() => { buildClassCards(); if (Classifier.isTrained()) { $("btnTrain").disabled = false; analyse(); } });
})();
