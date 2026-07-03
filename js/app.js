/*
 * app.js — orchestration: camera/upload -> ROI patch -> metrics + ML -> report.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const video = $("video"), frame = $("frame"), shot = $("shot");
  let stream = null;
  let lastPatch = null;   // canvas of the analysed ROI (for training too)

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
      analyse();
    };
    img.src = URL.createObjectURL(f);
  }

  // ---- analysis ------------------------------------------------------------
  let lastProfile = null, lastCondition = null, lastPlan = null;

  async function analyse() {
    if (!lastPatch) return;
    const imgData = lastPatch.getContext("2d").getImageData(0, 0, 224, 224);
    const profile = SkinMetrics.analyse(imgData);
    if (!profile.ok) { $("empty").hidden = false; $("report").hidden = true; $("empty").textContent = profile.reason; return; }

    lastProfile = profile;
    lastCondition = Classifier.isTrained() ? await Classifier.predict(lastPatch) : null;
    lastPlan = Recommend.build(profile, lastCondition, Recommend.SAMPLE_CATALOG);

    render(profile, lastCondition, lastPlan);
  }

  // One cohesive brand colour for normal scores; the warm accent appears ONLY
  // when a concern crosses a warning / critical threshold, so the eye is drawn to
  // what actually needs attention instead of a rainbow. Radiance is a POSITIVE
  // metric (higher = better skin) so it never turns "warning".
  const BAR_BRAND = "#8b93f8", BAR_WARN = "#fbbf24", BAR_CRIT = "#fb7185";
  function barColor(key, v) {
    if (key === "Radiance") return BAR_BRAND;   // high radiance is good, never a warning
    if (v >= 85) return BAR_CRIT;
    if (v >= 65) return BAR_WARN;
    return BAR_BRAND;
  }

  function render(p, cond, plan) {
    $("empty").hidden = true; $("report").hidden = false;

    $("mType").textContent = p.type;
    $("mToneTxt").textContent = "Monk " + p.tone.monk;
    $("toneChip").style.background = p.tone.hex;
    $("mUnder").textContent = p.undertone;

    // bars
    const bars = $("bars"); bars.innerHTML = "";
    Object.entries(p.scores).forEach(([k, v]) => {
      const el = document.createElement("div");
      el.className = "bar-item";
      el.innerHTML =
        `<span>${k}</span>
         <div class="bar-track"><div class="bar-fill" style="width:${v}%;background:${barColor(k, v)}"></div></div>
         <span class="bar-val">${v}</span>`;
      bars.appendChild(el);
    });

    // condition
    const co = $("conditionOut");
    if (cond) {
      const others = Object.entries(cond.confidences)
        .filter(([k]) => k !== cond.label)
        .sort((a, b) => b[1] - a[1]).slice(0, 3)
        .map(([k, v]) => `<li><span>${k}</span><span>${Math.round(v * 100)}%</span></li>`).join("");
      co.className = "condition";
      co.innerHTML =
        `<div class="cond-top">${cond.label} · ${cond.confidence}%</div>
         <ul class="cond-list">${others}</ul>`;
      $("mlState").textContent = `model: ${Classifier.totalExamples()} examples`;
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

    // routine
    const routine = $("routine"); routine.innerHTML = "";
    const mk = (title, items) => {
      const li = items.map((x) =>
        `<li>${x.product} <span class="ing">— ${x.note}</span></li>`).join("");
      return `<div class="slot"><h4>${title}</h4><ul>${li || "<li>moisturise</li>"}</ul></div>`;
    };
    let html = mk("☀ AM", plan.am) + mk("🌙 PM", plan.pm);
    if (plan.caution)
      html += `<div class="slot" style="grid-column:1/-1;border-color:#f8717155">
                 <h4 style="color:#f87171">⚠ Caution</h4>
                 <p style="margin:0;font-size:13px;line-height:1.5">${plan.caution}</p></div>`;
    routine.innerHTML = html;
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

  // ---- wire up -------------------------------------------------------------
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
