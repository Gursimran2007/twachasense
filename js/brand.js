/*
 * brand.js — the OPTIONAL online brand-integration layer.
 *
 * The ML/CV core stays 100% offline. This module is the only part that touches
 * the network, and only when a brand explicitly connects its store. It pulls a
 * brand's real product catalog, normalises it, and maps each SKU to an
 * ingredient "slot" the recommender already understands — so TwachaSense goes
 * from a demo catalog to recommending a brand's ACTUAL products for purchase.
 *
 * Three connect paths (a brand picks whatever it can give us):
 *   1. Pasted catalog JSON            (no network / no CORS — always works)
 *   2. A catalog JSON URL             (fetched; needs CORS-open hosting)
 *   3. A Shopify store domain         (fetched via the public /products.json)
 *
 * Nothing here ever takes a payment. Checkout is handed off to the brand's own
 * secure store (see cart.js) — we only assemble the order and open their site.
 */
(function (global) {
  "use strict";

  const STORE_KEY = "twacha_brand_v1";

  // ingredient slot -> keywords that identify a product for that slot
  const SLOT_KEYWORDS = {
    salicylic_acid:  ["salicylic", "bha", "acne", "blemish", "spot control", "clarifying"],
    niacinamide:     ["niacinamide", "sebum", "pore", "oil control"],
    centella:        ["centella", "cica", "madecassoside", "soothing", "calming"],
    vitamin_c:       ["vitamin c", "vit c", "ascorb", "brighten", "glow"],
    kojic_acid:      ["kojic", "dark spot", "pigment", "arbutin", "tranexamic"],
    hyaluronic:      ["hyaluron", "hydra", "hydrating", "moisture boost"],
    ceramide:        ["ceramide", "barrier", "repair", "moisturiz", "moisturis", "cream"],
    spf:             ["spf", "sunscreen", "sun ", "uv "],
    retinol:         ["retinol", "retinal", "retinoid", "bakuchiol"],
    gentle_cleanser: ["cleanser", "face wash", "facewash", "cleansing"],
  };

  let state = null;   // { brand, domain, currency, products:[], ingredients:{} }

  function loadSaved() {
    try { const raw = localStorage.getItem(STORE_KEY); if (raw) state = JSON.parse(raw); }
    catch (e) { state = null; }
    return state;
  }
  function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (e) {} }
  function disconnect() { state = null; localStorage.removeItem(STORE_KEY); }
  function get() { return state; }
  function isConnected() { return !!(state && state.products && state.products.length); }

  function slotFor(title, tags) {
    const hay = (title + " " + (tags || []).join(" ")).toLowerCase();
    for (const slot of Object.keys(SLOT_KEYWORDS)) {
      if (SLOT_KEYWORDS[slot].some((kw) => hay.includes(kw))) return slot;
    }
    return null;
  }

  // ---- normalisers ---------------------------------------------------------
  function normProduct(p) {
    const tags = Array.isArray(p.tags)
      ? p.tags
      : (typeof p.tags === "string" ? p.tags.split(",").map((s) => s.trim()) : []);
    return {
      id: String(p.id != null ? p.id : p.variantId || p.title),
      variantId: p.variantId != null ? String(p.variantId) : null,
      title: p.title || "Untitled",
      price: Number(p.price) || 0,
      image: p.image || "",
      url: p.url || "#",
      tags,
      slot: p.slot || slotFor(p.title || "", tags),
    };
  }

  // Shopify /products.json -> our product shape
  function fromShopify(json, domain) {
    const out = [];
    (json.products || []).forEach((p) => {
      const v = (p.variants && p.variants[0]) || {};
      const img = (p.images && p.images[0] && p.images[0].src) || (p.image && p.image.src) || "";
      out.push(normProduct({
        id: p.id, variantId: v.id, title: p.title,
        price: v.price, image: img,
        url: "https://" + domain + "/products/" + p.handle,
        tags: p.tags,
      }));
    });
    return out;
  }

  // Build the { ingredients: {slot: product} } map the recommender resolves
  // against — one representative (cheapest) product per slot.
  function buildIngredients(products) {
    const bySlot = {};
    products.forEach((p) => {
      if (!p.slot) return;
      if (!bySlot[p.slot] || p.price < bySlot[p.slot].price) {
        bySlot[p.slot] = {
          product: p.title, url: p.url, price: p.price,
          image: p.image, id: p.id, variantId: p.variantId,
        };
      }
    });
    return bySlot;
  }

  function adopt(brand, domain, currency, products) {
    const mapped = products.map(normProduct);
    state = {
      brand: brand || "Connected brand",
      domain: domain || null,
      currency: currency || "INR",
      products: mapped,
      ingredients: buildIngredients(mapped),
      connectedAt: new Date().toISOString(),
    };
    save();
    return state;
  }

  // ---- public connect ------------------------------------------------------
  // input: { mode:"json"|"url"|"shopify", value:string }
  async function connect(input) {
    const { mode, value } = input;
    if (mode === "json") {
      const json = typeof value === "string" ? JSON.parse(value) : value;
      return adopt(json.brand, json.domain, json.currency, json.products || []);
    }
    if (mode === "url") {
      const json = await fetch(value, { mode: "cors" }).then((r) => {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
      // support both our schema and a raw Shopify dump at the URL
      if (json.products && json.products[0] && json.products[0].variants) {
        const domain = new URL(value).host;
        return adopt(domain, domain, "USD", fromShopify(json, domain));
      }
      return adopt(json.brand, json.domain, json.currency, json.products || []);
    }
    if (mode === "shopify") {
      let domain = value.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
      const url = "https://" + domain + "/products.json?limit=250";
      const json = await fetch(url, { mode: "cors" }).then((r) => {
        if (!r.ok) throw new Error("HTTP " + r.status + " — store may block cross-origin access");
        return r.json();
      });
      const products = fromShopify(json, domain);
      if (!products.length) throw new Error("No products found at " + url);
      return adopt(domain, domain, "USD", products);
    }
    throw new Error("Unknown connect mode: " + mode);
  }

  // products grouped by the concern/slot they serve, for browsing the catalog
  function productsForSlots(slots) {
    if (!isConnected()) return [];
    const wanted = new Set(slots);
    return state.products.filter((p) => p.slot && wanted.has(p.slot));
  }

  loadSaved();

  global.Brand = {
    connect, disconnect, get, isConnected, productsForSlots,
    SLOT_KEYWORDS, _slotFor: slotFor,
  };
})(window);
