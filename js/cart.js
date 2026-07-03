/*
 * cart.js — the shopping cart.
 *
 * Collects the products a user wants from the recommended routine / catalog and
 * hands the finished order off to the BRAND'S OWN secure checkout. TwachaSense
 * never processes a payment itself — it assembles the cart and opens the brand's
 * store with the items prefilled:
 *   - Shopify:  https://{domain}/cart/{variantId}:{qty},{variantId}:{qty}
 *   - otherwise: opens each product's own product page to buy on the brand site.
 * This keeps all card/payment handling on the brand's PCI-compliant store.
 */
(function (global) {
  "use strict";

  const STORE_KEY = "twacha_cart_v1";
  let items = [];              // [{id, variantId, title, price, image, url, qty}]
  const listeners = [];

  function load() { try { const r = localStorage.getItem(STORE_KEY); if (r) items = JSON.parse(r); } catch (e) { items = []; } }
  function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(items)); } catch (e) {} emit(); }
  function emit() { listeners.forEach((fn) => fn(items)); }
  function onChange(fn) { listeners.push(fn); fn(items); }

  function add(p) {
    const key = p.id || p.title;
    const found = items.find((it) => (it.id || it.title) === key);
    if (found) found.qty += 1;
    else items.push({ id: p.id || p.title, variantId: p.variantId || null,
      title: p.title || p.product, price: Number(p.price) || 0,
      image: p.image || "", url: p.url || "#", qty: 1 });
    save();
  }
  function setQty(key, qty) {
    const it = items.find((x) => (x.id || x.title) === key);
    if (!it) return;
    it.qty = Math.max(0, qty);
    if (it.qty === 0) items = items.filter((x) => x !== it);
    save();
  }
  function remove(key) { items = items.filter((x) => (x.id || x.title) !== key); save(); }
  function clear() { items = []; save(); }
  function count() { return items.reduce((s, it) => s + it.qty, 0); }
  function total() { return items.reduce((s, it) => s + it.price * it.qty, 0); }
  function all() { return items; }

  // hand off to the brand's real checkout
  function checkout() {
    const brand = global.Brand && Brand.get();
    if (!items.length) return { ok: false, reason: "Cart is empty." };

    // Shopify permalink checkout when we have a domain + variant ids
    if (brand && brand.domain && items.every((it) => it.variantId)) {
      const pairs = items.map((it) => it.variantId + ":" + it.qty).join(",");
      const url = "https://" + brand.domain + "/cart/" + pairs;
      global.open(url, "_blank", "noopener");
      return { ok: true, mode: "shopify", url };
    }
    // fallback: open each product page on the brand's site to buy there
    const urls = items.map((it) => it.url).filter((u) => u && u !== "#");
    if (urls.length) {
      urls.forEach((u) => global.open(u, "_blank", "noopener"));
      return { ok: true, mode: "product-pages", opened: urls.length };
    }
    return { ok: false, reason: "Connect a brand store to check out (no product links available)." };
  }

  load();
  global.Cart = { add, setQty, remove, clear, count, total, all, onChange, checkout };
})(window);
