/*
 * brand_catalog.js — the ON-DEMAND brand link-up.
 *
 * There is deliberately NO visible "connect a store" UI. Instead, a brand is
 * linked here: give me a brand name + its product catalogue (or a Shopify
 * domain) and I fill in one of the globals below. On load the app auto-links it,
 * so the post-diagnosis routine recommends that brand's REAL products and the
 * cart checks out on the brand's own store. Leave both null → neutral demo catalog.
 *
 *   Option A — paste the catalogue (works fully offline, no CORS):
 *     window.BRAND_CATALOG = {
 *       brand: "Acme Skincare",
 *       domain: "acme.myshopify.com",      // optional — enables 1-click cart checkout
 *       currency: "INR",
 *       products: [
 *         { id:"1", variantId:"v1", title:"2% Salicylic Acid Toner", price:499,
 *           url:"https://acme.myshopify.com/products/bha", tags:["salicylic"] },
 *         { id:"2", variantId:"v2", title:"10% Niacinamide Serum", price:599,
 *           url:"https://acme.myshopify.com/products/nia", tags:["niacinamide"] }
 *         // …one row per SKU; `tags`/title keywords map it to a routine slot
 *       ]
 *     };
 *
 *   Option B — just a Shopify store (fetched from its public /products.json):
 *     window.BRAND_SHOPIFY = "acme.myshopify.com";
 */
window.BRAND_CATALOG = null;   // ← give me a brand + catalogue and I fill this in
window.BRAND_SHOPIFY = null;   // ← …or just a Shopify domain
