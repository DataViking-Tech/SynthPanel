/**
 * Tailwind config for the althing.dev static site (site/).
 *
 * This replaces the former Play-CDN inline `tailwind.config = {...}` blocks
 * that every page carried (identical on all pages): the only customisation
 * is the explicit system mono font stack.
 *
 * Build: `npm ci && npm run build` from scripts/site_tailwind/.
 * Output: site/assets/tailwind.css (committed; drift-guarded in CI).
 */
module.exports = {
  content: [
    // Every published page, plus the template index.html is rendered from —
    // scanning both means a template edit alone yields a correct rebuild.
    // site/ contains no node toolchain (it deploys raw to Cloudflare Pages),
    // so the recursive glob is safe.
    "../../site/**/*.html",
    "../../site/index.html.j2",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
};
