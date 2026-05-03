// Cloudflare Pages Advanced Mode worker.
//
// Implements the "Markdown for Agents" content-negotiation contract:
// when a client sends ``Accept: text/markdown``, this worker rewrites
// the request to fetch the corresponding pre-built ``.md`` rendition
// and returns it with ``Content-Type: text/markdown; charset=utf-8``,
// ``Vary: Accept``, and an approximate ``x-markdown-tokens`` header.
//
// Source markdown files are produced by ``scripts/render_site_markdown.py``
// and committed alongside the HTML pages they mirror.
//
// References:
//   - https://developers.cloudflare.com/fundamentals/reference/markdown-for-agents/
//   - https://developers.cloudflare.com/pages/functions/advanced-mode/
//
// Behavior:
//   - GET / HEAD with ``Accept: text/markdown`` and an existing ``.md``
//     rendition -> 200 with markdown body.
//   - Same request shape but no rendition file -> 406 Not Acceptable
//     with a JSON body listing the supported representations. Agents
//     that detect 406 should fall back to ``Accept: text/html``.
//   - Anything else -> pass through to static asset serving.

export default {
  async fetch(request, env) {
    const method = request.method;
    if (method !== "GET" && method !== "HEAD") {
      return env.ASSETS.fetch(request);
    }

    const accept = request.headers.get("accept") || "";
    if (!prefersMarkdown(accept)) {
      return env.ASSETS.fetch(request);
    }

    const url = new URL(request.url);
    const mdUrl = htmlPathToMarkdown(url);
    if (!mdUrl) {
      return env.ASSETS.fetch(request);
    }

    const mdRequest = new Request(mdUrl.toString(), {
      method,
      headers: new Headers(request.headers),
      redirect: "manual",
    });
    const mdResp = await env.ASSETS.fetch(mdRequest);

    if (mdResp.status !== 200) {
      // No rendition for this path. Per RFC 7231 §6.5.6, return 406
      // Not Acceptable so the client knows to retry without the
      // markdown preference.
      return notAcceptable(url);
    }

    // For HEAD, Cloudflare's static asset handler may already have
    // dropped the body — but we still need to compute headers based
    // on body length. Read body for GET only; for HEAD, trust
    // content-length from the upstream response if present.
    const body = method === "GET" ? await mdResp.text() : null;
    const contentLength =
      body !== null
        ? new TextEncoder().encode(body).byteLength
        : Number(mdResp.headers.get("content-length") || 0);
    const tokens = approximateTokenCount(body, contentLength);

    const headers = new Headers();
    headers.set("content-type", "text/markdown; charset=utf-8");
    headers.set("vary", "Accept");
    headers.set("x-markdown-tokens", String(tokens));
    headers.set("x-content-type-options", "nosniff");
    // Pass through cache-control if upstream set one; otherwise be
    // conservative.
    const cacheControl = mdResp.headers.get("cache-control");
    headers.set("cache-control", cacheControl || "public, max-age=300");
    if (body !== null) {
      headers.set("content-length", String(contentLength));
    }

    return new Response(method === "HEAD" ? null : body, {
      status: 200,
      headers,
    });
  },
};

// Parse an Accept header and return true iff ``text/markdown`` is the
// preferred representation. We deliberately do NOT match ``*/*`` or
// ``text/*`` — agents must opt in explicitly. This avoids surprising
// browsers that send ``Accept: text/html,*/*`` from being served
// markdown.
//
// Returns true when ``text/markdown`` appears with q > 0 AND has the
// highest q-value among the alternatives the client listed (ties
// resolve in markdown's favor since the client did ask for it).
export function prefersMarkdown(accept) {
  if (!accept) return false;
  let mdQ = -1;
  let bestOtherQ = -1;
  for (const raw of accept.split(",")) {
    const part = raw.trim();
    if (!part) continue;
    const segments = part.split(";").map((s) => s.trim());
    const type = segments[0].toLowerCase();
    let q = 1;
    for (const seg of segments.slice(1)) {
      if (seg.startsWith("q=")) {
        const parsed = parseFloat(seg.slice(2));
        if (!Number.isNaN(parsed)) q = parsed;
      }
    }
    if (q <= 0) continue;
    if (type === "text/markdown") {
      if (q > mdQ) mdQ = q;
    } else if (type !== "*/*" && type !== "text/*") {
      // Any concrete non-markdown type counts as "competing".
      if (q > bestOtherQ) bestOtherQ = q;
    }
  }
  if (mdQ < 0) return false;
  return mdQ >= bestOtherQ;
}

// Map an HTML route to the ``.md`` rendition path. Returns a URL or
// ``null`` if the path doesn't look like an HTML route.
//
// Mappings (Cloudflare Pages serves directories via ``index.html``):
//   ``/``                       -> ``/index.md``
//   ``/foo/``                   -> ``/foo/index.md``
//   ``/foo/bar.html``           -> ``/foo/bar.md``
//   ``/foo`` (extensionless)    -> ``/foo/index.md``
//   ``/foo.png``                -> null (asset)
export function htmlPathToMarkdown(url) {
  let path = url.pathname;
  if (!path) return null;

  if (path.endsWith("/")) {
    path = path + "index.md";
  } else if (path.endsWith(".html")) {
    path = path.slice(0, -5) + ".md";
  } else {
    const lastSegment = path.split("/").pop() || "";
    if (lastSegment.includes(".")) {
      // Has a non-html extension — not a page route.
      return null;
    }
    path = path + "/index.md";
  }
  const out = new URL(url.toString());
  out.pathname = path;
  out.search = "";
  return out;
}

// Approximate the OpenAI/Anthropic-style token count for a markdown
// body. ``cl100k_base`` averages ~4 chars per token for English; we
// use the same rule of thumb. This is ADVISORY — agents that need an
// exact count should run their own tokenizer.
export function approximateTokenCount(body, byteLength) {
  if (body !== null && body !== undefined) {
    return Math.max(1, Math.ceil(body.length / 4));
  }
  // HEAD path — only byteLength is available.
  return Math.max(1, Math.ceil(byteLength / 4));
}

function notAcceptable(url) {
  const body = JSON.stringify(
    {
      error: "not_acceptable",
      message:
        "No markdown rendition is published for this path. Retry with " +
        "Accept: text/html.",
      path: url.pathname,
      supported_media_types: ["text/html"],
    },
    null,
    2,
  );
  return new Response(body, {
    status: 406,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "vary": "Accept",
    },
  });
}
