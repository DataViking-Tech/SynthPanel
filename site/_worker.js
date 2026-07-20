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

// PostHog analytics snippet, injected into the <head> of every HTML
// response by injectAnalytics() below. The project API key is a
// PUBLISHABLE key (safe to commit). The init call is gated to the
// production hostname so local dev, CI, and *.pages.dev preview
// deploys never emit events.
const POSTHOG_SNIPPET = `<script>!function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture identify alias people set set_once set_config register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing reset isFeatureEnabled onFeatureFlags getFeatureFlag getFeatureFlagPayload reloadFeatureFlags group updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures getActiveMatchingSurveys getSurveys getNextSurveyStep onSessionId setPersonProperties startSessionRecording stopSessionRecording captureException".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);var h=window.location.hostname;if(h==="althing.dev"||h.endsWith(".althing.dev")){posthog.init("phc_AgGKBvc4HRVhFvN44pnF4vHhf2H3q8eHeRqF73BRLiyg",{api_host:"https://us.i.posthog.com",ui_host:"https://us.posthog.com",defaults:"2025-05-24",person_profiles:"identified_only"});}(function(){var SIBLINGS=["dataviking.tech","althing.dev","synthbench.org","traitprint.com"],SOURCE="althing";function bare(x){return x.replace(/^www\\./,"");}function decorate(){var self=bare(location.hostname),links=document.getElementsByTagName("a");for(var i=0;i<links.length;i++){var a=links[i],href=a.getAttribute("href");if(!href)continue;var u;try{u=new URL(href,location.href)}catch(e){continue}if(u.protocol!=="http:"&&u.protocol!=="https:")continue;var host=bare(u.hostname);if(host===self)continue;var sib=false;for(var j=0;j<SIBLINGS.length;j++){if(host===SIBLINGS[j]||host.slice(-(SIBLINGS[j].length+1))==="."+SIBLINGS[j]){sib=true;break}}if(!sib||u.searchParams.has("utm_source"))continue;u.searchParams.set("utm_source",SOURCE);u.searchParams.set("utm_medium","referral");u.searchParams.set("utm_campaign","cross-site");a.setAttribute("href",u.toString())}}if(document.readyState!=="loading")decorate();else document.addEventListener("DOMContentLoaded",decorate);})();</script>`;

// Inject the PostHog snippet into the <head> of HTML responses only.
// Guarded on content-type so JSON / markdown / binary assets pass
// through untouched (HTMLRewriter would otherwise mangle them).
async function injectAnalytics(response) {
  const ct = response.headers.get("content-type") || "";
  if (!ct.includes("text/html")) return response;
  return new HTMLRewriter()
    .on("head", {
      element(el) {
        el.append(POSTHOG_SNIPPET, { html: true });
      },
    })
    .transform(response);
}

export default {
  async fetch(request, env) {
    const method = request.method;
    if (method !== "GET" && method !== "HEAD") {
      return injectAnalytics(await env.ASSETS.fetch(request));
    }

    const accept = request.headers.get("accept") || "";
    if (!prefersMarkdown(accept)) {
      return injectAnalytics(await env.ASSETS.fetch(request));
    }

    const url = new URL(request.url);
    const mdUrl = htmlPathToMarkdown(url);
    if (!mdUrl) {
      return injectAnalytics(await env.ASSETS.fetch(request));
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
    // Pass through cache-control if upstream set one; otherwise default
    // to the HTML bucket per dvi-25f cache-control-policy.md (markdown
    // renditions are content variants of HTML pages).
    const cacheControl = mdResp.headers.get("cache-control");
    headers.set("cache-control", cacheControl || "private, no-store, must-revalidate");
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
