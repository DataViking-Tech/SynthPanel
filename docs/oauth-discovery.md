# OAuth / OIDC Discovery — Not Applicable

**Status:** N/A — SynthPanel publishes no `/.well-known/openid-configuration`
or `/.well-known/oauth-authorization-server` document.

**Bead:** sy-iaf (GH#416, AR-5)

**Skill referenced by the request:**
[oauth-discovery SKILL.md](https://isitagentready.com/.well-known/agent-skills/oauth-discovery/SKILL.md)

## Why

The agent-ready intake task for AR-5 is conditional:

> **If protected APIs exist**, publish `/.well-known/openid-configuration`
> (OIDC) or `/.well-known/oauth-authorization-server` (OAuth 2.0) with
> `issuer`, `authorization_endpoint`, `token_endpoint`, `jwks_uri`,
> `grant_types_supported`.

SynthPanel has no protected APIs to advertise. Each surface is unauthenticated
or out of scope for OAuth discovery:

| Surface | Transport | Auth model |
|---|---|---|
| `synthpanel` CLI | local process | none — runs on the user's machine |
| `synthpanel mcp-serve` | stdio (per Model Context Protocol) | none — bound to the spawning client |
| [`synthpanel.dev`](https://synthpanel.dev) | static site (Cloudflare Pages) | none — public docs only |

Provider credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`,
`GOOGLE_API_KEY` / `GEMINI_API_KEY`) are user-supplied and travel directly
from the local CLI to the upstream provider; SynthPanel itself never brokers
an OAuth flow.

## When this should be revisited

Re-open this if any of the following ship:

- A hosted SynthPanel API (HTTP) requiring per-user authentication.
- An MCP transport other than stdio (e.g. HTTP/SSE) that fronts a
  multi-tenant deployment.
- A first-party identity provider issuing tokens for SynthPanel resources.

In those cases the discovery document(s) above become required and AR-5
graduates from N/A to a build task.

## Related

- AR-1 — Link response headers (sy-7r1)
- AR-3 — Content-Signal in robots.txt (sy-7gl, merged)
- AR-4 — `/.well-known/api-catalog` (sy-czo)
- AR-6 — OAuth Protected Resource Metadata, RFC 9728 (sy-4nf) — also blocked
  on the "protected resource exists" precondition.
