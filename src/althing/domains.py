"""Domain prompt templates and generated persona validation.

Domain templates guide LLM persona generation toward realistic distributions
for common research domains.  The validation layer checks a panel for
diversity (age range, trait overlap, background variety) and returns
warnings when the generated set is too homogeneous.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Domain prompt templates
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, dict[str, str]] = {
    "career-tech": {
        "name": "Career Tech Workers",
        "description": "Software engineers, PMs, designers, and DevOps across seniority levels.",
        "template": (
            "Generate a diverse panel of technology professionals. Include a mix of:\n"
            "- Seniority levels: junior (0-2 yrs), mid (3-7 yrs), senior (8-15 yrs), staff/principal (15+ yrs)\n"
            "- Roles: frontend, backend, full-stack, DevOps/SRE, data, design, product management\n"
            "- Company sizes: startups, mid-size, enterprise, freelance/indie\n"
            "- Backgrounds: bootcamp grads, CS degrees, career changers, self-taught\n"
            "- Ages: spread across 22-55\n"
            "- Personality: mix analytical/creative, risk-averse/early-adopter, introverted/extroverted\n"
            "Each persona should have a distinct perspective shaped by their specific experience."
        ),
    },
    "hiring-managers": {
        "name": "Hiring Managers",
        "description": "People who evaluate, interview, and hire across industries.",
        "template": (
            "Generate a diverse panel of hiring managers and talent decision-makers. Include:\n"
            "- Industries: tech, healthcare, finance, retail, manufacturing, education\n"
            "- Team sizes they hire for: small (2-5), medium (10-20), large (50+)\n"
            "- Hiring volume: occasional (1-2/year), regular (monthly), high-volume (weekly)\n"
            "- Seniority: first-time managers, experienced directors, VP-level\n"
            "- Ages: spread across 28-60\n"
            "- Attitudes: process-driven vs. gut-feel, credential-focused vs. skills-first\n"
            "Each persona should reflect real hiring pain points and priorities."
        ),
    },
    "ai-tool-users": {
        "name": "AI Tool-Native Workers",
        "description": "Professionals who use AI tools daily in their workflow.",
        "template": (
            "Generate a diverse panel of professionals who actively use AI tools. Include:\n"
            "- AI adoption level: power users, daily users, cautious adopters, skeptics forced to use\n"
            "- Roles: content writers, analysts, developers, marketers, researchers, support agents\n"
            "- Tools used: ChatGPT, Claude, Copilot, Midjourney, domain-specific AI\n"
            "- Concerns: accuracy, privacy, job displacement, productivity, creativity\n"
            "- Ages: spread across 22-55\n"
            "- Tech comfort: digital natives, adapters, reluctant users\n"
            "Each persona should have specific opinions about AI's role in their work."
        ),
    },
    "small-business": {
        "name": "Small Business Owners",
        "description": "Founders and operators of businesses with 1-50 employees.",
        "template": (
            "Generate a diverse panel of small business owners and operators. Include:\n"
            "- Business types: retail, services, SaaS, food/hospitality, consulting, trades\n"
            "- Stage: pre-revenue, early (1-2 yrs), established (3-10 yrs), mature (10+ yrs)\n"
            "- Team size: solo, 2-5, 6-20, 20-50 employees\n"
            "- Revenue: bootstrapped, profitable, struggling, funded\n"
            "- Ages: spread across 25-65\n"
            "- Mindset: growth-focused, lifestyle, serial entrepreneur, accidental founder\n"
            "Each persona should have distinct operational constraints and priorities."
        ),
    },
    "healthcare-professionals": {
        "name": "Healthcare Professionals",
        "description": "Clinicians, administrators, and allied health workers.",
        "template": (
            "Generate a diverse panel of healthcare professionals. Include:\n"
            "- Roles: physicians, nurses, pharmacists, therapists, administrators, technicians\n"
            "- Settings: hospital, private practice, clinic, telehealth, long-term care\n"
            "- Specialties: primary care, emergency, pediatrics, mental health, surgery\n"
            "- Experience: early career (1-3 yrs), mid-career (5-15 yrs), veteran (20+ yrs)\n"
            "- Ages: spread across 25-65\n"
            "- Tech attitude: eager adopter, cautious, resistant, depends-on-evidence\n"
            "Each persona should reflect real clinical workflow constraints and patient care priorities."
        ),
    },
    "education-k12": {
        "name": "K-12 Educators",
        "description": "Teachers, administrators, and support staff across grade bands.",
        "template": (
            "Generate a diverse panel of K-12 educators. Include:\n"
            "- Roles: classroom teachers, principals, assistant principals, paraprofessionals,\n"
            "  district staff, instructional coaches, school counselors, special-ed teachers\n"
            "- Grade bands: early elementary (K-2), upper elementary (3-5), middle (6-8), high (9-12)\n"
            "- Subjects (secondary): ELA, math, science, social studies, electives, CTE\n"
            "- School types: public, charter, magnet, private, rural, urban, suburban\n"
            "- Experience: novice (0-3 yrs), mid-career (4-15 yrs), veteran (15+ yrs)\n"
            "- Ages: spread across 23-62\n"
            "- Attitudes: tech-forward, traditionalist, burned-out, idealistic, pragmatic\n"
            "Each persona should reflect real classroom constraints, district politics, and student-population context."
        ),
    },
    "enterprise-buyers": {
        "name": "Enterprise Software Buyers",
        "description": "Decision-makers who evaluate and purchase B2B software at >250-seat orgs.",
        "template": (
            "Generate a diverse panel of enterprise software buyers. Include:\n"
            "- Functions: IT/CIO org, security, finance/procurement, HR, sales ops,\n"
            "  marketing ops, engineering leadership, operations\n"
            "- Buyer roles: economic buyer, technical evaluator, end-user champion, procurement gatekeeper\n"
            "- Org sizes: mid-market (250-2k seats), enterprise (2k-25k), F500 (25k+)\n"
            "- Industries: financial services, healthcare, manufacturing, retail, public sector, tech\n"
            "- Deal sizes typically signed off: <$50k, $50-250k, $250k-1M, $1M+\n"
            "- Ages: spread across 32-60\n"
            "- Attitudes: risk-averse vs. innovator, vendor-loyal vs. best-of-breed, cost-led vs. outcome-led\n"
            "Each persona should reflect real procurement constraints (security review, legal, budget cycles) and the\n"
            "specific stakeholders they must align before signing."
        ),
    },
    "creators": {
        "name": "Content Creators",
        "description": "Independent creators monetizing audiences across platforms.",
        "template": (
            "Generate a diverse panel of content creators. Include:\n"
            "- Primary platforms: YouTube, TikTok, Instagram, Twitch, Substack, podcasting, X/Twitter\n"
            "- Audience size: nano (<10k), micro (10-100k), mid (100k-1M), macro (1M+)\n"
            "- Niches: lifestyle, gaming, education, tech, beauty, fitness, finance, comedy, news\n"
            "- Monetization mix: ad revenue, brand deals, merch, subscriptions/memberships, courses, affiliate\n"
            "- Career stage: side hustle, transitioning full-time, full-time established, post-burnout pivot\n"
            "- Ages: spread across 19-50\n"
            "- Attitudes: algorithm-chaser vs. community-builder, brand-friendly vs. anti-sponsorship,\n"
            "  optimistic vs. burnt-out\n"
            "Each persona should reflect real platform-specific economics and creator-burnout dynamics."
        ),
    },
    "graduate-students": {
        "name": "Graduate Students",
        "description": "Master's and PhD students across discipline clusters.",
        "template": (
            "Generate a diverse panel of graduate students. Include:\n"
            "- Degree programs: PhD, research master's, professional master's (MBA, MPH, MSW, MFA, JD)\n"
            "- Discipline clusters: STEM (CS, bio, physics, engineering), humanities (history, literature,\n"
            "  philosophy), social sciences (psych, econ, sociology), professional (medicine, law, business)\n"
            "- Year in program: first-year, mid-program, ABD/dissertating, defending\n"
            "- Funding situation: fully funded, TA-supported, partial scholarship, self-funded/loans\n"
            "- Institution type: R1 research, R2, regional, professional school, international\n"
            "- Ages: spread across 22-42\n"
            "- Career intent: academia/professoriate, industry research, government/policy, entrepreneurship,\n"
            "  unsure/exploring\n"
            "- Attitudes: idealistic, jaded, burned-out, careerist, vocational\n"
            "Each persona should reflect real advisor-relationship dynamics, funding precarity,\n"
            "and post-grad job-market anxieties."
        ),
    },
}


class DomainNotFoundError(ValueError):
    """Raised when a domain name is not in the registry."""

    def __init__(self, name: str) -> None:
        known = ", ".join(sorted(_TEMPLATES))
        super().__init__(f"Unknown domain: {name!r} (known: {known})")
        self.name = name


def get_domain_template(name: str) -> dict[str, str]:
    """Return a domain template by name.

    Raises :class:`DomainNotFoundError` if *name* is not registered.
    """
    if name not in _TEMPLATES:
        raise DomainNotFoundError(name)
    return _TEMPLATES[name]


def list_domain_templates() -> list[dict[str, str]]:
    """Return metadata for all registered domain templates."""
    return [{"name": name, "description": t["description"]} for name, t in sorted(_TEMPLATES.items())]


# ---------------------------------------------------------------------------
# Persona diversity validation
# ---------------------------------------------------------------------------

# Age brackets for diversity analysis.
_AGE_BRACKETS = [
    (0, 25, "under-25"),
    (25, 35, "25-34"),
    (35, 45, "35-44"),
    (45, 55, "45-54"),
    (55, 200, "55+"),
]


def _age_bracket(age: int) -> str:
    for lo, hi, label in _AGE_BRACKETS:
        if lo <= age < hi:
            return label
    return "unknown"


def validate_persona_diversity(
    personas: list[dict[str, Any]],
    *,
    min_age_brackets: int = 2,
    max_trait_overlap: float = 0.7,
    min_unique_backgrounds: int = 2,
) -> list[str]:
    """Check a persona panel for diversity and return warnings.

    Args:
        personas: List of persona dicts (must have at least ``name``).
        min_age_brackets: Minimum number of distinct age brackets expected.
        max_trait_overlap: Maximum fraction of shared traits between any
            two personas before warning (0.0 = no overlap, 1.0 = identical).
        min_unique_backgrounds: Minimum number of distinct background texts.

    Returns:
        List of warning strings.  Empty list means the panel looks diverse.
    """
    if len(personas) < 2:
        return []

    warnings: list[str] = []

    # --- Age range ---
    ages = [p["age"] for p in personas if isinstance(p.get("age"), (int, float))]
    if ages:
        brackets = {_age_bracket(int(a)) for a in ages}
        if len(brackets) < min(min_age_brackets, len(personas)):
            warnings.append(
                f"Low age diversity: all personas fall in {len(brackets)} age bracket(s) "
                f"({', '.join(sorted(brackets))}). Consider spreading across age ranges."
            )
    elif len(personas) >= 3:
        warnings.append("No age data on personas — cannot assess age diversity.")

    # --- Trait overlap ---
    trait_sets: list[set[str]] = []
    for p in personas:
        raw = p.get("personality_traits", [])
        if isinstance(raw, str):
            raw = [t.strip().lower() for t in raw.split(",") if t.strip()]
        elif isinstance(raw, list):
            raw = [str(t).strip().lower() for t in raw]
        else:
            raw = []
        trait_sets.append(set(raw))

    non_empty = [ts for ts in trait_sets if ts]
    if len(non_empty) >= 2:
        high_overlap_pairs = 0
        total_pairs = 0
        for i in range(len(non_empty)):
            for j in range(i + 1, len(non_empty)):
                total_pairs += 1
                union = non_empty[i] | non_empty[j]
                if union:
                    overlap = len(non_empty[i] & non_empty[j]) / len(union)
                    if overlap > max_trait_overlap:
                        high_overlap_pairs += 1
        if high_overlap_pairs > total_pairs * 0.5:
            warnings.append(
                f"High trait overlap: {high_overlap_pairs}/{total_pairs} persona pairs "
                f"share >{max_trait_overlap:.0%} of traits. Consider more varied personality profiles."
            )

    # --- Background variety ---
    backgrounds = [
        p["background"].strip().lower()
        for p in personas
        if isinstance(p.get("background"), str) and p["background"].strip()
    ]
    if backgrounds:
        unique = len(set(backgrounds))
        if unique < min(min_unique_backgrounds, len(personas)):
            warnings.append(
                f"Low background variety: only {unique} unique background(s) across "
                f"{len(personas)} personas. Consider more distinct professional contexts."
            )

    return warnings
