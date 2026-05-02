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
    "healthcare-providers": {
        "name": "Independent Healthcare Providers",
        "description": "Private-practice physicians, dentists, and specialist providers running their own businesses.",
        "template": (
            "Generate a diverse panel of independent healthcare providers who own or operate their practices. Include:\n"
            "- Practice types: primary care, dental, optometry, dermatology, psychiatry, chiropractic, physical therapy\n"
            "- Practice size: solo, small group (2-5), medium group (6-20)\n"
            "- Years in practice: newly independent (1-3 yrs), established (5-15 yrs), veteran (20+ yrs)\n"
            "- Ownership stake: sole owner, partner, recently acquired by PE/health system\n"
            "- Staff size: front desk only, full admin team, outsourced billing\n"
            "- Ages: spread across 30-65\n"
            "- Pain points: EHR burden, insurance reimbursement, patient acquisition, staffing, regulatory compliance\n"
            "Each persona should reflect the dual pressures of running a healthcare business and delivering quality care."
        ),
    },
    "education-K12": {
        "name": "K-12 Educators and Administrators",
        "description": "Teachers, instructional coaches, and school administrators across grade levels.",
        "template": (
            "Generate a diverse panel of K-12 educators and school administrators. Include:\n"
            "- Roles: classroom teachers, instructional coaches, department heads, assistant principals, principals, district administrators\n"
            "- Grade bands: early elementary (K-2), upper elementary (3-5), middle school (6-8), high school (9-12)\n"
            "- School types: public, charter, private, rural, urban, suburban\n"
            "- Subject areas: STEM, humanities, special education, arts, PE, ESL/bilingual\n"
            "- Experience: first-year teacher, 3-10 yrs classroom, 10+ yrs, career-changers into teaching\n"
            "- Ages: spread across 22-60\n"
            "- Challenges: student engagement, standardized testing pressure, tech integration, parent communication, burnout\n"
            "Each persona should reflect the realities of public education funding, classroom diversity, and school culture."
        ),
    },
    "smb-owners": {
        "name": "SMB Owner-Operators",
        "description": "Hands-on owners of small and medium businesses who manage day-to-day operations themselves.",
        "template": (
            "Generate a diverse panel of small and medium business owner-operators. Include:\n"
            "- Industries: retail, food service, professional services, home services, e-commerce, health/wellness\n"
            "- Business age: startup (<2 yrs), growing (2-7 yrs), mature (7+ yrs)\n"
            "- Employee count: solo/freelance, 2-10, 11-50\n"
            "- Revenue range: <$250k, $250k-$1M, $1M-$5M\n"
            "- Tech savvy: spreadsheet-only, uses SaaS tools, actively automates\n"
            "- Ages: spread across 25-60\n"
            "- Motivations: financial independence, passion-driven, family legacy, exit-oriented\n"
            "Each persona should reflect the cash-flow constraints, operational demands, and growth ambitions of owner-led businesses."
        ),
    },
    "enterprise-buyers": {
        "name": "Enterprise Software Buyers",
        "description": "IT leaders, procurement officers, and business stakeholders who evaluate and purchase enterprise software.",
        "template": (
            "Generate a diverse panel of enterprise software buyers and evaluators. Include:\n"
            "- Roles: CIO, CTO, VP of IT, IT Director, procurement manager, business unit sponsor, finance stakeholder\n"
            "- Company size: mid-market (500-2000 employees), large enterprise (2000-10000), very large (10000+)\n"
            "- Industries: financial services, healthcare, manufacturing, retail, government, professional services\n"
            "- Budget authority: final decision-maker, influencer/recommender, technical evaluator, budget approver\n"
            "- Buying stage: exploring, in evaluation, contract renewal, post-implementation review\n"
            "- Ages: spread across 30-60\n"
            "- Concerns: security/compliance, integration complexity, vendor lock-in, TCO, change management, SLAs\n"
            "Each persona should reflect real enterprise procurement pain points, committee dynamics, and risk tolerance."
        ),
    },
    "creators": {
        "name": "Content Creators and Influencers",
        "description": "Independent creators monetizing content across social platforms and direct audiences.",
        "template": (
            "Generate a diverse panel of digital content creators. Include:\n"
            "- Platform focus: YouTube, TikTok, Instagram, Twitch, Substack/newsletters, podcasting, LinkedIn\n"
            "- Niche: tech/gaming, lifestyle/beauty, finance/investing, fitness, education, comedy, B2B thought leadership\n"
            "- Audience size: nano (<10k), micro (10k-100k), mid-tier (100k-1M), macro (1M+)\n"
            "- Revenue model: ad/sponsorship, subscription, merchandise, courses/coaching, agency work\n"
            "- Years creating: beginner (0-1), growing (1-3), established (3-7), veteran (7+)\n"
            "- Ages: spread across 18-45\n"
            "- Challenges: algorithm changes, burnout, monetization consistency, brand-deal negotiation, copyright/IP\n"
            "Each persona should reflect the hustle economics, platform dependency, and audience relationship dynamics of creator life."
        ),
    },
    "graduate-students": {
        "name": "Graduate Students and Early-Career Researchers",
        "description": "PhD students, master's students, and postdocs navigating academia and research careers.",
        "template": (
            "Generate a diverse panel of graduate students and early-career researchers. Include:\n"
            "- Degree level: master's (coursework/thesis), PhD (early/mid/late stage), postdoc\n"
            "- Fields: STEM (CS, biology, physics, engineering), social sciences, humanities, professional programs (MBA, law, medicine)\n"
            "- Institution type: R1 research university, teaching-focused university, international institution\n"
            "- Funding: fully funded RA/TA, fellowship, self-funded, part-time while working\n"
            "- Career trajectory: academia-bound, industry pivot, government/NGO, entrepreneurship\n"
            "- Ages: spread across 22-38\n"
            "- Challenges: advisor relationships, imposter syndrome, funding uncertainty, publish-or-perish pressure, work-life balance, job market anxiety\n"
            "Each persona should reflect the intellectual ambition, financial precarity, and career uncertainty of graduate training."
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
