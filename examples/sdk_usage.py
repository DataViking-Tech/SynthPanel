"""End-to-end demo of the synthpanel Python SDK.

Run with::

    export ANTHROPIC_API_KEY=sk-...
    python examples/sdk_usage.py

The script walks through the eight public entry points in order, using
only the arguments the average caller needs. Each step prints a short
summary so you can confirm the API surface without reading source.

Every call here also works identically from inside a Jupyter notebook —
``from synth_panel import quick_poll`` and you're off.
"""

from __future__ import annotations

from synth_panel import (
    extend_panel,
    get_panel_result,
    list_instruments,
    list_panel_results,
    list_personas,
    quick_poll,
    run_panel,
    run_prompt,
)


def _divider(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    # 1) run_prompt — the simplest possible call.
    _divider("1. run_prompt — one-shot LLM call")
    prompt = run_prompt(
        "In one sentence, what makes a synthetic focus group useful?",
        model="haiku",
    )
    print(f"model: {prompt.model}")
    print(f"cost:  {prompt.cost}")
    print(f"reply: {prompt.response}")

    # 2) list_personas — see the bundled persona packs.
    _divider("2. list_personas — available persona packs")
    packs = list_personas()
    for pack in packs[:5]:
        kind = "bundled" if pack.get("builtin") else "user"
        print(f"  [{kind}] {pack['id']:<25} {pack['persona_count']} personas")

    # 3) list_instruments — see the bundled branching instruments.
    _divider("3. list_instruments — available instrument packs")
    instruments = list_instruments()
    for inst in instruments[:5]:
        print(f"  [{inst['source']}] {inst['id']:<30} {inst['description'][:60]}")

    # 4) quick_poll — a single question across a persona pack.
    _divider("4. quick_poll — single question, bundled personas")
    poll = quick_poll(
        "What's the most confusing thing about B2B SaaS pricing pages?",
        pack_id="general-consumer",
        model="haiku",
        synthesis=True,
    )
    print(f"result_id:   {poll.result_id}")
    print(f"panelists:   {len(poll.responses)}")
    print(f"total cost:  {poll.total_cost}")
    if poll.synthesis:
        rec = poll.synthesis.get("recommendation", "")
        print(f"recommendation (first 120 chars): {rec[:120]}")

    # 5) run_panel — full branching instrument against a bundled pack.
    _divider("5. run_panel — v3 branching instrument")
    panel = run_panel(
        pack_id="general-consumer",
        instrument_pack="pricing-discovery",
        model="haiku",
    )
    print(f"result_id:       {panel.result_id}")
    print(f"personas:        {panel.persona_count}")
    print(f"rounds executed: {[r['name'] for r in panel.rounds]}")
    print(f"routing path:    {[step['round'] for step in panel.path]}")
    print(f"total cost:      {panel.total_cost}")

    # 6) extend_panel — append a single ad-hoc follow-up round.
    _divider("6. extend_panel — follow-up round")
    extended = extend_panel(
        result_id=panel.result_id,
        questions="What single change would make you trust this brand more?",
        model="haiku",
    )
    print(f"rounds now: {[r['name'] for r in extended.rounds]}")
    print(f"path now:   {[step['round'] for step in extended.path]}")

    # 7) list_panel_results — see what's on disk.
    _divider("7. list_panel_results — saved results")
    saved = list_panel_results()
    for entry in saved[:3]:
        print(f"  {entry['id']:<45} model={entry.get('model')} n={entry.get('persona_count')}")

    # 8) get_panel_result — reload a specific result by id.
    _divider("8. get_panel_result — reload a saved result")
    reloaded = get_panel_result(panel.result_id)
    print(f"reloaded {reloaded.result_id}")
    print(f"question_count: {reloaded.question_count}")
    print(f"terminal round: {reloaded.terminal_round}")


if __name__ == "__main__":
    main()
