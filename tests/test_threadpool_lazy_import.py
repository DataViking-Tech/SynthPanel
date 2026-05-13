"""sy-2wa: load-chain hygiene for pyodide consumers (CF Python Workers).

Under pyodide, ``concurrent.futures.ThreadPoolExecutor`` exists as a stub
but ``.submit()`` silently hangs. Binding it in the load chain of
``synth_panel.ensemble`` means any downstream code path that touches the
synthpanel namespace can reach a submit() that will deadlock the Worker.

These tests pin the contract:

1. ``from synth_panel.ensemble import synthesize_panel`` must not import
   ``concurrent.futures`` at any point in the load chain.
2. Under a pyodide-emulating sys.modules patch where importing
   ``concurrent.futures`` raises, the same import still succeeds — the
   ensemble surface is fully threadpool-free at load time.
3. Module namespaces of ``orchestrator``, ``synthesis``, and
   ``perturbation`` must not bind ``ThreadPoolExecutor`` until the
   threaded entry points are actually called.

We use subprocesses for the load-chain assertions because pytest collects
every test in one interpreter — by the time these tests run,
``synth_panel.*`` (and transitively ``concurrent.futures``) are already
in ``sys.modules``. A subprocess gives us a fresh interpreter so the
"never imported" claim is observable rather than inferred.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a fresh interpreter and return the result.

    Uses the same Python executable as the test runner so the editable
    ``synth_panel`` install is on the path. Stderr is captured so failures
    surface in pytest output.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_ensemble_load_does_not_import_concurrent_futures() -> None:
    """Acceptance criterion 1 (sy-2wa): ``from synth_panel.ensemble import
    synthesize_panel`` must not pull ``concurrent.futures`` into sys.modules.
    """
    script = textwrap.dedent(
        """
        import sys
        from synth_panel.ensemble import synthesize_panel  # noqa: F401

        if "concurrent.futures" in sys.modules:
            print("LEAK: concurrent.futures imported during ensemble load")
            sys.exit(1)
        if "concurrent" in sys.modules:
            print("LEAK: concurrent (parent) imported during ensemble load")
            sys.exit(1)
        print("OK")
        """
    )
    result = _run_in_subprocess(script)
    assert result.returncode == 0, (
        f"synth_panel.ensemble load chain pulled in concurrent.futures.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_ensemble_loads_with_poisoned_concurrent_futures() -> None:
    """Acceptance criterion 2 (sy-2wa): emulate pyodide by poisoning
    ``concurrent.futures`` so any access raises. ``synth_panel.ensemble``
    must still load — proving the load chain truly never touches it.
    """
    script = textwrap.dedent(
        """
        import sys
        import types

        # Poison: any attribute lookup raises. ``from concurrent.futures
        # import ThreadPoolExecutor`` runs __getattr__ on this module after
        # the import system resolves the submodule from sys.modules.
        poison = types.ModuleType("concurrent.futures")

        def _poisoned_attr(name):
            raise AssertionError(
                f"sy-2wa regression: concurrent.futures.{name} accessed "
                "during synth_panel.ensemble load"
            )

        poison.__getattr__ = _poisoned_attr
        parent = types.ModuleType("concurrent")
        parent.futures = poison
        sys.modules["concurrent"] = parent
        sys.modules["concurrent.futures"] = poison

        from synth_panel.ensemble import synthesize_panel  # noqa: F401
        print("OK")
        """
    )
    result = _run_in_subprocess(script)
    assert result.returncode == 0, (
        f"synth_panel.ensemble failed to load under poisoned concurrent.futures.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_threadpoolexecutor_not_bound_in_module_namespaces() -> None:
    """Acceptance criterion 3 (sy-2wa): the modules that *do* spawn
    threadpools (orchestrator, synthesis, perturbation) must not bind
    ``ThreadPoolExecutor`` / ``as_completed`` at module top, even after
    they've been loaded. Catches a regression where someone re-adds the
    top-level import "for convenience".
    """
    script = textwrap.dedent(
        """
        import sys
        from synth_panel.ensemble import synthesize_panel  # noqa: F401
        import synth_panel.orchestrator
        import synth_panel.synthesis
        import synth_panel.perturbation

        leaks = []
        for modname in (
            "synth_panel.orchestrator",
            "synth_panel.synthesis",
            "synth_panel.perturbation",
        ):
            mod = sys.modules[modname]
            for sym in ("ThreadPoolExecutor", "as_completed"):
                if hasattr(mod, sym):
                    leaks.append(f"{modname}.{sym}")

        if leaks:
            print("LEAKS:", leaks)
            sys.exit(1)
        print("OK")
        """
    )
    result = _run_in_subprocess(script)
    assert result.returncode == 0, (
        f"Threadpool symbols bound at module top — move imports inside "
        f"threaded functions (sy-2wa).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout
