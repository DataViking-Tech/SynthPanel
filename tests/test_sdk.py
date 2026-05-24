"""Tests for the public SDK surface in :mod:`synth_panel.sdk`.

Covers:
* Import surface — the eight exported functions and three dataclasses
  are reachable from the package root.
* ``__all__`` correctness — no stray internals leak, no dead names.
* ``run_prompt`` round-trip against a mocked :class:`LLMClient`.
* ``quick_poll`` / ``run_panel`` wiring — verify they reach the
  shared runners with the right inputs.
* ``list_*`` / ``get_panel_result`` — delegate to :mod:`synth_panel.mcp.data`.
* Validation errors for empty personas/questions + persona schema.
* ``PanelResult`` dict-like compatibility (``__getitem__``, ``.to_dict``).
* Zero reliance on the optional ``mcp`` extra — the SDK must import
  cleanly without the ``mcp`` package.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point the persistence layer at a temp dir for every test."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    # Reset the module-level shared client so each test starts fresh.
    import synth_panel.sdk as _sdk

    _sdk._shared_client = None


# ---------------------------------------------------------------------------
# Surface: imports and __all__ contract
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_eight_public_functions_reachable_from_root(self):
        import synth_panel

        for name in (
            "run_prompt",
            "quick_poll",
            "run_panel",
            "extend_panel",
            "list_personas",
            "list_instruments",
            "list_panel_results",
            "get_panel_result",
        ):
            assert hasattr(synth_panel, name), f"synth_panel.{name} is missing"
            assert callable(getattr(synth_panel, name))

    def test_three_result_dataclasses_reachable_from_root(self):
        from synth_panel import PanelResult, PollResult, PromptResult

        assert PanelResult.__name__ == "PanelResult"
        assert PollResult.__name__ == "PollResult"
        assert PromptResult.__name__ == "PromptResult"

    def test_all_is_explicit_and_sorted(self):
        import synth_panel

        assert "__all__" in dir(synth_panel)
        # No duplicates and each name is actually exported.
        assert len(synth_panel.__all__) == len(set(synth_panel.__all__))
        for name in synth_panel.__all__:
            assert hasattr(synth_panel, name), f"__all__ names missing attribute: {name}"

    def test_sdk_does_not_require_mcp_extra(self):
        """Importing the SDK module must not trigger the mcp library.

        synth_panel.sdk reaches into synth_panel.mcp.data (pure
        yaml/json) but NOT into synth_panel.mcp.server (which imports
        fastmcp). This test runs in an env where `mcp` is installed, so
        we can only assert that fresh importing does not raise — the
        stronger guarantee (SDK works without `[mcp]` extras) is
        verified by the no-extras leg of CI.
        """
        import importlib

        # Force a re-import to prove the module is self-sufficient.
        import synth_panel.sdk

        reloaded = importlib.reload(synth_panel.sdk)
        assert hasattr(reloaded, "run_prompt")
        assert hasattr(reloaded, "quick_poll")


# ---------------------------------------------------------------------------
# run_prompt
# ---------------------------------------------------------------------------


class TestRunPrompt:
    def test_returns_prompt_result_with_cost_and_usage(self):
        from synth_panel import run_prompt
        from synth_panel.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="resp-1",
            model="claude-haiku-4-5-20251001",
            content=[TextBlock(text="Hello back")],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        with patch("synth_panel.sdk.LLMClient") as MockClient:
            MockClient.return_value.send.return_value = mock_response
            result = run_prompt("Say hello", model="haiku")

        assert result.response == "Hello back"
        assert result.model == "claude-haiku-4-5-20251001"
        assert result.usage["input_tokens"] == 10
        assert result.cost.startswith("$")

    def test_empty_prompt_raises(self):
        from synth_panel import run_prompt

        with pytest.raises(ValueError, match="non-empty"):
            run_prompt("")

    def test_default_model_chosen_from_environment(self, monkeypatch):
        """Default model follows the provider-preference chain in env."""
        from synth_panel.sdk import _default_model

        for key in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        assert _default_model() == "sonnet"

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _default_model() == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# quick_poll / run_panel
# ---------------------------------------------------------------------------


class TestQuickPoll:
    def test_empty_question_raises(self):
        from synth_panel import quick_poll

        with pytest.raises(ValueError, match="non-empty"):
            quick_poll("", personas=[{"name": "A"}])

    def test_requires_personas_or_pack_id(self):
        from synth_panel import quick_poll

        with pytest.raises(ValueError, match="No personas"):
            quick_poll("q?")

    def test_calls_run_panel_sync_with_single_question(self):
        """quick_poll wraps the question and drives the shared runner."""
        from synth_panel import quick_poll
        from synth_panel.cost import TokenUsage

        fake_usage = TokenUsage(input_tokens=1, output_tokens=1)
        fake_cost = MagicMock()
        fake_cost.format_usd.return_value = "$0.01"
        # Real float so build_metadata's round(cost.total_cost, 6) stays
        # JSON-serializable when the metadata block is now persisted (#525).
        fake_cost.total_cost = 0.01
        fake_cost.__add__ = lambda self, other: self

        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = (
                [],  # panelist_results
                [{"persona": "A", "responses": [{"response": "ok"}], "usage": {}, "cost": "$0.01", "error": None}],
                fake_usage,
                fake_cost,
                None,
                None,
            )
            result = quick_poll("Does this work?", personas=[{"name": "A"}])

        assert result.question == "Does this work?"
        assert len(result.responses) == 1
        # The runner was passed exactly one question, wrapping the string.
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["questions"] == [{"text": "Does this work?"}]

    def test_persists_questions_metadata(self):
        """sy-oyl: the saved result must carry the question (and any
        response_schema) so poll_summary's _detect_kind can honor declared
        types after reload. Previously this kwarg was dropped on the
        single-question save path, so reloaded results fell back to
        inference and misclassified text answers containing version strings
        as scales.
        """
        from synth_panel import quick_poll
        from synth_panel.cost import TokenUsage

        fake_usage = TokenUsage(input_tokens=1, output_tokens=1)
        fake_cost = MagicMock()
        fake_cost.format_usd.return_value = "$0.01"
        # Real float so build_metadata's round(cost.total_cost, 6) stays
        # JSON-serializable when the metadata block is now persisted (#525).
        fake_cost.total_cost = 0.01
        fake_cost.__add__ = lambda self, other: self

        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
            patch("synth_panel.sdk.save_panel_result") as mock_save,
        ):
            mock_runner.return_value = (
                [],
                [{"persona": "A", "responses": [{"response": "ok"}], "usage": {}, "cost": "$0.01", "error": None}],
                fake_usage,
                fake_cost,
                None,
                None,
            )
            mock_save.return_value = "result-stub"
            quick_poll("Does v1.5.1 feel agent-ready?", personas=[{"name": "A"}])

        kwargs = mock_save.call_args.kwargs
        assert kwargs["questions"] == [{"text": "Does v1.5.1 feel agent-ready?"}]


class TestRunPanel:
    def test_requires_question_source(self):
        from synth_panel import run_panel

        with pytest.raises(ValueError, match="questions|instrument|instrument_pack"):
            run_panel(personas=[{"name": "A"}])

    def test_variants_out_of_range_raises(self):
        from synth_panel import run_panel

        with pytest.raises(ValueError, match="variants must be"):
            run_panel(personas=[{"name": "A"}], questions=["q?"], variants=99)

    def test_persona_without_name_raises(self):
        from synth_panel import run_panel

        with pytest.raises(ValueError, match="name"):
            run_panel(personas=[{"age": 30}], questions=["q?"])

    def test_question_strings_are_auto_wrapped(self):
        """Pass a list of strings and they become question dicts."""
        from synth_panel import run_panel
        from synth_panel.cost import TokenUsage

        fake_usage = TokenUsage(input_tokens=1, output_tokens=1)
        fake_cost = MagicMock()
        fake_cost.format_usd.return_value = "$0.01"
        # Real float so build_metadata's round(cost.total_cost, 6) stays
        # JSON-serializable when the metadata block is now persisted (#525).
        fake_cost.total_cost = 0.01

        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = ([], [], fake_usage, fake_cost, None, None)
            run_panel(
                personas=[{"name": "Alice"}],
                questions=["First?", "Second?"],
            )
            kwargs = mock_runner.call_args.kwargs
            assert kwargs["questions"] == [{"text": "First?"}, {"text": "Second?"}]

    def test_flat_questions_path_persists_questions_with_schema(self):
        """sy-oyl: the flat ``questions=[{...}]`` save path now threads the
        normalised question list (including ``response_schema``) into
        ``save_panel_result`` so reloads preserve declared types.
        """
        from synth_panel import run_panel
        from synth_panel.cost import TokenUsage

        fake_usage = TokenUsage(input_tokens=1, output_tokens=1)
        fake_cost = MagicMock()
        fake_cost.format_usd.return_value = "$0.01"
        # Real float so build_metadata's round(cost.total_cost, 6) stays
        # JSON-serializable when the metadata block is now persisted (#525).
        fake_cost.total_cost = 0.01
        fake_cost.__add__ = lambda self, other: self

        questions_in = [
            {"text": "Would v1.5.1 feel agent-ready?", "response_schema": {"type": "text"}},
            {"text": "Confidence (1-5)?", "response_schema": {"type": "scale", "min": 1, "max": 5}},
        ]
        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
            patch("synth_panel.sdk.save_panel_result") as mock_save,
        ):
            mock_runner.return_value = ([], [], fake_usage, fake_cost, None, None)
            mock_save.return_value = "result-stub"
            run_panel(personas=[{"name": "Alice"}], questions=questions_in)

        kwargs = mock_save.call_args.kwargs
        saved_questions = kwargs["questions"]
        assert len(saved_questions) == 2
        # response_schema must round-trip — that's the whole point of the fix.
        assert saved_questions[0]["response_schema"] == {"type": "text"}
        assert saved_questions[1]["response_schema"] == {"type": "scale", "min": 1, "max": 5}

    def test_instrument_pack_takes_precedence_over_questions(self, monkeypatch):
        """If instrument_pack is given, questions/instrument are ignored."""
        from synth_panel import run_panel

        # Mock the pack loader to return a tiny v1 instrument.
        def fake_load_pack(name):
            assert name == "dummy-pack"
            return {
                "name": "dummy-pack",
                "instrument": {
                    "version": 1,
                    "questions": [{"text": "From pack"}],
                },
            }

        monkeypatch.setattr("synth_panel.sdk._data_load_instrument_pack", fake_load_pack)

        from synth_panel.cost import TokenUsage
        from synth_panel.orchestrator import MultiRoundResult, RoundResult

        # Stub the multi-round runner so we don't hit the network.
        fake_mr = MultiRoundResult(
            rounds=[
                RoundResult(
                    name="round_1",
                    panelist_results=[],
                    synthesis=None,
                    usage=TokenUsage(input_tokens=0, output_tokens=0),
                )
            ],
            path=[],
            terminal_round="round_1",
            final_synthesis=None,
            warnings=[],
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )
        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_multi_round_sync", return_value=fake_mr) as mock_mr,
        ):
            out = run_panel(
                personas=[{"name": "A"}],
                instrument_pack="dummy-pack",
                questions=["IGNORED"],
            )
        # The multi-round runner was called (instrument path won), not the
        # single-round runner.
        assert mock_mr.called
        assert out.result_id


class TestRunPanelExtractSchema:
    """v1.0.4 P4 (hq-r39v): caller-facing ``response_schema=MyPydanticClass``.

    The internal dispatch in :func:`synth_panel._runners.resolve_extract_schema`
    landed in v1.0.3; this class pins the SDK boundary — that the type
    annotation has widened, that a Pydantic class flows through correctly,
    and that the back-compat shapes (string name, dict, None) still work
    unchanged.
    """

    @staticmethod
    def _stub_run_panel_sync():
        from synth_panel.cost import TokenUsage

        fake_usage = TokenUsage(input_tokens=1, output_tokens=1)
        fake_cost = MagicMock()
        fake_cost.format_usd.return_value = "$0.01"
        # Real float so build_metadata's round(cost.total_cost, 6) stays
        # JSON-serializable when the metadata block is now persisted (#525).
        fake_cost.total_cost = 0.01
        fake_cost.__add__ = lambda self, other: self
        return fake_usage, fake_cost

    def test_basemodel_class_accepted_at_sdk_boundary(self):
        """AC: ``extract_schema=MyPydanticClass`` reaches the runner as a
        resolved envelope carrying the typed model, not as the raw class."""
        from pydantic import BaseModel, Field

        from synth_panel import run_panel

        class FeatureChoice(BaseModel):
            feature: str = Field(..., min_length=1)
            confidence: int = Field(..., ge=1, le=5)

        fake_usage, fake_cost = self._stub_run_panel_sync()
        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = ([], [], fake_usage, fake_cost, None, None)
            run_panel(
                personas=[{"name": "Alice"}],
                questions=["Which feature first?"],
                extract_schema=FeatureChoice,
            )
        kwargs = mock_runner.call_args.kwargs
        envelope = kwargs["extract_schema"]
        assert isinstance(envelope, dict)
        assert envelope["model"] is FeatureChoice
        assert envelope["schema"]["type"] == "object"
        assert "feature" in envelope["schema"]["properties"]
        assert "confidence" in envelope["schema"]["properties"]

    def test_signature_advertises_basemodel(self):
        """Type hint at the SDK boundary must include ``type[BaseModel]`` —
        callers (and IDE/type-checkers) should see the typed-class branch
        without having to read the docstring."""
        import inspect

        from pydantic import BaseModel

        from synth_panel import run_panel

        sig = inspect.signature(run_panel)
        param = sig.parameters["extract_schema"]
        # The annotation is a string-evaluated PEP 604 union; just check
        # that BaseModel appears in the resolved annotation set.
        try:
            from typing import get_args, get_type_hints

            hints = get_type_hints(run_panel)
            args = get_args(hints["extract_schema"])
            assert any(
                arg is BaseModel or (isinstance(arg, type) and issubclass(arg, BaseModel)) or arg == type[BaseModel]
                for arg in args
            ), f"BaseModel not in extract_schema annotation args: {args}"
        except (TypeError, NameError):
            # Fallback: stringified annotation must mention BaseModel.
            assert "BaseModel" in str(param.annotation)

    def test_string_name_still_works(self):
        """Back-compat: a registered name is resolved into the correct
        envelope (schema from the bundled registry, model from MODEL_REGISTRY)."""
        from synth_panel import run_panel
        from synth_panel.structured.models import AnnotatedChoice

        fake_usage, fake_cost = self._stub_run_panel_sync()
        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = ([], [], fake_usage, fake_cost, None, None)
            run_panel(
                personas=[{"name": "Alice"}],
                questions=["pick"],
                extract_schema="annotated_choice",
            )
        envelope = mock_runner.call_args.kwargs["extract_schema"]
        assert envelope["model"] is AnnotatedChoice
        assert envelope["schema"]["type"] == "object"

    def test_dict_still_works(self):
        """Back-compat: an inline JSON Schema dict resolves to envelope
        with ``model=None`` and the dict carried verbatim as ``schema``."""
        from synth_panel import run_panel

        raw = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        fake_usage, fake_cost = self._stub_run_panel_sync()
        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = ([], [], fake_usage, fake_cost, None, None)
            run_panel(
                personas=[{"name": "Alice"}],
                questions=["q"],
                extract_schema=raw,
            )
        envelope = mock_runner.call_args.kwargs["extract_schema"]
        assert envelope == {"schema": raw, "model": None}

    def test_none_still_passes_through(self):
        """Back-compat: omitting ``extract_schema`` (or passing None) must
        keep ``extract_schema=None`` on the runner — no envelope wrapping."""
        from synth_panel import run_panel

        fake_usage, fake_cost = self._stub_run_panel_sync()
        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = ([], [], fake_usage, fake_cost, None, None)
            run_panel(personas=[{"name": "A"}], questions=["q"])
        assert mock_runner.call_args.kwargs["extract_schema"] is None

    def test_invalid_type_raises_at_sdk_boundary(self):
        """Anything that isn't a BaseModel subclass / dict / str / None
        is rejected by ``resolve_extract_schema`` before the runner is
        ever called — no LLM spend on a malformed argument."""
        from synth_panel import run_panel

        with pytest.raises(TypeError, match="extract_schema"):
            run_panel(
                personas=[{"name": "A"}],
                questions=["q"],
                extract_schema=42,  # type: ignore[arg-type]
            )

    def test_basemodel_validation_error_surfaces_via_unpack(self):
        """ValidationError surfacing path: when the orchestrator unpacks
        the resolved envelope and runs ``model_validate`` on a wire-valid
        but typed-invalid payload, the field-path error is reachable —
        this is the mechanism the orchestrator uses to populate
        ``extraction_validation_error`` on a per-response basis (see
        ``orchestrator.py`` around the ``_PydanticValidationError`` catch).
        Reproduce the unpack + validate flow without the LLM in scope."""
        from pydantic import BaseModel, Field, ValidationError

        from synth_panel._runners import resolve_extract_schema
        from synth_panel.orchestrator import _unpack_extract_schema

        class TypedLikert(BaseModel):
            rating: int = Field(..., ge=1, le=5)

        envelope = resolve_extract_schema(TypedLikert)
        json_schema, pyd_model = _unpack_extract_schema(envelope)
        assert json_schema == envelope["schema"]
        assert pyd_model is TypedLikert
        # Wire-valid (matches generated JSON Schema) but breaks the typed
        # 1..5 constraint — this is the case the surfacing path is for.
        with pytest.raises(ValidationError) as exc_info:
            pyd_model.model_validate({"rating": 7})
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("rating",) for err in errors)

    def test_mixed_registry_pydantic_and_named_coexist(self):
        """A user-defined Pydantic class (not in MODEL_REGISTRY) and a
        registered name must both work in the same process — neither
        path leaks state into the other. This is the "mixed registry
        use" coverage from the AC."""
        from pydantic import BaseModel

        from synth_panel import run_panel
        from synth_panel.structured.models import AnnotatedChoice

        class CustomPick(BaseModel):
            label: str

        fake_usage, fake_cost = self._stub_run_panel_sync()
        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = ([], [], fake_usage, fake_cost, None, None)
            run_panel(
                personas=[{"name": "A"}],
                questions=["q"],
                extract_schema=CustomPick,
            )
            run_panel(
                personas=[{"name": "A"}],
                questions=["q"],
                extract_schema="annotated_choice",
            )
        first_env = mock_runner.call_args_list[0].kwargs["extract_schema"]
        second_env = mock_runner.call_args_list[1].kwargs["extract_schema"]
        assert first_env["model"] is CustomPick
        assert second_env["model"] is AnnotatedChoice
        assert first_env["schema"] != second_env["schema"]


# ---------------------------------------------------------------------------
# list_* and get_panel_result
# ---------------------------------------------------------------------------


class TestListDelegates:
    def test_list_personas_returns_bundled_packs(self):
        from synth_panel import list_personas

        packs = list_personas()
        # At least one bundled pack ships with the package.
        assert len(packs) >= 1
        assert all("id" in p for p in packs)

    def test_list_instruments_returns_bundled_packs(self):
        from synth_panel import list_instruments

        packs = list_instruments()
        assert len(packs) >= 1
        assert all("id" in p for p in packs)

    def test_list_panel_results_empty_in_clean_dir(self):
        from synth_panel import list_panel_results

        assert list_panel_results() == []


class TestGetPanelResult:
    def test_returns_panel_result_dataclass(self):
        from synth_panel import get_panel_result
        from synth_panel.mcp.data import save_panel_result

        rid = save_panel_result(
            results=[{"persona": "A", "responses": [{"response": "hi"}]}],
            model="haiku",
            total_usage={"input_tokens": 1, "output_tokens": 1},
            total_cost="$0.01",
            persona_count=1,
            question_count=1,
        )

        out = get_panel_result(rid)
        assert out.result_id == rid
        assert out.model == "haiku"
        assert out.persona_count == 1
        # Dict-like access for back-compat with callers that used to
        # read the raw MCP payload.
        assert out["model"] == "haiku"
        assert "model" in out
        assert out.to_dict()["model"] == "haiku"

    def test_missing_result_raises_filenotfound(self):
        from synth_panel import get_panel_result

        with pytest.raises(FileNotFoundError):
            get_panel_result("nope-does-not-exist")


# ---------------------------------------------------------------------------
# sp-nn8k: PanelResult surfaces cost_is_estimated + warnings
# ---------------------------------------------------------------------------


class TestPanelResultCostFallback:
    """The single-round SDK builder must flag DEFAULT_PRICING fallbacks."""

    def _call(self, *, model, contributing_models=None):
        from synth_panel.cost import CostEstimate, TokenUsage
        from synth_panel.sdk import _build_panel_result_from_single_round

        cost = CostEstimate()
        return _build_panel_result_from_single_round(
            result_id="rid-test",
            model=model,
            personas=[{"name": "A"}],
            questions=[{"text": "q"}],
            result_dicts=[],
            panelist_usage=TokenUsage(),
            panelist_cost=cost,
            synthesis_dict=None,
            metadata=None,
            total_usage=TokenUsage(),
            total_cost=cost,
            contributing_models=contributing_models,
        )

    def test_priced_model_has_no_warning_and_flag_false(self):
        panel = self._call(model="claude-sonnet-4-6")
        assert panel.warnings == []
        assert panel.cost_is_estimated is False

    def test_unpriced_model_emits_warning_and_sets_flag(self):
        panel = self._call(model="totally-unknown-model-x9")
        assert panel.cost_is_estimated is True
        assert len(panel.warnings) == 1
        assert "totally-unknown-model-x9" in panel.warnings[0]
        assert "DEFAULT_PRICING fallback" in panel.warnings[0]

    def test_contributing_models_drives_detection(self):
        """When contributing_models is passed it supersedes the single model check,
        so mixed-model runs surface each offender."""
        panel = self._call(
            model="sonnet",
            contributing_models=["sonnet", "exotic-7b", "haiku", "mystery-13b"],
        )
        assert panel.cost_is_estimated is True
        joined = "\n".join(panel.warnings)
        assert "exotic-7b" in joined
        assert "mystery-13b" in joined
        assert "sonnet" not in joined
        assert "haiku" not in joined
        assert len(panel.warnings) == 2


# ---------------------------------------------------------------------------
# Extension path
# ---------------------------------------------------------------------------


class TestExtendPanel:
    def test_missing_sessions_raises(self):
        from synth_panel import extend_panel
        from synth_panel.mcp.data import save_panel_result

        # Save a result but don't save any sessions for it.
        rid = save_panel_result(
            results=[{"persona": "A", "responses": []}],
            model="haiku",
            total_usage={},
            total_cost="$0",
            persona_count=1,
            question_count=1,
        )
        with pytest.raises((ValueError, FileNotFoundError)):
            extend_panel(rid, "Follow-up?")
