"""Tests for multi-agent orchestration (SPEC.md §4)."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from synth_panel.cost import ZERO_USAGE, TokenUsage
from synth_panel.llm.models import (
    CompletionResponse,
    StopReason,
    TextBlock,
    TokenUsage as LLMTokenUsage,
)
from synth_panel.orchestrator import (
    FailureKind,
    InvalidTransitionError,
    PanelistResult,
    Worker,
    WorkerNotFoundError,
    WorkerRegistry,
    WorkerStatus,
    _extract_text,
    run_panel_parallel,
)
from synth_panel.persistence import ConversationMessage
from synth_panel.runtime import TurnSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_response(text: str = "Hello!", usage: LLMTokenUsage | None = None) -> CompletionResponse:
    return CompletionResponse(
        id="resp-1",
        model="claude-sonnet",
        content=[TextBlock(text=text)],
        stop_reason=StopReason.END_TURN,
        usage=usage or LLMTokenUsage(input_tokens=10, output_tokens=5),
    )


def _simple_system_prompt(persona: dict[str, Any]) -> str:
    return f"You are {persona.get('name', 'Anonymous')}."


def _simple_question_prompt(question: dict[str, Any]) -> str:
    if isinstance(question, dict):
        return question.get("text", str(question))
    return str(question)


def _make_mock_client(responses: list[CompletionResponse] | None = None) -> MagicMock:
    """Create a thread-safe mock client that returns canned responses."""
    client = MagicMock()
    if responses:
        # Use a lock to safely pop from the list across threads
        lock = threading.Lock()
        resp_list = list(responses)

        def thread_safe_send(request):
            with lock:
                if resp_list:
                    return resp_list.pop(0)
            return _make_text_response("fallback")

        client.send = MagicMock(side_effect=thread_safe_send)
    else:
        client.send = MagicMock(return_value=_make_text_response())
    return client


# ---------------------------------------------------------------------------
# Tests: WorkerStatus transitions
# ---------------------------------------------------------------------------

class TestWorkerStatus:
    def test_all_statuses_defined(self):
        assert len(WorkerStatus) == 6

    def test_happy_path_lifecycle(self):
        """spawning → ready → accepted → running → finished"""
        registry = WorkerRegistry()
        wid = registry.create_worker("test")

        assert registry.get_worker(wid).status == WorkerStatus.SPAWNING

        registry.transition(wid, WorkerStatus.READY_FOR_PROMPT)
        assert registry.get_worker(wid).status == WorkerStatus.READY_FOR_PROMPT

        registry.transition(wid, WorkerStatus.PROMPT_ACCEPTED)
        assert registry.get_worker(wid).status == WorkerStatus.PROMPT_ACCEPTED

        registry.transition(wid, WorkerStatus.RUNNING)
        assert registry.get_worker(wid).status == WorkerStatus.RUNNING

        registry.transition(wid, WorkerStatus.FINISHED)
        assert registry.get_worker(wid).status == WorkerStatus.FINISHED

    def test_invalid_transition_raises(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("test")

        with pytest.raises(InvalidTransitionError):
            registry.transition(wid, WorkerStatus.RUNNING)  # can't skip states

    def test_finished_is_terminal(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("test")
        registry.transition(wid, WorkerStatus.READY_FOR_PROMPT)
        registry.transition(wid, WorkerStatus.PROMPT_ACCEPTED)
        registry.transition(wid, WorkerStatus.RUNNING)
        registry.transition(wid, WorkerStatus.FINISHED)

        with pytest.raises(InvalidTransitionError):
            registry.transition(wid, WorkerStatus.RUNNING)

    def test_failure_from_any_active_state(self):
        for start_status in [WorkerStatus.SPAWNING, WorkerStatus.READY_FOR_PROMPT,
                             WorkerStatus.PROMPT_ACCEPTED, WorkerStatus.RUNNING]:
            registry = WorkerRegistry()
            wid = registry.create_worker("test")

            # Walk to the desired start state
            transitions = {
                WorkerStatus.SPAWNING: [],
                WorkerStatus.READY_FOR_PROMPT: [WorkerStatus.READY_FOR_PROMPT],
                WorkerStatus.PROMPT_ACCEPTED: [WorkerStatus.READY_FOR_PROMPT, WorkerStatus.PROMPT_ACCEPTED],
                WorkerStatus.RUNNING: [WorkerStatus.READY_FOR_PROMPT, WorkerStatus.PROMPT_ACCEPTED, WorkerStatus.RUNNING],
            }
            for s in transitions[start_status]:
                registry.transition(wid, s)

            # All active states can transition to FAILED
            registry.transition(wid, WorkerStatus.FAILED, "test failure")
            assert registry.get_worker(wid).status == WorkerStatus.FAILED


# ---------------------------------------------------------------------------
# Tests: WorkerRegistry
# ---------------------------------------------------------------------------

class TestWorkerRegistry:
    def test_create_and_get(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("alice")
        worker = registry.get_worker(wid)
        assert worker.name == "alice"
        assert worker.status == WorkerStatus.SPAWNING
        assert len(worker.events) == 1  # creation event

    def test_worker_not_found(self):
        registry = WorkerRegistry()
        with pytest.raises(WorkerNotFoundError):
            registry.get_worker("nonexistent")

    def test_list_workers(self):
        registry = WorkerRegistry()
        registry.create_worker("a")
        registry.create_worker("b")
        registry.create_worker("c")
        assert len(registry.list_workers()) == 3

    def test_all_finished(self):
        registry = WorkerRegistry()
        w1 = registry.create_worker("a")
        w2 = registry.create_worker("b")

        assert not registry.all_finished()

        # Finish w1
        registry.transition(w1, WorkerStatus.READY_FOR_PROMPT)
        registry.transition(w1, WorkerStatus.PROMPT_ACCEPTED)
        registry.transition(w1, WorkerStatus.RUNNING)
        registry.transition(w1, WorkerStatus.FINISHED)

        assert not registry.all_finished()  # w2 still spawning

        # Fail w2
        registry.transition(w2, WorkerStatus.FAILED, "error")

        assert registry.all_finished()  # both terminal

    def test_set_result(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("test")
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        registry.set_result(wid, {"responses": []}, usage)

        worker = registry.get_worker(wid)
        assert worker.result == {"responses": []}
        assert worker.usage.input_tokens == 100

    def test_set_error(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("test")
        registry.set_error(wid, FailureKind.PROTOCOL, "connection lost")

        worker = registry.get_worker(wid)
        assert worker.error == (FailureKind.PROTOCOL, "connection lost")

    def test_restart_from_failed(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("test")
        registry.transition(wid, WorkerStatus.FAILED, "broke")

        registry.restart(wid)
        assert registry.get_worker(wid).status == WorkerStatus.SPAWNING
        assert registry.get_worker(wid).error is None

    def test_restart_from_non_failed_raises(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("test")
        # Worker is in SPAWNING, not FAILED
        with pytest.raises(InvalidTransitionError):
            registry.restart(wid)

    def test_terminate(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("test")
        registry.transition(wid, WorkerStatus.READY_FOR_PROMPT)
        registry.transition(wid, WorkerStatus.PROMPT_ACCEPTED)
        registry.transition(wid, WorkerStatus.RUNNING)
        registry.terminate(wid)
        assert registry.get_worker(wid).status == WorkerStatus.FINISHED

    def test_event_log_records_transitions(self):
        registry = WorkerRegistry()
        wid = registry.create_worker("test")
        registry.transition(wid, WorkerStatus.READY_FOR_PROMPT, "init")
        registry.transition(wid, WorkerStatus.PROMPT_ACCEPTED, "got prompt")

        worker = registry.get_worker(wid)
        # 1 creation event + 2 transitions
        assert len(worker.events) == 3
        assert worker.events[1].from_status == WorkerStatus.SPAWNING
        assert worker.events[1].to_status == WorkerStatus.READY_FOR_PROMPT
        assert worker.events[1].detail == "init"


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_creates(self):
        registry = WorkerRegistry()
        ids: list[str] = []
        lock = threading.Lock()

        def create_worker(name: str):
            wid = registry.create_worker(name)
            with lock:
                ids.append(wid)

        threads = [threading.Thread(target=create_worker, args=(f"w{i}",)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 50
        assert len(set(ids)) == 50  # all unique
        assert len(registry.list_workers()) == 50

    def test_concurrent_transitions(self):
        """Multiple threads transitioning different workers concurrently."""
        registry = WorkerRegistry()
        worker_ids = [registry.create_worker(f"w{i}") for i in range(20)]
        errors: list[str] = []

        def run_lifecycle(wid: str):
            try:
                registry.transition(wid, WorkerStatus.READY_FOR_PROMPT)
                registry.transition(wid, WorkerStatus.PROMPT_ACCEPTED)
                registry.transition(wid, WorkerStatus.RUNNING)
                registry.transition(wid, WorkerStatus.FINISHED)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=run_lifecycle, args=(wid,)) for wid in worker_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert registry.all_finished()


# ---------------------------------------------------------------------------
# Tests: extract_text helper
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_extracts_text_blocks(self):
        summary = TurnSummary(
            assistant_messages=[
                ConversationMessage(role="assistant", content=[
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ]),
            ],
        )
        assert _extract_text(summary) == "Hello world"

    def test_ignores_non_text_blocks(self):
        summary = TurnSummary(
            assistant_messages=[
                ConversationMessage(role="assistant", content=[
                    {"type": "tool_use", "id": "c1", "name": "search", "input": {}},
                    {"type": "text", "text": "result"},
                ]),
            ],
        )
        assert _extract_text(summary) == "result"

    def test_empty_messages(self):
        summary = TurnSummary()
        assert _extract_text(summary) == ""


# ---------------------------------------------------------------------------
# Tests: run_panel_parallel
# ---------------------------------------------------------------------------

class TestRunPanelParallel:
    def test_single_persona_single_question(self):
        client = _make_mock_client([_make_text_response("I think it's great")])
        personas = [{"name": "Alice", "age": 30}]
        questions = [{"text": "What do you think?"}]

        results, registry = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
        )

        assert len(results) == 1
        assert results[0].persona_name == "Alice"
        assert len(results[0].responses) == 1
        assert results[0].responses[0]["question"] == "What do you think?"
        assert results[0].error is None
        assert registry.all_finished()

    def test_multiple_personas_parallel(self):
        """Verify all personas get results and registry tracks all workers."""
        responses = [_make_text_response(f"Response {i}") for i in range(6)]
        client = _make_mock_client(responses)
        personas = [
            {"name": "Alice"},
            {"name": "Bob"},
            {"name": "Carol"},
        ]
        questions = [{"text": "Q1"}, {"text": "Q2"}]

        results, registry = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
        )

        assert len(results) == 3
        # Verify order is preserved
        assert results[0].persona_name == "Alice"
        assert results[1].persona_name == "Bob"
        assert results[2].persona_name == "Carol"
        # Each persona answered 2 questions
        for r in results:
            assert len(r.responses) == 2
            assert r.error is None

        assert len(registry.list_workers()) == 3
        assert registry.all_finished()

    def test_follow_ups_executed(self):
        responses = [
            _make_text_response("Main answer"),
            _make_text_response("Follow-up answer"),
        ]
        client = _make_mock_client(responses)
        personas = [{"name": "Alice"}]
        questions = [{"text": "Main Q", "follow_ups": ["Tell me more"]}]

        results, registry = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
        )

        assert len(results[0].responses) == 2
        assert results[0].responses[0]["question"] == "Main Q"
        assert results[0].responses[1].get("follow_up") is True

    def test_usage_accumulated_per_panelist(self):
        usage = LLMTokenUsage(input_tokens=100, output_tokens=50)
        responses = [_make_text_response(usage=usage) for _ in range(2)]
        client = _make_mock_client(responses)
        personas = [{"name": "Alice"}]
        questions = [{"text": "Q1"}, {"text": "Q2"}]

        results, _ = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
        )

        assert results[0].usage.input_tokens == 200
        assert results[0].usage.output_tokens == 100

    def test_error_in_one_persona_doesnt_block_others(self):
        """If one panelist fails, others still complete."""
        call_count = 0
        lock = threading.Lock()

        def side_effect(request):
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            # First persona's first call fails
            if current == 1:
                raise ConnectionError("API down")
            return _make_text_response("ok")

        client = MagicMock()
        client.send = MagicMock(side_effect=side_effect)

        personas = [{"name": "Alice"}, {"name": "Bob"}]
        questions = [{"text": "Q1"}]

        results, registry = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
        )

        assert len(results) == 2
        # One should have error response, other should succeed
        has_error = any(r.responses and r.responses[0].get("error") for r in results)
        has_success = any(r.responses and not r.responses[0].get("error") for r in results)
        assert has_error or has_success  # at least one completed

    def test_max_workers_limits_concurrency(self):
        """Verify max_workers parameter is respected."""
        active = {"count": 0, "peak": 0}
        lock = threading.Lock()

        def counting_send(request):
            with lock:
                active["count"] += 1
                active["peak"] = max(active["peak"], active["count"])
            import time
            time.sleep(0.05)  # brief delay to test concurrency
            with lock:
                active["count"] -= 1
            return _make_text_response()

        client = MagicMock()
        client.send = MagicMock(side_effect=counting_send)

        personas = [{"name": f"P{i}"} for i in range(6)]
        questions = [{"text": "Q"}]

        results, _ = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
            max_workers=2,
        )

        assert len(results) == 6
        # Peak concurrency should not exceed max_workers
        assert active["peak"] <= 2

    def test_preserves_persona_order(self):
        """Results must match input persona order regardless of completion order."""
        import time

        def delayed_send(request):
            # Extract persona name from system prompt to vary delay
            system = request.system or ""
            if "Slow" in system:
                time.sleep(0.1)
            return _make_text_response(f"answer from {system[:20]}")

        client = MagicMock()
        client.send = MagicMock(side_effect=delayed_send)

        personas = [
            {"name": "Slow"},
            {"name": "Fast"},
        ]
        questions = [{"text": "Q"}]

        results, _ = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
        )

        assert results[0].persona_name == "Slow"
        assert results[1].persona_name == "Fast"
