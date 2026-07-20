"""Cross-cutting test suite for althing attachments (hq-3o1r).

Per-feature unit tests live next to the feature in ``tests/test_attachments.py``,
``tests/test_attachments_caching.py``, ``tests/test_attachments_filter.py``,
``tests/test_attachments_pdf.py``, and ``tests/test_fetch.py``.

This package layers cross-cutting tests on top:

* :mod:`tests.attachments.test_persistence` — CAS round-trip + refs.json +
  ``result_format_version`` bump + ``ANNOTATED_CHOICE_SCHEMA`` shape.
* :mod:`tests.attachments.test_integration` — parse → validate → run → persist
  → readback against a mock LLM.
* :mod:`tests.attachments.fixtures` — generators for sample image / pdf / html
  bytes used across the integration tests; produced at runtime so the suite
  stays small and portable across linux/mac/win.
"""
