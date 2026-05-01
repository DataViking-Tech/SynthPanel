"""Tests for synth_panel.text_width display-width helpers."""

from __future__ import annotations

from synth_panel.text_width import char_width, display_width, pad, truncate


class TestDisplayWidth:
    def test_ascii(self) -> None:
        assert display_width("hello") == 5
        assert display_width("") == 0
        assert display_width(" ") == 1

    def test_accented_latin_precomposed(self) -> None:
        # NFC-precomposed forms count as their visible cells.
        assert display_width("José") == 4
        assert display_width("Renée") == 5
        assert display_width("Naïve") == 5

    def test_accented_latin_decomposed(self) -> None:
        # NFD-decomposed (base + combining mark) should normalize to NFC
        # and report the same width as the precomposed form.
        nfd_jose = "José"  # 'e' + COMBINING ACUTE ACCENT
        assert display_width(nfd_jose) == 4

    def test_cjk_wide(self) -> None:
        # CJK characters render as 2 cells each in monospace terminals.
        assert display_width("王芳") == 4
        assert display_width("日本語") == 6

    def test_emoji_wide(self) -> None:
        # Common emoji blocks render as 2 cells.
        assert display_width("🌸") == 2
        assert display_width("🚀") == 2
        assert display_width("Naoko 🌸") == 8  # 5 + 1 + 2

    def test_zero_width_chars(self) -> None:
        # Zero-width joiner / format chars contribute 0.
        assert display_width("a​b") == 2  # zero-width space
        assert display_width("a‍b") == 2  # zero-width joiner

    def test_control_chars(self) -> None:
        assert display_width("a\x00b") == 2
        assert display_width("a\tb") == 2  # tab is Cc, width 0


class TestCharWidth:
    def test_ascii_letter(self) -> None:
        assert char_width("a") == 1

    def test_wide_cjk(self) -> None:
        assert char_width("王") == 2

    def test_emoji(self) -> None:
        assert char_width("🌸") == 2
        assert char_width("☀") == 2  # Misc Symbols block

    def test_combining_mark(self) -> None:
        assert char_width("́") == 0  # COMBINING ACUTE ACCENT

    def test_empty(self) -> None:
        assert char_width("") == 0


class TestPad:
    def test_ascii_left(self) -> None:
        assert pad("ab", 5) == "ab   "
        assert pad("abcde", 5) == "abcde"
        assert pad("abcdef", 5) == "abcdef"  # no truncation

    def test_ascii_right(self) -> None:
        assert pad("ab", 5, align="right") == "   ab"

    def test_ascii_center(self) -> None:
        assert pad("ab", 6, align="center") == "  ab  "
        assert pad("ab", 5, align="center") == " ab  "  # extra space goes right

    def test_wide_chars(self) -> None:
        # "王芳" is 4 cells. Pad to 6 → 2 spaces.
        assert pad("王芳", 6) == "王芳  "
        assert pad("王芳", 6, align="right") == "  王芳"

    def test_emoji(self) -> None:
        # "🌸" is 2 cells. Pad to 5 → 3 spaces.
        assert pad("🌸", 5) == "🌸   "

    def test_mixed(self) -> None:
        # "Naoko 🌸" is 8 cells. Pad to 12 → 4 spaces.
        assert pad("Naoko 🌸", 12) == "Naoko 🌸    "

    def test_alignment_consistency(self) -> None:
        # The whole point: padded strings should all reach the same
        # rendered width regardless of input character widths.
        names = ["José", "Naoko 🌸", "王芳", "Sarah Chen"]
        target = max(display_width(n) for n in names) + 2
        padded = [pad(n, target) for n in names]
        assert all(display_width(p) == target for p in padded)


class TestTruncate:
    def test_ascii_no_truncation(self) -> None:
        assert truncate("hello", 10) == "hello"

    def test_ascii_truncated(self) -> None:
        assert truncate("hello world", 5) == "hello"

    def test_with_ellipsis(self) -> None:
        assert truncate("hello world", 8, ellipsis="…") == "hello w…"

    def test_wide_chars_no_split(self) -> None:
        # Truncating "王芳子" to 5 cells: "王" (2) + "芳" (2) = 4. The
        # next "子" would push to 6, so it must be dropped — we should
        # not return a half-width fragment.
        assert truncate("王芳子", 5) == "王芳"

    def test_wide_chars_exact(self) -> None:
        assert truncate("王芳", 4) == "王芳"

    def test_emoji_not_split(self) -> None:
        # "ab🌸" is 4 cells. Truncate to 3 → "ab" (no half-emoji).
        assert truncate("ab🌸", 3) == "ab"
