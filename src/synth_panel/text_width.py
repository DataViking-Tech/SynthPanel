"""Display-width helpers for terminal output with non-ASCII text.

Persona names and synthesized themes are user-supplied — they may contain
CJK characters, accented Latin (with combining marks), and emoji. Naive
``len()``-based padding (``f"{name:<30}"``, ``str.ljust``) counts code
points rather than rendered cells, so those characters break column
alignment.

This module uses only ``unicodedata`` from the stdlib (``synthpanel``
intentionally keeps dependencies minimal). It handles the common cases:
East Asian Wide / Fullwidth (width 2), combining marks and zero-width
formatting characters (width 0), and the most common emoji ranges
(width 2). It does *not* try to resolve grapheme clusters or ZWJ
sequences — those produce slight misalignment in very rare inputs but
the alternative is a third-party dep (``wcwidth``) or a multi-kilobyte
table of grapheme break properties.
"""

from __future__ import annotations

import unicodedata

# Unicode codepoint ranges that render as 2 cells in most terminals but
# are reported as Neutral by ``unicodedata.east_asian_width``.
_WIDE_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F300, 0x1F64F),  # Misc Symbols and Pictographs + Emoticons
    (0x1F680, 0x1F6FF),  # Transport and Map Symbols
    (0x1F700, 0x1F77F),  # Alchemical Symbols
    (0x1F780, 0x1F7FF),  # Geometric Shapes Extended
    (0x1F800, 0x1F8FF),  # Supplemental Arrows-C
    (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    (0x1FA00, 0x1FA6F),  # Chess Symbols
    (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
    (0x2600, 0x26FF),    # Miscellaneous Symbols (☀ ☁ ⚡ etc.)
    (0x2700, 0x27BF),    # Dingbats (✂ ✈ ❤ etc.)
)


def _is_wide_emoji(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _WIDE_EMOJI_RANGES)


def char_width(ch: str) -> int:
    """Return the rendered width (in terminal cells) of a single character.

    - ASCII printable: 1
    - Combining marks, format chars, control chars: 0
    - East Asian Wide / Fullwidth: 2
    - Emoji in common Unicode blocks: 2
    - Everything else: 1
    """
    if not ch:
        return 0
    cp = ord(ch)
    # ASCII fast path
    if 0x20 <= cp < 0x7F:
        return 1
    cat = unicodedata.category(ch)
    # Combining (Mn = nonspacing mark, Me = enclosing mark),
    # format/zero-width (Cf), and control (Cc) — width 0.
    if cat in ("Mn", "Me", "Cf", "Cc"):
        return 0
    eaw = unicodedata.east_asian_width(ch)
    if eaw in ("W", "F"):
        return 2
    if _is_wide_emoji(cp):
        return 2
    return 1


def display_width(s: str) -> int:
    """Return the rendered width of ``s`` in terminal cells.

    Normalizes to NFC first so decomposed accented forms (e.g. ``"e"`` +
    combining acute) collapse into a single precomposed code point and
    contribute width 1 instead of 1 + 0.
    """
    if not s:
        return 0
    s = unicodedata.normalize("NFC", s)
    return sum(char_width(ch) for ch in s)


def pad(s: str, width: int, *, align: str = "left") -> str:
    """Pad ``s`` to ``width`` rendered cells with spaces.

    Drop-in replacement for ``f"{s:<{width}}"`` / ``s.ljust(width)`` that
    accounts for wide and zero-width characters. ``align`` is ``"left"``
    (default), ``"right"``, or ``"center"``. If ``s`` is already at or
    over ``width``, it is returned unchanged.
    """
    s = unicodedata.normalize("NFC", s)
    cur = sum(char_width(ch) for ch in s)
    if cur >= width:
        return s
    extra = width - cur
    if align == "right":
        return " " * extra + s
    if align == "center":
        left = extra // 2
        return " " * left + s + " " * (extra - left)
    return s + " " * extra


def truncate(s: str, max_width: int, *, ellipsis: str = "") -> str:
    """Truncate ``s`` to fit within ``max_width`` rendered cells.

    Counterpart to ``s[:n]`` that respects rendered width. If ``ellipsis``
    is provided and the string is truncated, the ellipsis is appended and
    counted against the budget.
    """
    s = unicodedata.normalize("NFC", s)
    cur = sum(char_width(ch) for ch in s)
    if cur <= max_width:
        return s
    elps_w = sum(char_width(ch) for ch in ellipsis)
    target = max(0, max_width - elps_w)
    out: list[str] = []
    used = 0
    for ch in s:
        w = char_width(ch)
        if used + w > target:
            break
        out.append(ch)
        used += w
    return "".join(out) + ellipsis
