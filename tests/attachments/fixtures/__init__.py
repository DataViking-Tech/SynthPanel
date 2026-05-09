"""Runtime fixture generators for attachment tests.

Producing fixture bytes in code (rather than checking in binary blobs)
keeps the repo small and portable across linux/mac/win — the same
function call yields byte-identical output regardless of host filesystem
or git autocrlf quirks. Each generator returns the smallest payload that
still round-trips through the relevant decoder.
"""

from __future__ import annotations

import struct
import zlib

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def tiny_png(width: int = 1, height: int = 1, *, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """Return a valid 1x1 (or arbitrary-size) RGB PNG.

    Hand-rolled so the test suite has no Pillow dependency.
    """

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b""
    r, g, b = color
    row = bytes([r, g, b]) * width
    for _ in range(height):
        raw += b"\x00" + row  # filter byte
    idat = zlib.compress(raw)
    return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def tiny_jpeg() -> bytes:
    """Return a minimal valid JPEG (1x1, white).

    Smallest known JFIF/JPEG payload that decodes cleanly. Used to assert
    media_type sniffing and persistence; we never decode the pixels.
    """
    return bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
        "070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c"
        "1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b0c180d"
        "0d1832211c213232323232323232323232323232323232323232323232323232323232"
        "323232323232323232323232323232323232323232323232ffc00011080001000103"
        "012200021101031101ffc4001f0000010501010101010100000000000000000102030"
        "405060708090a0bffc400b5100002010303020403050504040000017d010203000411"
        "05122131410613516107227114328191a1082342b1c11552d1f02433627282090a16"
        "1718191a25262728292a3435363738393a434445464748494a535455565758595a63"
        "6465666768696a737475767778797a838485868788898a92939495969798999aa2a3"
        "a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9da"
        "e1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101"
        "0101000000000000010203040506070809000affc400b51100020102040403040705"
        "04040001027700010203110405213106124151076171132232810814429112335215"
        "1ffda000c03010002110311003f00fbd2800a2800a2803ffd9"
    )


def tiny_gif() -> bytes:
    """Return a minimal valid 1x1 GIF89a (transparent)."""
    return bytes.fromhex("47494638396101000100800000ffffff00000021f9040100000000002c000000000100010000020144003b")


def tiny_webp() -> bytes:
    """Return a minimal valid 1x1 WebP (lossy)."""
    return bytes.fromhex(
        "524946462600000057454250565038200a000000010000400006001100002f00005650384c00000000302f410000000028"
    )


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------


def tiny_pdf_text() -> bytes:
    """Return a minimal valid text-bearing PDF (single page, 'Hello').

    Hand-built so we're not coupled to pypdfium2 install state for tests
    that only need bytes-on-disk semantics (CAS, persistence, sniffing).
    """
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type/Catalog/Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type/Pages/Kids[3 0 R]/Count 1>> endobj\n"
        b"3 0 obj <</Type/Page/Parent 2 0 R/Resources<<"
        b"/Font<</F1 5 0 R>>>>"
        b"/MediaBox[0 0 612 792]/Contents 4 0 R>> endobj\n"
        b"4 0 obj <</Length 44>> stream\n"
        b"BT /F1 24 Tf 100 700 Td (Hello, attachments!) Tj ET\n"
        b"endstream endobj\n"
        b"5 0 obj <</Type/Font/Subtype/Type1/BaseFont/Helvetica>> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"0000000053 00000 n \n"
        b"0000000098 00000 n \n"
        b"0000000180 00000 n \n"
        b"0000000260 00000 n \n"
        b"trailer <</Size 6/Root 1 0 R>>\n"
        b"startxref\n320\n%%EOF\n"
    )
    return body


def tiny_pdf_encrypted_marker() -> bytes:
    """Return a stub PDF that includes an /Encrypt entry in the trailer.

    The bytes are NOT a valid encrypted PDF — only enough to trip the
    "encrypted" probe heuristic (presence of the /Encrypt token).
    """
    return b"%PDF-1.4\n1 0 obj <<>> endobj\ntrailer <</Encrypt 99 0 R>>\n%%EOF\n"


# ---------------------------------------------------------------------------
# HTML / SPA / paywalled mocks
# ---------------------------------------------------------------------------


def html_markdown_friendly() -> bytes:
    return (
        b"<!doctype html><html><head><title>Doc</title></head><body>"
        b"<article><h1>Heading</h1><p>Paragraph one.</p>"
        b"<p>Paragraph two with <a href='https://example.com'>link</a>.</p>"
        b"</article></body></html>"
    )


def html_spa_shell() -> bytes:
    """SPA shell: nearly-empty body, content rendered client-side."""
    return (
        b"<!doctype html><html><head><title>App</title></head><body>"
        b"<div id='root'></div>"
        b"<script>document.getElementById('root').innerText='loaded';</script>"
        b"</body></html>"
    )


def html_paywalled() -> bytes:
    return (
        b"<!doctype html><html><head><title>Paywall</title></head><body>"
        b"<section class='paywall'><h2>Subscribe to read</h2>"
        b"<p>This article is for subscribers only.</p></section>"
        b"</body></html>"
    )


__all__ = [
    "html_markdown_friendly",
    "html_paywalled",
    "html_spa_shell",
    "tiny_gif",
    "tiny_jpeg",
    "tiny_pdf_encrypted_marker",
    "tiny_pdf_text",
    "tiny_png",
    "tiny_webp",
]


def _selftest() -> None:
    """Smoke-test signatures so a malformed fixture is caught early."""
    assert tiny_png().startswith(b"\x89PNG")
    assert tiny_jpeg().startswith(b"\xff\xd8\xff")
    assert tiny_gif().startswith(b"GIF89a")
    assert tiny_webp()[:4] == b"RIFF" and tiny_webp()[8:12] == b"WEBP"
    assert tiny_pdf_text().startswith(b"%PDF-")
    assert b"/Encrypt" in tiny_pdf_encrypted_marker()
    for fn in (html_markdown_friendly, html_spa_shell, html_paywalled):
        assert b"<html" in fn()
    # Decode round-trip: PNG must decompress cleanly.
    sig = b"\x89PNG\r\n\x1a\n"
    assert tiny_png().startswith(sig)
    # Pull the IDAT chunk and decompress to confirm checksum integrity.
    payload = tiny_png()[len(sig) :]
    pos = 0
    while pos < len(payload):
        length = int.from_bytes(payload[pos : pos + 4], "big")
        tag = payload[pos + 4 : pos + 8]
        chunk_data = payload[pos + 8 : pos + 8 + length]
        if tag == b"IDAT":
            zlib.decompress(chunk_data)
            return
        pos += 8 + length + 4
    raise AssertionError("PNG missing IDAT")


_selftest()
