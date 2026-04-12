from __future__ import annotations

from functools import lru_cache
import gzip
from pathlib import Path

from .tokenization import normalize_alphabet


@lru_cache(maxsize=None)
def _fasta_translation_tables(alphabet: str) -> tuple[bytes, bytes]:
    normalized_alphabet = normalize_alphabet(alphabet)
    allowed = set(normalized_alphabet)
    fallback = "N" if "N" in allowed else None
    translation = bytearray(range(256))
    delete_bytes = bytearray()

    for byte_value in range(256):
        ch = chr(byte_value)
        upper = ch.upper()
        if ch.isalpha() and upper in allowed:
            translation[byte_value] = ord(upper)
            continue
        if ch.isalpha() and fallback is not None:
            translation[byte_value] = ord(fallback)
            continue
        delete_bytes.append(byte_value)

    return bytes(translation), bytes(delete_bytes)


def sanitize_fasta_bytes(path: Path, alphabet: str) -> tuple[bytes, set[int]]:
    translation, delete_bytes = _fasta_translation_tables(alphabet)
    output: bytearray = bytearray()
    observed: set[int] = set()

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(b">"):
                continue
            filtered = line.translate(translation, delete_bytes)
            if not filtered:
                continue
            output.extend(filtered)
            observed.update(filtered)

    return bytes(output), observed
