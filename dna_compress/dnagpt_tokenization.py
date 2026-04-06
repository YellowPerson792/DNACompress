from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TokenizedDNASource:
    species: str
    prefix_token: str | None
    prefix_ids: list[int]
    dna_token_ids: list[int]
    dna_token_base_lengths: list[int]
    total_bases: int


def _normalize_sequence(source: bytes) -> str:
    return source.decode("ascii").strip().upper()


def _chunk_sequence(sequence: str, chunk_size: int) -> list[str]:
    return [sequence[index : index + chunk_size] for index in range(0, len(sequence), chunk_size)]


def resolve_species_prefix_token(species: str, species_prefix_map: dict[str, str] | None) -> str | None:
    if not species_prefix_map:
        return None
    token_name = species_prefix_map.get(species)
    if token_name is None:
        return None
    cleaned = token_name.strip()
    if not cleaned:
        return None
    return cleaned


def tokenize_dna_source(
    *,
    species: str,
    source: bytes,
    tokenizer,
    kmer_size: int,
    species_prefix_map: dict[str, str] | None = None,
) -> TokenizedDNASource:
    prefix_token = resolve_species_prefix_token(species, species_prefix_map)
    prefix_ids: list[int] = []
    if prefix_token is not None:
        token_text = prefix_token if prefix_token.startswith("<") else f"<{prefix_token}>"
        token_id = tokenizer.piece_to_id(token_text)
        if token_id == tokenizer.unk_id and token_text != tokenizer.id_to_piece(tokenizer.unk_id):
            raise ValueError(
                f"Unknown DNAGPT special token '{prefix_token}' configured for species '{species}'."
            )
        prefix_ids.append(int(token_id))

    sequence = _normalize_sequence(source)
    pieces = _chunk_sequence(sequence, kmer_size)
    dna_token_ids = [int(tokenizer.piece_to_id(piece)) for piece in pieces if piece]
    dna_token_base_lengths = [len(piece) for piece in pieces if piece]
    return TokenizedDNASource(
        species=species,
        prefix_token=prefix_token,
        prefix_ids=prefix_ids,
        dna_token_ids=dna_token_ids,
        dna_token_base_lengths=dna_token_base_lengths,
        total_bases=len(sequence),
    )


def tokenize_dna_sources(
    *,
    species_names: Iterable[str],
    sources: Iterable[bytes],
    tokenizer,
    kmer_size: int,
    species_prefix_map: dict[str, str] | None = None,
) -> list[TokenizedDNASource]:
    return [
        tokenize_dna_source(
            species=species,
            source=source,
            tokenizer=tokenizer,
            kmer_size=kmer_size,
            species_prefix_map=species_prefix_map,
        )
        for species, source in zip(species_names, sources)
    ]
