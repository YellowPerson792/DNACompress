from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import DataConfig, ModelConfig
from .nugget_tokenization import (
    NUGGET_TOKENIZERS as BYTECAPTION_TOKENIZERS,
    NuggetCacheSourceDescriptor as ByteCaptionCacheSourceDescriptor,
    NuggetTokenCacheStats as ByteCaptionTokenCacheStats,
    NuggetTokenizedSource as ByteCaptionTokenizedSource,
    NuggetTokenizerSpec as ByteCaptionTokenizerSpec,
    apply_nugget_tokenizer_to_model_config,
    build_nugget_tokenizer_spec,
    tokenize_nugget_source,
    tokenize_nugget_sources_with_cache,
)


def build_bytecaption_tokenizer_spec(data_config: DataConfig, model_config: ModelConfig) -> ByteCaptionTokenizerSpec:
    return build_nugget_tokenizer_spec(data_config, model_config)


def apply_bytecaption_tokenizer_to_model_config(model_config: ModelConfig, spec: ByteCaptionTokenizerSpec) -> None:
    apply_nugget_tokenizer_to_model_config(model_config, spec)


def tokenize_bytecaption_source(
    *,
    species: str,
    source_name: str | None = None,
    source: bytes,
    spec: ByteCaptionTokenizerSpec,
    species_prefix_map: dict[str, str] | None = None,
    drop_tail_to_full_kmer: bool = False,
) -> ByteCaptionTokenizedSource:
    return tokenize_nugget_source(
        species=species,
        source_name=source_name,
        source=source,
        spec=spec,
        species_prefix_map=species_prefix_map,
        drop_tail_to_full_kmer=drop_tail_to_full_kmer,
    )


def _bytecaption_cache_dir(configured_dir: str | None) -> str:
    if configured_dir is None:
        return ".dna_cache/bytecaption"
    configured = Path(configured_dir)
    if configured.is_absolute():
        return str(configured / "bytecaption")
    return str(configured / "bytecaption")


def tokenize_bytecaption_sources_with_cache(
    *,
    source_descriptors: Iterable[ByteCaptionCacheSourceDescriptor],
    spec: ByteCaptionTokenizerSpec,
    dataset_dir: Path,
    cache_enabled: bool,
    cache_dir: str | None,
    species_prefix_map: dict[str, str] | None = None,
    drop_tail_to_full_kmer: bool = False,
    split_scope: str | None = None,
) -> tuple[list[ByteCaptionTokenizedSource], ByteCaptionTokenCacheStats]:
    # Reuse the mature Nugget token cache implementation, but force a distinct
    # directory namespace so ByteCaption token caches never collide with Nugget.
    return tokenize_nugget_sources_with_cache(
        source_descriptors=source_descriptors,
        spec=spec,
        dataset_dir=dataset_dir,
        cache_enabled=cache_enabled,
        cache_dir=_bytecaption_cache_dir(cache_dir),
        species_prefix_map=species_prefix_map,
        drop_tail_to_full_kmer=drop_tail_to_full_kmer,
        split_scope=f"bytecaption_{split_scope}" if split_scope is not None else "bytecaption",
    )
