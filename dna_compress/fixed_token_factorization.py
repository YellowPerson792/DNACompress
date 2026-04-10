from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FixedTokenArithmeticFactorizer:
    alphabet: str
    model_merge_size: int
    arithmetic_merge_size: int
    regular_vocab_size: int
    special_token_ids: torch.Tensor
    special_token_id_to_index: torch.Tensor
    chunk_vocab_sizes: tuple[int, ...]
    chunk_divisors: torch.Tensor

    @property
    def base(self) -> int:
        return len(self.alphabet)

    @property
    def root_vocab_size(self) -> int:
        return 1 + self.chunk_vocab_sizes[0]

    @property
    def max_emitted_vocab_size(self) -> int:
        vocab_sizes = [self.root_vocab_size, *self.chunk_vocab_sizes[1:]]
        if int(self.special_token_ids.numel()) > 0:
            vocab_sizes.append(int(self.special_token_ids.numel()))
        return max(vocab_sizes)

    def to(self, device: torch.device | str) -> "FixedTokenArithmeticFactorizer":
        device = torch.device(device)
        return FixedTokenArithmeticFactorizer(
            alphabet=self.alphabet,
            model_merge_size=self.model_merge_size,
            arithmetic_merge_size=self.arithmetic_merge_size,
            regular_vocab_size=self.regular_vocab_size,
            special_token_ids=self.special_token_ids.to(device=device),
            special_token_id_to_index=self.special_token_id_to_index.to(device=device),
            chunk_vocab_sizes=self.chunk_vocab_sizes,
            chunk_divisors=self.chunk_divisors.to(device=device),
        )

    def decode_chunk_symbols(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(dtype=torch.long)
        chunks: list[torch.Tensor] = []
        for divisor, vocab_size in zip(self.chunk_divisors.tolist(), self.chunk_vocab_sizes):
            chunk = torch.div(token_ids, int(divisor), rounding_mode="floor")
            chunks.append(torch.remainder(chunk, int(vocab_size)))
        return torch.stack(chunks, dim=1)


@dataclass(frozen=True)
class FixedTokenFactorization:
    root_probabilities: torch.Tensor
    root_symbols: torch.Tensor
    regular_step_probabilities: tuple[torch.Tensor, ...]
    regular_step_symbols: tuple[torch.Tensor, ...]
    regular_row_positions: torch.Tensor
    special_step_probabilities: torch.Tensor
    special_step_symbols: torch.Tensor
    special_row_positions: torch.Tensor
    target_log_probs: torch.Tensor

    @property
    def emitted_symbol_count(self) -> int:
        count = int(self.root_symbols.shape[0])
        count += sum(int(step.shape[0]) for step in self.regular_step_symbols)
        count += int(self.special_step_symbols.shape[0])
        return count


def build_fixed_token_arithmetic_factorizer(
    *,
    vocab_size: int,
    special_token_ids: list[int] | tuple[int, ...],
    model_merge_size: int,
    arithmetic_merge_size: int,
    alphabet: str,
) -> FixedTokenArithmeticFactorizer:
    if model_merge_size <= 0:
        raise ValueError("model_merge_size must be >= 1")
    if arithmetic_merge_size <= 0:
        raise ValueError("arithmetic_merge_size must be >= 1")
    if arithmetic_merge_size > model_merge_size:
        raise ValueError("arithmetic_merge_size must be <= model_merge_size")

    base = len(alphabet)
    regular_vocab_size = base ** model_merge_size
    special_ids = sorted({int(token_id) for token_id in special_token_ids})
    if regular_vocab_size + len(special_ids) != vocab_size:
        raise ValueError(
            "vocab_size must equal base**model_merge_size plus the number of special tokens "
            f"(expected {regular_vocab_size + len(special_ids)}, got {vocab_size})"
        )
    if any(token_id < regular_vocab_size or token_id >= vocab_size for token_id in special_ids):
        raise ValueError("special_token_ids must refer to the trailing special-token range.")

    chunk_vocab_sizes: list[int] = []
    remaining = model_merge_size
    while remaining > 0:
        chunk_size = min(arithmetic_merge_size, remaining)
        chunk_vocab_sizes.append(base ** chunk_size)
        remaining -= chunk_size

    divisors: list[int] = []
    running = 1
    for vocab_width in reversed(chunk_vocab_sizes[1:]):
        running *= vocab_width
        divisors.append(running)
    divisors = list(reversed(divisors))
    divisors.append(1)

    special_token_id_to_index = torch.full((vocab_size,), -1, dtype=torch.long)
    if special_ids:
        special_token_id_to_index[torch.tensor(special_ids, dtype=torch.long)] = torch.arange(
            len(special_ids),
            dtype=torch.long,
        )

    return FixedTokenArithmeticFactorizer(
        alphabet=alphabet,
        model_merge_size=model_merge_size,
        arithmetic_merge_size=arithmetic_merge_size,
        regular_vocab_size=regular_vocab_size,
        special_token_ids=torch.tensor(special_ids, dtype=torch.long),
        special_token_id_to_index=special_token_id_to_index,
        chunk_vocab_sizes=tuple(chunk_vocab_sizes),
        chunk_divisors=torch.tensor(divisors, dtype=torch.long),
    )


def factorize_fixed_token_log_probs(
    log_probs: torch.Tensor,
    target_token_ids: torch.Tensor,
    factorizer: FixedTokenArithmeticFactorizer,
) -> FixedTokenFactorization:
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be a 2D tensor of shape [num_targets, vocab_size].")
    if target_token_ids.ndim != 1:
        raise ValueError("target_token_ids must be a 1D tensor.")
    if log_probs.shape[0] != target_token_ids.shape[0]:
        raise ValueError("log_probs and target_token_ids must have the same leading dimension.")

    device = log_probs.device
    work_log_probs = log_probs.float()
    target_token_ids = target_token_ids.to(device=device, dtype=torch.long)
    target_log_probs = work_log_probs.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)

    regular_probs = work_log_probs[:, : factorizer.regular_vocab_size].exp().reshape(
        work_log_probs.shape[0],
        *factorizer.chunk_vocab_sizes,
    )
    if int(factorizer.special_token_ids.numel()) > 0:
        special_probs = work_log_probs.index_select(1, factorizer.special_token_ids).exp()
        other_mass = special_probs.sum(dim=1)
    else:
        special_probs = work_log_probs.new_zeros((work_log_probs.shape[0], 0))
        other_mass = work_log_probs.new_zeros((work_log_probs.shape[0],))

    if regular_probs.ndim == 2:
        root_regular_mass = regular_probs
    else:
        root_regular_mass = regular_probs.sum(dim=tuple(range(2, regular_probs.ndim)))

    special_target_indices = factorizer.special_token_id_to_index[target_token_ids]
    special_mask = special_target_indices >= 0
    regular_mask = ~special_mask

    root_probabilities = torch.cat([other_mass.unsqueeze(1), root_regular_mass], dim=1)
    root_symbols = torch.zeros((target_token_ids.shape[0],), dtype=torch.long, device=device)

    regular_row_indices = regular_mask.nonzero(as_tuple=False).squeeze(1)
    regular_row_positions = torch.full((target_token_ids.shape[0],), -1, dtype=torch.long, device=device)
    regular_step_probabilities: list[torch.Tensor] = []
    regular_step_symbols: list[torch.Tensor] = []

    if int(regular_row_indices.numel()) > 0:
        regular_target_ids = target_token_ids.index_select(0, regular_row_indices)
        regular_chunk_symbols = factorizer.decode_chunk_symbols(regular_target_ids)
        root_symbols[regular_row_indices] = regular_chunk_symbols[:, 0] + 1
        regular_row_positions[regular_row_indices] = torch.arange(
            regular_row_indices.shape[0],
            dtype=torch.long,
            device=device,
        )

        current = regular_probs.index_select(0, regular_row_indices)
        for step_index in range(1, len(factorizer.chunk_vocab_sizes)):
            chosen_previous = regular_chunk_symbols[:, step_index - 1]
            current = current[torch.arange(current.shape[0], device=device), chosen_previous]
            if current.ndim == 2:
                step_mass = current
            else:
                step_mass = current.sum(dim=tuple(range(2, current.ndim)))
            conditional = step_mass / step_mass.sum(dim=1, keepdim=True).clamp_min(1e-30)
            regular_step_probabilities.append(conditional)
            regular_step_symbols.append(regular_chunk_symbols[:, step_index])

    special_row_indices = special_mask.nonzero(as_tuple=False).squeeze(1)
    special_row_positions = torch.full((target_token_ids.shape[0],), -1, dtype=torch.long, device=device)
    if int(special_row_indices.numel()) > 0:
        root_symbols[special_row_indices] = 0
        special_row_positions[special_row_indices] = torch.arange(
            special_row_indices.shape[0],
            dtype=torch.long,
            device=device,
        )
        special_step_probabilities = special_probs.index_select(0, special_row_indices)
        special_step_probabilities = special_step_probabilities / special_step_probabilities.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-30)
        special_step_symbols = special_target_indices.index_select(0, special_row_indices)
    else:
        special_step_probabilities = work_log_probs.new_zeros((0, int(factorizer.special_token_ids.numel())))
        special_step_symbols = torch.zeros((0,), dtype=torch.long, device=device)

    return FixedTokenFactorization(
        root_probabilities=root_probabilities,
        root_symbols=root_symbols,
        regular_step_probabilities=tuple(regular_step_probabilities),
        regular_step_symbols=tuple(regular_step_symbols),
        regular_row_positions=regular_row_positions,
        special_step_probabilities=special_step_probabilities,
        special_step_symbols=special_step_symbols,
        special_row_positions=special_row_positions,
        target_log_probs=target_log_probs,
    )
