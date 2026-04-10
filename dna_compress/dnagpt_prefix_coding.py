from __future__ import annotations

from dataclasses import dataclass

import torch


DNAGPT_PREFIX_BASE_ORDER = "NAGCT"
DNAGPT_PREFIX_ALPHABET_SIZE = 1 + len(DNAGPT_PREFIX_BASE_ORDER)


@dataclass(frozen=True)
class DNAGPTPrefixTrie:
    dna_token_ids: torch.Tensor
    reserved_token_ids: torch.Tensor
    token_id_to_node_index: torch.Tensor
    node_exact_token_ids: torch.Tensor
    node_depths: torch.Tensor
    node_base_ids: torch.Tensor
    node_prefix_path_indices: torch.Tensor
    child_indices: torch.Tensor
    nodes_by_depth: tuple[torch.Tensor, ...]
    max_token_length: int
    root_index: int = 0
    base_order: str = DNAGPT_PREFIX_BASE_ORDER

    @property
    def root_child_indices(self) -> torch.Tensor:
        return self.child_indices[self.root_index]

    @property
    def exact_node_indices(self) -> torch.Tensor:
        return (self.node_exact_token_ids >= 0).nonzero(as_tuple=False).squeeze(1)

    def to(self, device: torch.device | str) -> "DNAGPTPrefixTrie":
        device = torch.device(device)
        return DNAGPTPrefixTrie(
            dna_token_ids=self.dna_token_ids.to(device=device),
            reserved_token_ids=self.reserved_token_ids.to(device=device),
            token_id_to_node_index=self.token_id_to_node_index.to(device=device),
            node_exact_token_ids=self.node_exact_token_ids.to(device=device),
            node_depths=self.node_depths.to(device=device),
            node_base_ids=self.node_base_ids.to(device=device),
            node_prefix_path_indices=self.node_prefix_path_indices.to(device=device),
            child_indices=self.child_indices.to(device=device),
            nodes_by_depth=tuple(item.to(device=device) for item in self.nodes_by_depth),
            max_token_length=self.max_token_length,
            root_index=self.root_index,
            base_order=self.base_order,
        )


@dataclass(frozen=True)
class DNAGPTPrefixFactorization:
    emitted_probabilities: torch.Tensor
    emitted_symbols: torch.Tensor
    emitted_valid_mask: torch.Tensor
    target_log_probs: torch.Tensor

    @property
    def emitted_symbol_count(self) -> int:
        return int(self.emitted_valid_mask.sum().item())


@dataclass(frozen=True)
class DNAGPTGroupedPrefixFactorization:
    step_probabilities: tuple[torch.Tensor, ...]
    step_symbols: tuple[torch.Tensor, ...]
    step_row_positions: tuple[torch.Tensor, ...]
    target_log_probs: torch.Tensor

    @property
    def emitted_symbol_count(self) -> int:
        return sum(int(step.shape[0]) for step in self.step_symbols)

    @property
    def max_emitted_vocab_size(self) -> int:
        if not self.step_probabilities:
            return 0
        return max(int(step.shape[1]) for step in self.step_probabilities)


def _gather_row_values(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return source.new_zeros((source.shape[0], 0))
    return source.gather(1, indices)


def _compute_chunk_ids(chunk_digits: torch.Tensor, chunk_length: int, base: int) -> torch.Tensor:
    values = torch.zeros((chunk_digits.shape[0],), dtype=torch.long, device=chunk_digits.device)
    for offset in range(chunk_length):
        values = values * base + chunk_digits[:, offset].to(dtype=torch.long)
    return values


def _term_block_offset(
    *,
    include_other: bool,
    term_length: int,
    base: int,
) -> int:
    offset = 1 if include_other else 0
    for length in range(1, term_length):
        offset += base ** length
    return offset


def _root_vocab_size(merge_size: int, max_token_length: int, base: int) -> int:
    term_max_length = min(merge_size, max_token_length)
    width = 1 + sum(base ** length for length in range(1, term_max_length + 1))
    if max_token_length > merge_size:
        width += base ** merge_size
    return width


def grouped_prefix_vocab_size(
    *,
    merge_size: int,
    max_token_length: int,
    base: int = len(DNAGPT_PREFIX_BASE_ORDER),
) -> int:
    if merge_size <= 0:
        raise ValueError("merge_size must be >= 1")
    if merge_size > max_token_length:
        raise ValueError("merge_size must be <= max_token_length")
    return _root_vocab_size(merge_size, max_token_length, base)


def build_dnagpt_prefix_trie(tokenizer) -> DNAGPTPrefixTrie:
    base_to_index = {base: index for index, base in enumerate(DNAGPT_PREFIX_BASE_ORDER)}
    dna_token_ids: list[int] = []
    reserved_token_ids: list[int] = []
    dna_pieces: dict[str, int] = {}

    for token_id in range(len(tokenizer)):
        piece = tokenizer.id_to_piece(token_id)
        if piece.startswith("<") and piece.endswith(">"):
            reserved_token_ids.append(token_id)
            continue
        if any(base not in base_to_index for base in piece):
            raise ValueError(f"Unexpected DNAGPT token '{piece}' outside DNA alphabet {DNAGPT_PREFIX_BASE_ORDER}.")
        dna_token_ids.append(token_id)
        dna_pieces[piece] = token_id

    if not dna_token_ids:
        raise ValueError("DNAGPT tokenizer does not contain any DNA tokens.")

    max_token_length = max(len(piece) for piece in dna_pieces)
    node_strings: list[str] = [""]
    node_index_by_piece: dict[str, int] = {"": 0}
    node_depths: list[int] = [0]
    node_exact_token_ids: list[int] = [-1]
    node_prefix_paths: list[list[int]] = [[-1] * max_token_length]
    child_rows: list[list[int]] = [[-1] * len(DNAGPT_PREFIX_BASE_ORDER)]
    nodes_by_depth: list[list[int]] = [[0]] + [[] for _ in range(max_token_length)]

    for piece, token_id in sorted(dna_pieces.items(), key=lambda item: (len(item[0]), item[0])):
        prefix = ""
        prefix_path: list[int] = [-1] * max_token_length
        for offset, base in enumerate(piece):
            prefix += base
            node_index = node_index_by_piece.get(prefix)
            if node_index is None:
                node_index = len(node_strings)
                node_index_by_piece[prefix] = node_index
                node_strings.append(prefix)
                node_depths.append(len(prefix))
                node_exact_token_ids.append(-1)
                node_prefix_paths.append([-1] * max_token_length)
                child_rows.append([-1] * len(DNAGPT_PREFIX_BASE_ORDER))
                nodes_by_depth[len(prefix)].append(node_index)

                parent_piece = prefix[:-1]
                parent_index = node_index_by_piece[parent_piece]
                child_rows[parent_index][base_to_index[base]] = node_index

            prefix_path[offset] = node_index
        node_exact_token_ids[node_index_by_piece[piece]] = token_id
        node_prefix_paths[node_index_by_piece[piece]] = prefix_path

    node_base_ids = torch.full((len(node_strings), max_token_length), -1, dtype=torch.long)
    for node_index, piece in enumerate(node_strings):
        for offset, base in enumerate(piece):
            node_base_ids[node_index, offset] = base_to_index[base]

    token_id_to_node_index = torch.full((len(tokenizer),), -1, dtype=torch.long)
    for piece, token_id in dna_pieces.items():
        token_id_to_node_index[token_id] = node_index_by_piece[piece]

    return DNAGPTPrefixTrie(
        dna_token_ids=torch.tensor(dna_token_ids, dtype=torch.long),
        reserved_token_ids=torch.tensor(reserved_token_ids, dtype=torch.long),
        token_id_to_node_index=token_id_to_node_index,
        node_exact_token_ids=torch.tensor(node_exact_token_ids, dtype=torch.long),
        node_depths=torch.tensor(node_depths, dtype=torch.long),
        node_base_ids=node_base_ids,
        node_prefix_path_indices=torch.tensor(node_prefix_paths, dtype=torch.long),
        child_indices=torch.tensor(child_rows, dtype=torch.long),
        nodes_by_depth=tuple(torch.tensor(indices, dtype=torch.long) for indices in nodes_by_depth),
        max_token_length=max_token_length,
    )


def _compute_node_probability_tables(
    log_probs: torch.Tensor,
    trie: DNAGPTPrefixTrie,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    work_log_probs = log_probs.float()
    num_rows = work_log_probs.shape[0]
    num_nodes = trie.node_depths.shape[0]
    exact_node_probs = work_log_probs.new_zeros((num_rows, num_nodes))
    exact_node_indices = trie.exact_node_indices
    if exact_node_indices.numel() > 0:
        exact_token_ids = trie.node_exact_token_ids.index_select(0, exact_node_indices)
        exact_node_probs[:, exact_node_indices] = work_log_probs.index_select(1, exact_token_ids).exp()

    subtree_probs = exact_node_probs.clone()
    for depth in range(trie.max_token_length - 1, -1, -1):
        node_indices = trie.nodes_by_depth[depth]
        if node_indices.numel() == 0:
            continue
        children = trie.child_indices.index_select(0, node_indices)
        valid_children = children >= 0
        if bool(valid_children.any().item()):
            safe_children = children.clamp_min(0)
            gathered = subtree_probs.index_select(1, safe_children.reshape(-1)).reshape(
                num_rows,
                node_indices.shape[0],
                len(DNAGPT_PREFIX_BASE_ORDER),
            )
            child_mass = (gathered * valid_children.unsqueeze(0)).sum(dim=-1)
        else:
            child_mass = subtree_probs.new_zeros((num_rows, node_indices.shape[0]))
        subtree_probs[:, node_indices] = exact_node_probs[:, node_indices] + child_mass

    if trie.reserved_token_ids.numel() > 0:
        other_mass = work_log_probs.index_select(1, trie.reserved_token_ids).exp().sum(dim=1)
    else:
        other_mass = work_log_probs.new_zeros((num_rows,))

    return exact_node_probs, subtree_probs, other_mass


def factorize_dnagpt_log_probs_to_base_prefix_stream(
    log_probs: torch.Tensor,
    target_token_ids: torch.Tensor,
    trie: DNAGPTPrefixTrie,
) -> DNAGPTPrefixFactorization:
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be a 2D tensor of shape [num_targets, vocab_size].")
    if target_token_ids.ndim != 1:
        raise ValueError("target_token_ids must be a 1D tensor.")
    if log_probs.shape[0] != target_token_ids.shape[0]:
        raise ValueError("log_probs and target_token_ids must have the same leading dimension.")

    device = log_probs.device
    target_token_ids = target_token_ids.to(device=device, dtype=torch.long)
    target_node_indices = trie.token_id_to_node_index[target_token_ids]
    if bool((target_node_indices < 0).any().item()):
        raise ValueError("DNAGPT prefix factorization only supports DNA target token ids.")

    exact_node_probs, subtree_probs, other_mass = _compute_node_probability_tables(log_probs, trie)
    target_log_probs = log_probs.float().gather(1, target_token_ids.unsqueeze(1)).squeeze(1)
    target_lengths = trie.node_depths.index_select(0, target_node_indices)
    target_base_ids = trie.node_base_ids.index_select(0, target_node_indices)
    target_prefix_nodes = trie.node_prefix_path_indices.index_select(0, target_node_indices)

    num_targets = target_token_ids.shape[0]
    max_steps = trie.max_token_length
    emitted_probabilities = log_probs.new_zeros((num_targets, max_steps, DNAGPT_PREFIX_ALPHABET_SIZE))
    emitted_symbols = torch.zeros((num_targets, max_steps), dtype=torch.long, device=device)
    emitted_valid_mask = torch.zeros((num_targets, max_steps), dtype=torch.bool, device=device)

    root_children = trie.root_child_indices.unsqueeze(0).expand(num_targets, -1)
    root_dna_mass = _gather_row_values(subtree_probs, root_children)
    root_total = other_mass + subtree_probs[:, trie.root_index]
    emitted_probabilities[:, 0, 0] = other_mass / root_total.clamp_min(1e-30)
    emitted_probabilities[:, 0, 1:] = root_dna_mass / root_total.unsqueeze(1).clamp_min(1e-30)
    emitted_symbols[:, 0] = target_base_ids[:, 0] + 1
    emitted_valid_mask[:, 0] = True

    for step in range(1, max_steps):
        continue_mask = target_lengths > step
        if bool(continue_mask.any().item()):
            continue_nodes = target_prefix_nodes[continue_mask, step - 1]
            continue_children = trie.child_indices.index_select(0, continue_nodes)
            continue_total = subtree_probs[continue_mask].gather(1, continue_nodes.unsqueeze(1)).squeeze(1)
            emitted_probabilities[continue_mask, step, 0] = (
                exact_node_probs[continue_mask].gather(1, continue_nodes.unsqueeze(1)).squeeze(1)
                / continue_total.clamp_min(1e-30)
            )
            emitted_probabilities[continue_mask, step, 1:] = (
                _gather_row_values(subtree_probs[continue_mask], continue_children)
                / continue_total.unsqueeze(1).clamp_min(1e-30)
            )
            emitted_symbols[continue_mask, step] = target_base_ids[continue_mask, step] + 1
            emitted_valid_mask[continue_mask, step] = True

        stop_mask = target_lengths == step
        if bool(stop_mask.any().item()):
            stop_nodes = target_node_indices[stop_mask]
            stop_children = trie.child_indices.index_select(0, stop_nodes)
            stop_total = subtree_probs[stop_mask].gather(1, stop_nodes.unsqueeze(1)).squeeze(1)
            emitted_probabilities[stop_mask, step, 0] = (
                exact_node_probs[stop_mask].gather(1, stop_nodes.unsqueeze(1)).squeeze(1)
                / stop_total.clamp_min(1e-30)
            )
            emitted_probabilities[stop_mask, step, 1:] = (
                _gather_row_values(subtree_probs[stop_mask], stop_children)
                / stop_total.unsqueeze(1).clamp_min(1e-30)
            )
            emitted_symbols[stop_mask, step] = 0
            emitted_valid_mask[stop_mask, step] = True

    return DNAGPTPrefixFactorization(
        emitted_probabilities=emitted_probabilities,
        emitted_symbols=emitted_symbols,
        emitted_valid_mask=emitted_valid_mask,
        target_log_probs=target_log_probs,
    )


def factorize_dnagpt_log_probs_to_grouped_prefix_stream(
    log_probs: torch.Tensor,
    target_token_ids: torch.Tensor,
    trie: DNAGPTPrefixTrie,
    merge_size: int,
) -> DNAGPTGroupedPrefixFactorization:
    if merge_size <= 0:
        raise ValueError("merge_size must be >= 1")
    if merge_size > trie.max_token_length:
        raise ValueError("merge_size must be <= trie.max_token_length")
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be a 2D tensor of shape [num_targets, vocab_size].")
    if target_token_ids.ndim != 1:
        raise ValueError("target_token_ids must be a 1D tensor.")
    if log_probs.shape[0] != target_token_ids.shape[0]:
        raise ValueError("log_probs and target_token_ids must have the same leading dimension.")

    device = log_probs.device
    base = len(DNAGPT_PREFIX_BASE_ORDER)
    work_log_probs = log_probs.float()
    target_token_ids = target_token_ids.to(device=device, dtype=torch.long)
    target_node_indices = trie.token_id_to_node_index[target_token_ids]
    if bool((target_node_indices < 0).any().item()):
        raise ValueError("DNAGPT grouped prefix factorization only supports DNA target token ids.")

    exact_node_probs, subtree_probs, other_mass = _compute_node_probability_tables(work_log_probs, trie)
    target_log_probs = work_log_probs.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)
    target_lengths = trie.node_depths.index_select(0, target_node_indices)
    target_base_ids = trie.node_base_ids.index_select(0, target_node_indices)
    target_prefix_nodes = trie.node_prefix_path_indices.index_select(0, target_node_indices)

    num_targets = target_token_ids.shape[0]
    step_probabilities: list[torch.Tensor] = []
    step_symbols: list[torch.Tensor] = []
    step_row_positions: list[torch.Tensor] = []
    active_rows = torch.arange(num_targets, device=device, dtype=torch.long)
    current_nodes = torch.full((num_targets,), trie.root_index, dtype=torch.long, device=device)
    current_depth = 0

    while active_rows.numel() > 0:
        max_remaining = trie.max_token_length - current_depth
        term_max_length = min(merge_size, max_remaining)
        include_other = current_depth == 0
        include_cont = max_remaining > merge_size
        active_exact = exact_node_probs.index_select(0, active_rows)
        active_subtree = subtree_probs.index_select(0, active_rows)
        active_nodes = current_nodes.index_select(0, active_rows)
        active_target_lengths = target_lengths.index_select(0, active_rows)
        active_target_base_ids = target_base_ids.index_select(0, active_rows)
        active_prefix_nodes = target_prefix_nodes.index_select(0, active_rows)

        descendant_nodes_by_length: list[torch.Tensor] = []
        frontier = active_nodes.unsqueeze(1)
        for _ in range(term_max_length):
            frontier = trie.child_indices.index_select(0, frontier.reshape(-1)).reshape(frontier.shape[0], -1)
            descendant_nodes_by_length.append(frontier)

        step_width = (1 if include_other else 0) + sum(base ** length for length in range(1, term_max_length + 1))
        if include_cont:
            step_width += base ** merge_size
        probabilities = work_log_probs.new_zeros((active_rows.shape[0], step_width))

        if include_other:
            probabilities[:, 0] = other_mass.index_select(0, active_rows)
            denominator = probabilities[:, 0] + active_subtree[:, trie.root_index]
            offset = 1
        else:
            denominator = (
                active_subtree.gather(1, active_nodes.unsqueeze(1)).squeeze(1)
                - active_exact.gather(1, active_nodes.unsqueeze(1)).squeeze(1)
            )
            offset = 0

        for length, descendant_nodes in enumerate(descendant_nodes_by_length, start=1):
            block_width = base ** length
            probabilities[:, offset : offset + block_width] = _gather_row_values(active_exact, descendant_nodes)
            offset += block_width

        if include_cont:
            descendant_nodes = descendant_nodes_by_length[merge_size - 1]
            probabilities[:, offset : offset + (base ** merge_size)] = (
                _gather_row_values(active_subtree, descendant_nodes)
                - _gather_row_values(active_exact, descendant_nodes)
            )

        probabilities = probabilities / denominator.unsqueeze(1).clamp_min(1e-30)

        remaining_lengths = active_target_lengths - current_depth
        symbols = torch.zeros((active_rows.shape[0],), dtype=torch.long, device=device)
        next_active_mask = remaining_lengths > merge_size
        for term_length in range(1, term_max_length + 1):
            term_mask = remaining_lengths == term_length
            if not bool(term_mask.any().item()):
                continue
            chunk_ids = _compute_chunk_ids(
                active_target_base_ids[term_mask, current_depth : current_depth + term_length],
                chunk_length=term_length,
                base=base,
            )
            symbols[term_mask] = _term_block_offset(
                include_other=include_other,
                term_length=term_length,
                base=base,
            ) + chunk_ids

        if include_cont and bool(next_active_mask.any().item()):
            chunk_ids = _compute_chunk_ids(
                active_target_base_ids[next_active_mask, current_depth : current_depth + merge_size],
                chunk_length=merge_size,
                base=base,
            )
            cont_offset = (1 if include_other else 0) + sum(base ** length for length in range(1, term_max_length + 1))
            symbols[next_active_mask] = cont_offset + chunk_ids

        row_positions = torch.full((num_targets,), -1, dtype=torch.long, device=device)
        row_positions[active_rows] = torch.arange(active_rows.shape[0], dtype=torch.long, device=device)
        step_probabilities.append(probabilities)
        step_symbols.append(symbols)
        step_row_positions.append(row_positions)

        if not bool(next_active_mask.any().item()):
            break

        current_depth += merge_size
        active_rows = active_rows[next_active_mask]
        current_nodes = torch.full((num_targets,), trie.root_index, dtype=torch.long, device=device)
        current_nodes[active_rows] = active_prefix_nodes[next_active_mask, current_depth - 1]

    return DNAGPTGroupedPrefixFactorization(
        step_probabilities=tuple(step_probabilities),
        step_symbols=tuple(step_symbols),
        step_row_positions=tuple(step_row_positions),
        target_log_probs=target_log_probs,
    )
