from typing import Optional
import torch

from .scorer import NuggetScorer
from .adaptors.score_feeder import NuggetScoreFeeder


def _resolve_adaptor(model):
    model_type = model.config.model_type
    if model_type in ['bert']:
        from .adaptors.bert import adapt_bert

        return adapt_bert, model.config.d_model
    if model_type in ['bart', 'mbart']:
        from .adaptors.bart import adapt_bart

        return adapt_bart, model.config.d_model
    if model_type in ['t5']:
        from .adaptors.t5 import adapt_t5

        return adapt_t5, model.config.d_model
    if model_type in ['llama']:
        from .adaptors.llama import adapt_llama

        return adapt_llama, model.config.hidden_size
    raise NotImplementedError


def nuggify(
        model, scorer_layer: int = 3, residual_start: int = 0, residual_end: int = -1,
        value_ffn: bool = True, straight_through: bool = True, ratio: Optional[float] = None,
):
    """
    :param model: A base Huggingface/Transformer model
    :param scorer_layer: The scorer takes the features of the `scorer_layer`-th layer in transformers.
    :param residual_start: See below.
    :param residual_end: The residual connections will be built from encoder to
    layers[residual_start:residual_end] in decoder.
    :param value_ffn: Append a value FFN to the nugget encodings.
    :param straight_through: If True, will subtract the score value from the forward pass; otherwise
    :param ratio: Nugget ratio.
    the nugget score could affect the forward pass.
    """
    residual_end = residual_end if residual_end > 0 else model.config.num_hidden_layers

    adapt_fn, hidden_size = _resolve_adaptor(model)
    feeder = NuggetScoreFeeder(straight_through, enable=True)
    feeder, scorer_feat, encoder, decoder = adapt_fn(feeder, model, scorer_layer, residual_start, residual_end)
    scorer = NuggetScorer(scorer_feat, hidden_size, feeder, value_ffn, ratio)
    nugget_kwargs = dict(
        scorer_layer=scorer_layer, residual_start=residual_start, residual_end=residual_end,
        value_ffn=value_ffn, straight_through=straight_through,
        ratio=ratio,
    )
    return scorer, encoder, decoder, nugget_kwargs


def save_nuggets(scorer: NuggetScorer, save_path: str, nugget_kwargs):
    states = {'non_linear': scorer.non_linear.state_dict()}
    if scorer.value_ffn is not None:
        states['value_ffn'] = scorer.value_ffn.state_dict()
    dump = {'scorer_states': states, 'kwargs': nugget_kwargs}
    torch.save(dump, save_path)


def load_nuggets(model, save_path: str):
    loaded = torch.load(save_path, map_location='cpu')
    scorer, encoder, decoder, nugget_kwargs = nuggify(model, **loaded['kwargs'])
    scorer.non_linear.load_state_dict(loaded['scorer_states']['non_linear'])
    if scorer.value_ffn is not None:
        scorer.value_ffn.load_state_dict(loaded['scorer_states']['value_ffn'])
    return scorer, encoder, decoder, nugget_kwargs
