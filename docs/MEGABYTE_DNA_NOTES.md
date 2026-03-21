# Megabyte for DNA Compression

## Core idea

MEGABYTE is a byte-level autoregressive Transformer. It avoids external tokenization and models raw bytes directly, which fits DNA text files naturally because the source is already a character stream.

The architecture is split into two scales:

1. Global model
   - Split the sequence into fixed-size patches of `P` bytes.
   - Shift each patch by one byte/patch worth of context so the model remains autoregressive.
   - Encode each patch into a higher-level patch representation.

2. Local model
   - Use the global patch representation as coarse context.
   - Predict each byte inside the patch autoregressively with a smaller local Transformer.

This reduces the cost of modeling long byte streams compared with a single full-resolution Transformer, while still preserving byte-level likelihoods needed by arithmetic coding.

## Why it matches DNA compression

- The DNACorpus files are plain-text nucleotide sequences, so a byte model can be trained directly.
- DNA has a tiny visible alphabet (`A/C/G/T` here), but long-range structure still matters. The global-local split is a reasonable compromise between local motif modeling and longer genomic regularity.
- Compression only needs next-symbol probabilities. MEGABYTE produces those probabilities directly, which can be fed into arithmetic coding.

## This project's implementation choices

- Upstream implementation source: `third_party/shjwudp_megabyte`
- Default model path: `model/megabyte.py`
- Vocabulary:
  - `0..255` for raw byte values
  - `257` for `pad`
  - `258` for `eos`
- Training target:
  - language modeling loss on raw bytes
- Compression target:
  - sequential arithmetic coding of test bytes plus one `eos`

## Current initial experiment

- Species: `HoSa`
- Train / val / test slices:
  - 2 MB / 256 KB / 256 KB
- Sequence length: `512`
- Patch size: `8`
- Model size: about `11.2M` parameters
- Best test loss so far: about `1.95 bits/byte`

## How to run

Train + evaluate:

```powershell
D:\MLLMs\.venv\Scripts\python.exe scripts\run_dna_experiment.py --config configs\dna_megabyte_hosa_initial.json --mode all
```

Only re-run evaluation/compression from the saved checkpoint:

```powershell
D:\MLLMs\.venv\Scripts\python.exe scripts\run_dna_experiment.py --config configs\dna_megabyte_hosa_initial.json --mode eval
```
