"""Run GenCoder reproduction experiments.

Quick smoke run, including train + compress + exact reconstruction:

    python scripts\\run_gencoder_experiment.py \
        --config configs\\dna_gencoder_smoke.json \
        --mode all

Run the full DNACorpus reproduction one group at a time:

    python scripts\\run_gencoder_experiment.py \
        --config configs\\dna_gencoder_dnacorpus.json \
        --mode all \
        --run eukaryotic_dnacorpus

    python scripts\\run_gencoder_experiment.py \
        --config configs\\dna_gencoder_dnacorpus.json \
        --mode all \
        --run prokaryotic_dnacorpus

    python scripts\\run_gencoder_experiment.py \
        --config configs\\dna_gencoder_dnacorpus.json \
        --mode all \
        --run hosa_learning_curve

Train first, then compress later from the saved checkpoint:

    python scripts\\run_gencoder_experiment.py \
        --config configs\\dna_gencoder_dnacorpus.json \
        --mode train \
        --run eukaryotic_dnacorpus

    python scripts\\run_gencoder_experiment.py \
        --config configs\\dna_gencoder_dnacorpus.json \
        --mode compress \
        --run eukaryotic_dnacorpus

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.gencoder import load_gencoder_config, run_gencoder_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GenCoder reproduction experiments on DNACorpus.")
    parser.add_argument("--config", required=True, help="Path to a GenCoder JSON config.")
    parser.add_argument(
        "--mode",
        choices=["train", "compress", "all", "decompress-check"],
        default="all",
        help="Run training, compression, both, or validate saved compressed artifacts.",
    )
    parser.add_argument("--run", help="Run only the named config entry.")
    parser.add_argument("--output-dir", help="Override config output_dir.")
    parser.add_argument("--device", help="Override config device.")
    parser.add_argument("--epochs", type=int, help="Override epochs for selected runs.")
    parser.add_argument("--max-bytes-per-source", type=int, help="Override per-source byte cap.")
    parser.add_argument("--no-save-artifacts", action="store_true", help="Skip writing latent/residual artifacts.")
    args = parser.parse_args()

    config = load_gencoder_config(args.config)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.device = args.device
    for run_config in config.runs:
        if args.run is not None and run_config.name != args.run:
            continue
        if args.epochs is not None:
            run_config.epochs = args.epochs
        if args.max_bytes_per_source is not None:
            run_config.max_bytes_per_source = args.max_bytes_per_source
        if args.no_save_artifacts:
            run_config.save_artifacts = False

    results = run_gencoder_config(config, mode=args.mode, run_name=args.run)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
