#!/usr/bin/env python
"""Command-line entry point for the MRvF simulation pipeline.

    python scripts/run_pipeline.py train    --variant both            # train the triple-regime models
    python scripts/run_pipeline.py evaluate --stats                   # RMSE vs SNR for DM / DL-4p / Triple
    python scripts/run_pipeline.py all                                # train both, then evaluate them

Data locations come from ``configs/default.toml`` and can be overridden with --data-dir / --echotimes.
Trained models and results go to ``--output-dir`` (default ``runs/``), never over the shipped ``results/``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # allows running without `pip install -e .`

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from mrvf.config import load_config, with_overrides  # noqa: E402
from mrvf.pipeline import compare_results, run_evaluation, run_training  # noqa: E402
from mrvf.training import get_device  # noqa: E402

SHIPPED = {
    "DL-4p": ROOT / "results/t2snr_results_v4/models/t2snr_noisy_4param_v4.pt",
    "Triple-NF": ROOT / "results/triple_regime_nf_results_v1/models/triple_nf_best.pt",
    "Triple-Noisy": ROOT / "results/triple_regime_results_v1/models/triple_regime_best.pt",
}
SHIPPED_REFERENCE = ROOT / "results/ablation_results/ablation_rmse.json"
FOLDER = {"noisy": "triple_regime_noisy", "noise-free": "triple_regime_noise_free"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=ROOT / "configs/default.toml", help="TOML config file")
    common.add_argument("--data-dir", type=Path, help="folder with the QuasiRand_*.mat dictionaries")
    common.add_argument("--echotimes", type=Path, help="path to echotimes.mat")
    common.add_argument("--output-dir", type=Path, default=ROOT / "runs", help="where to write results (default: runs/)")
    common.add_argument("--device", choices=["cpu", "cuda"], help="default: cuda if available")
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = p.add_subparsers(dest="command", required=True)

    def add_train_args(sp):
        sp.add_argument("--variant", choices=["noisy", "noise-free", "both"], default="both")
        sp.add_argument("--epochs", type=int, help="override max epochs (config default: 200)")
        sp.add_argument("--n-samples", type=int, help="override the mixed-SNR training-set size")
        sp.add_argument("--snr-levels", type=int, nargs="+", help="override the SNR levels")
        sp.add_argument("--seed", type=int, help="override the torch/subsample seed")
        sp.add_argument("--no-figures", action="store_true")

    def add_eval_args(sp):
        sp.add_argument("--dl4p", type=Path, help="DL-4p checkpoint (default: shipped)")
        sp.add_argument("--triple-nf", type=Path, help="Triple-NF checkpoint (default: shipped)")
        sp.add_argument("--triple-noisy", type=Path, help="Triple-Noisy checkpoint (default: shipped)")
        sp.add_argument("--stats", action="store_true", help="also compute paired Wilcoxon tests on |error|")
        sp.add_argument("--compare-to", type=Path, help="reference ablation_rmse.json to diff against")
        sp.add_argument("--no-figures", action="store_true")

    add_train_args(sub.add_parser("train", parents=[common], help="train triple-regime model(s)"))
    add_eval_args(sub.add_parser("evaluate", parents=[common], help="RMSE vs SNR for all methods"))
    sp_all = sub.add_parser("all", parents=[common], help="train both variants, then evaluate them")
    add_train_args(sp_all)
    sp_all.add_argument("--stats", action="store_true")
    sp_all.add_argument("--compare-to", type=Path)
    return p


def make_config(args):
    from dataclasses import replace

    cfg = load_config(args.config if args.config.exists() else None)
    # Paths written in the config file are relative to the repository root, so the script works from any directory.
    # Paths given on the command line are relative to where you ran the command, as usual.
    paths = replace(
        cfg.paths,
        dict_dir=(ROOT / cfg.paths.dict_dir).resolve(),
        echotimes_file=(ROOT / cfg.paths.echotimes_file).resolve(),
    )
    if args.data_dir:
        paths = replace(paths, dict_dir=args.data_dir.resolve())
    if args.echotimes:
        paths = replace(paths, echotimes_file=args.echotimes.resolve())
    cfg = replace(cfg, paths=paths)
    over = {}
    for arg, field in (("epochs", "epochs"), ("n_samples", "n_samples"), ("seed", "torch_seed")):
        if getattr(args, arg, None) is not None:
            over[field] = getattr(args, arg)
    if getattr(args, "seed", None) is not None:
        over["subsample_seed"] = args.seed
    if getattr(args, "snr_levels", None):
        over["snr_levels"] = tuple(args.snr_levels)
    return with_overrides(cfg, **over)


def do_train(args, cfg, device):
    variants = ["noisy", "noise-free"] if args.variant == "both" else [args.variant]
    trained = {}
    for v in variants:
        out = args.output_dir / FOLDER[v]
        logging.info("=== training %s -> %s ===", v, out)
        run_training(cfg, v, out, device, make_figures=not args.no_figures)
        stem = "triple_regime" if v == "noisy" else "triple_nf"
        trained["Triple-Noisy" if v == "noisy" else "Triple-NF"] = out / "models" / f"{stem}_best.pt"
    return trained


def do_evaluate(args, cfg, device, checkpoints):
    out = args.output_dir / "evaluation"
    logging.info("=== evaluating %s -> %s ===", ", ".join(checkpoints), out)
    res = run_evaluation(cfg, checkpoints, out, device, with_stats=args.stats, make_figures=not getattr(args, "no_figures", False))
    if args.compare_to:
        diff = compare_results(res["results"], args.compare_to)
        logging.info("vs %s: %d values, max |diff| = %.3g, max relative = %.3g%%",
                     args.compare_to, diff["n_values"], diff["max_abs_diff"], diff["max_rel_diff"] * 100)
    return res


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = make_config(args)
    device = get_device(args.device)
    logging.info("device: %s", device)

    if args.command == "train":
        do_train(args, cfg, device)
    elif args.command == "evaluate":
        ckpts = {
            "DL-4p": args.dl4p or SHIPPED["DL-4p"],
            "Triple-NF": args.triple_nf or SHIPPED["Triple-NF"],
            "Triple-Noisy": args.triple_noisy or SHIPPED["Triple-Noisy"],
        }
        missing = [f"{k}: {v}" for k, v in ckpts.items() if not Path(v).exists()]
        if missing:
            logging.error("Checkpoint(s) not found:\n  %s", "\n  ".join(missing))
            return 2
        do_evaluate(args, cfg, device, ckpts)
    elif args.command == "all":
        trained = do_train(args, cfg, device)
        ckpts = {"DL-4p": SHIPPED["DL-4p"], **trained}  # DL-4p is not retrained by this CLI
        if not Path(ckpts["DL-4p"]).exists():
            ckpts.pop("DL-4p")
        do_evaluate(args, cfg, device, ckpts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
