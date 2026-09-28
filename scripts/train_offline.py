"""Train and evaluate offline RAVEN on ring navigation.

Example:
    python scripts/train_offline.py --config configs/offline/ring_c2.yaml --seed 1
"""

import argparse
from pathlib import Path

from raven.offline import run_offline
from raven.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="overrides the seed in the config")
    parser.add_argument("--out", default=None, help="output directory (default: runs/<config>_s<seed>)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. teacher.steps=200000")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    if args.seed is not None:
        config["seed"] = args.seed
    out = args.out or Path("runs") / f"{Path(args.config).stem}_s{config['seed']}"
    run_offline(config, out, device=args.device)


if __name__ == "__main__":
    main()
