"""Train and evaluate online RAVEN.

Examples:
    python scripts/train_online.py --config configs/online/mpe.yaml --task spread --seed 1
    python scripts/train_online.py --config configs/online/smac.yaml --task MMM2 --seed 1 --device cuda
    python scripts/train_online.py --config configs/online/navigation.yaml --task ring-3 --seed 1
    python scripts/train_online.py --config configs/online/mpe.yaml --task spread --set model.variant=no_comm
"""

import argparse
from pathlib import Path

from raven.online import train_online
from raven.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default=None, help="task name, e.g. MMM2, spread or ring-3")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="output directory (default: runs/<task>_<variant>_s<seed>)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. training.t_max=200000")
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    if args.task is not None:
        config["task"]["name"] = args.task
    if args.seed is not None:
        config["seed"] = args.seed
    variant = config.get("model", {}).get("variant", "full")
    out = args.out or Path("runs") / f"{config['task']['name']}_{variant}_s{config['seed']}"
    train_online(config, out, device=args.device)


if __name__ == "__main__":
    main()
