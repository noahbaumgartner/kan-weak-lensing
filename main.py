import math
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that "modules" and "src" are importable
# regardless of the working directory Hydra switches to.
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.modules.reduction import reduced_dim


def _make_width(in_d, out_d, n_h, *hidden_widths):
    n_h = int(n_h)
    if n_h > len(hidden_widths):
        raise ValueError(
            f"make_width: n_hidden_layers={n_h} exceeds provided widths ({len(hidden_widths)})"
        )
    return [int(in_d)] + [int(w) for w in hidden_widths[:n_h]] + [int(out_d)]


OmegaConf.register_new_resolver("make_width", _make_width, replace=True)
OmegaConf.register_new_resolver("reduced_dim", reduced_dim, replace=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    trainer = instantiate(cfg.trainer, cfg=cfg, _recursive_=False)
    results = trainer.train()
    final_loss = float(results["test_loss"][-1])
    return final_loss if math.isfinite(final_loss) else 1e10


if __name__ == "__main__":
    main()
