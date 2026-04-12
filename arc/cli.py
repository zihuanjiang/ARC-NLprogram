"""
Command-line interface for running ARC experiments.
Handles Hydra configuration and delegates to run_experiment.
"""
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

from arc.run import run_experiment

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "config")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_experiment(cfg_dict)


if __name__ == "__main__":
    main()
