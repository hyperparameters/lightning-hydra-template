from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="demo", config_name="badic")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
