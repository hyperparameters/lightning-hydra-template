from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="demo", config_name="comp")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))


@hydra.main(config_path="demo", config_name="comp")
def main2(cfg):
    print(OmegaConf.to_yaml(cfg))
    train = instantiate(cfg.train)
    dataset = instantiate(cfg.dataset)

    print(type(train))
    print(type(dataset))


if __name__ == "__main__":
    main()