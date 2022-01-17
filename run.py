import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    lightning_module: pl.LightningModule = instantiate(cfg.lightning_module)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    trainer.fit(lightning_module, datamodule=datamodule)

if __name__ == "__main__":
    run()
