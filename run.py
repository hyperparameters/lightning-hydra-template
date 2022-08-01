import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate, get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import sys
from pytorch_lightning.loggers import WandbLogger
import wandb


@hydra.main(config_path="config", config_name="config")
def run(cfg: DictConfig):
    sys.path.append(get_original_cwd())
    print(OmegaConf.to_yaml(cfg))
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    lightning_module: pl.LightningModule = instantiate(cfg.lightning_module)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            # logger.experiment.config.update(cfg)
            wandb.config.update(cfg)
            wandb.save(".hydra/*.yaml")
    trainer.fit(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    run()
