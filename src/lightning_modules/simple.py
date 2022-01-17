from turtle import forward
import pytorch_lightning as pl
from typing import Any, List
import torch


class SimpleLitModule(pl.LightningModule):
    def __init__(self,model, loss, optimizer, metrics, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        self.model = model
        self.optimizer_wrapper = optimizer
        self.loss = loss
        self.metrics = metrics
    
    def configure_optimizers(self):
        optimizer = self.optimizer_wrapper(params = self.model.parameters())
        return optimizer

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        #log metrics
        for name, metric in self.metrics.train.items():
            value = metric(preds, targets)
            self.log(f"train/{name}", value, on_step=False, on_epoch=True, prog_bar=True)

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # log val metrics
        for name, metric in self.metrics.val.items():
            value = metric(preds, targets)
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        for name, metric in self.metrics.train.items():
            metric.reset()
        for name, metric in self.metrics.val.items():
            metric.reset()
        
