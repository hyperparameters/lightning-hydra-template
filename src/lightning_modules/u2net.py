import pytorch_lightning as pl
from typing import Any, List
import torch
from ..utils import tensor2im
from torchvision.utils import make_grid
import wandb


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def create_visuals(inputs, labels, pred, max_images=5):
    pred = normPRED(pred)

    if inputs.shape[0] > 1:
        input_grid = make_grid(inputs[:max_images], 5)
        pred_grid = make_grid(pred[:max_images], 5)
        gt_grid = make_grid(labels[:max_images], 5)

        input_img = tensor2im(input_grid)
        pred_img = tensor2im(pred_grid, rescale=False, keep_alpha_channel=True)
        gt_img = tensor2im(gt_grid, rescale=False, keep_alpha_channel=True)
    else:
        input_img = tensor2im(inputs[0])
        pred_img = tensor2im(pred[0], rescale=False, keep_alpha_channel=True)
        gt_img = tensor2im(labels[0], rescale=False, keep_alpha_channel=True)

    return input_img, gt_img, pred_img


class U2NetModule(pl.LightningModule):
    def __init__(
        self, model, loss, optimizer, metrics, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = model
        self.optimizer_wrapper = optimizer
        self.loss = loss
        self.metrics = metrics

    def configure_optimizers(self):
        optimizer = self.optimizer_wrapper(params=self.model.parameters())
        return optimizer

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        bce_loss = self.loss["bce_loss"]
        loss0 = bce_loss(d0, labels_v)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (loss0.data.item(), loss1.data.item(
        # ), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

        return loss0, loss

    def matting_losses(self, labels, preds, true_bg=None):
        alpha_loss = self.loss["alpha_loss"]
        alpha_window_loss = self.loss["alpha_window_loss"]
        composition_loss = self.loss["composition_loss"]
        rgb_loss = self.loss["rgb_loss"]

        pred_alpha = preds[:, -1:, :, :]
        pred_rgb = preds[:, :-1, :, :]

        true_alpha = labels[:, -1:, :, :]
        true_rgb = labels[:, :-1, :, :]

        _a_loss = alpha_loss(true_alpha, pred_alpha)

        window_mask = ((true_alpha > 0) & (true_alpha < 1)).float()
        _a_window_loss = alpha_window_loss(true_alpha, pred_alpha, window_mask)

        _c_loss = composition_loss(true_alpha, pred_alpha, true_rgb, true_bg)

        _rgb_loss = rgb_loss(true_rgb, pred_rgb, true_alpha)
        _rgb_window_loss = rgb_loss(true_rgb, pred_rgb, window_mask)

        return {
            "alpha_loss": _a_loss,
            "alpha_window_loss": _a_window_loss,
            "composition_loss": _c_loss,
            "rgb_loss": _rgb_loss,
            "rgb_window_loss": _rgb_window_loss,
        }

    def compute_loss(self, d0, d1, d2, d3, d4, d5, d6, label, true_bg=None):
        # u2net loss
        _, bce_loss = self.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
        losses = self.matting_losses(label, d1, true_bg)
        losses["bce_loss"] = bce_loss
        return losses

    def step(self, batch: Any):
        inputs, labels, true_bg = batch["image"], batch["label"], batch.get("bg", None)

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        true_bg = true_bg.type(torch.FloatTensor) if true_bg is not None else true_bg

        # # wrap them in Variable
        # if torch.cuda.is_available():
        #     inputs_v, labels_v = Variable(inputs.to(config.device), requires_grad=False), Variable(labels.to(config.device),
        #                                                                                            requires_grad=False)
        #     true_bg_v = Variable(true_bg.to(config.device),
        #                          requires_grad=False) if true_bg is not None else true_bg
        # else:
        #     inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(
        #         labels, requires_grad=False)
        #     true_bg_v = Variable(
        #         true_bg, requires_grad=False) if true_bg is not None else true_bg

        d0, d1, d2, d3, d4, d5, d6 = self.forward(inputs)
        losses = self.compute_loss(d0, d1, d2, d3, d4, d5, d6, labels, true_bg)
        # loss2, loss = muti_bce_loss_fusion(
        #     d0, d1, d2, d3, d4, d5, d6, labels_v)

        # losses = matting_losses(
        #     inputs_v, labels_v, d1, true_bg_v)
        return losses, d1, labels, inputs

    def training_step(self, batch: Any, batch_idx: int):
        losses, preds, targets, inputs = self.step(batch)

        loss_log = {}
        final_loss = 0
        for loss_name in losses.keys():
            weight = self.loss["loss_weight"][loss_name]
            loss_value = losses[loss_name]
            final_loss += weight * loss_value
            loss_log["train_" + loss_name] = loss_value.detach().cpu()

        loss_log["train_loss"] = final_loss.detach().cpu()

        # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict(loss_log, on_step=True, on_epoch=True, prog_bar=True)

        # log metrics
        # for name, metric in self.metrics.train.items():
        #     value = metric(preds, targets)
        #     self.log(
        #         f"train_{name}", value, on_step=False, on_epoch=True, prog_bar=True
        #     )
        return {"loss": final_loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):

        losses, preds, targets, inputs = self.step(batch)

        loss_log = {}
        final_loss = 0
        for loss_name in losses.keys():
            weight = self.loss["loss_weight"][loss_name]
            loss_value = losses[loss_name]
            final_loss += weight * loss_value
            loss_log["val_" + loss_name] = loss_value.detach().cpu()

        loss_log["val_loss"] = final_loss.detach().cpu()

        self.log_dict(
            loss_log, on_step=False, on_epoch=True, prog_bar=True, reduce_fx="mean"
        )

        # log val metrics
        # for name, metric in self.metrics.val.items():
        #     value = metric(preds, targets)
        #     self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": final_loss,
            "preds": preds,
            "targets": targets,
            "inputs": inputs,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        # log images
        val_rgba_images = []
        val_alpha_images = []

        for output in outputs:
            pred = output["preds"]
            labels = output["targets"]
            inputs = output["inputs"]

            input_img, label_img, pred_img = create_visuals(
                inputs.detach().cpu(), labels.detach().cpu(), pred.detach().cpu()
            )
            val_rgba_images.append(
                [wandb.Image(input_img), wandb.Image(label_img), wandb.Image(pred_img)]
            )

            val_alpha_images.append(
                [
                    wandb.Image(input_img),
                    wandb.Image(label_img[:, :, -1]),
                    wandb.Image(pred_img[:, :, -1]),
                ]
            )

        val_alpha_table = wandb.Table(
            columns=["Input Image", "GT mask", "Pred Mask"], data=val_alpha_images
        )
        val_rgba_table = wandb.Table(
            columns=["Input Image", "GT rgba", "Pred rgba"], data=val_rgba_images
        )
        wandb.log({"val mask": val_alpha_table, "val rgba": val_rgba_table})

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        for name, metric in self.metrics.train.items():
            metric.reset()
        for name, metric in self.metrics.val.items():
            metric.reset()
