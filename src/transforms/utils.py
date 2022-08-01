import functools
from scipy.ndimage import distance_transform_edt as distance
import random
import os
from PIL import Image
import torch
import cv2 as cv

# import kornia
import torchvision.transforms as T
import matplotlib
from torch import einsum
import torch
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision
import math
import wandb
from os import path
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision
import math
import cv2
import shutil
import time
import glob

mean = np.array([0.485, 0.456, 0.406])[None, None, :]
std = np.array([0.229, 0.224, 0.225])[None, None, :]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# import wandb
# def update_wandb_logs(metric_names, metric_vals):
#     assert len(metric_names)==len(metric_vals)
#     dict={}
#     for i in range(len(metric_names)):

#         try:dict[metric_names[i]]=metric_vals[i].item()
#         except:dict[metric_names[i]]=metric_vals[i]
#     wandb.log(dict)


def accuracy(predicted, target):  # returns non-mean acc
    preds = torch.argmax(predicted, dim=1)
    return (torch.sum(preds == target).item() + 0.0001) / target.size(0)


def r(x):
    return str(round(x, 4))


def exp_ma_debiased(losses, pct=0.9):
    n = len(losses)
    if n == 1:
        return losses[0]
    gamma = np.array([pct])
    exps = gamma ** np.array(list(range(1, n))[::-1])
    if n > 2:
        exps[-1] = exps[-1] * (1 - pct)
    pcts = np.concatenate([exps, [0.1]], 0)
    debiased = pcts / np.sum(pcts)
    return r(np.sum(debiased * losses))


def ema(losses, pct=0.9):
    return exp_ma_debiased(losses, pct)


def convert_old_to_newModule(model, layer_type_old, layer_to_convert):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_old_to_newModule(
                module, layer_type_old, layer_to_convert
            )

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = layer_to_convert
            model._modules[name] = layer_new
    return model


def freeze_unfreeze_BN(
    model, bn_type=torch.nn.modules.batchnorm.BatchNorm2d, freeze=False
):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = freeze_unfreeze_BN(
                module, bn_type=bn_type, freeze=freeze
            )
        if type(module) == bn_type:
            for param in module.parameters():
                param.requires_grad = not freeze
    return model


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish_jit(x, inplace=False):
    # inplace ignored
    return MishJitAutoFn.apply(x)


class MishJit(nn.Module):
    def __init__(self, inplace=False):
        super(MishJit, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishJitAutoFn.apply(x)


def dice_loss_classes(inpu, target):  # [bs,2,h,w]

    ip = inpu[:, 0, :, :].contiguous().view(-1)
    tar = target[:, 0, :, :].contiguous().view(-1)
    intersection = (ip * tar).sum()
    union = ip.sum() + tar.sum()
    score1 = 1 - 2 * (intersection / union)

    ip = inpu[:, 1, :, :].contiguous().view(-1)
    tar = target[:, 1, :, :].contiguous().view(-1)
    intersection = (ip * tar).sum()
    union = ip.sum() + tar.sum()
    score2 = 1 - 2 * (intersection / union)

    return score1, score2


def one_hot_my(a):
    g1 = a[:, :] == 0
    g2 = a[:, :] == 1
    return torch.stack((g1, g2), 0)


def one_hot_gpu(a):
    g1 = a[:, :] == 0
    g2 = a[:, :] == 1
    return torch.stack((g1, g2), 1)


soft_max = torch.nn.Softmax(dim=1)


# ----------------------------------------------------------------LRFinder
class LRFinder(object):
    def __init__(
        self, model, dataloader, opt, lr_max=1, lr_min=1e-7, pct_epoch=1, mom=0.9
    ):
        self.model = model
        self.dataloader = dataloader
        self.opt = opt
        self.lr_max = lr_max = 1
        self.lr_min = lr_min
        self.pct_epoch = pct_epoch
        self.mom = mom

        self.total_steps = int(len(dataloader) * pct_epoch) + 0.00001  # total steps
        self.mult = (lr_max / lr_min) ** (1 / (self.total_steps - 1))

    def find(self, return_lists=False):
        averaged_loss = 0
        losses = []
        log_lrs = []
        lr = self.lr_min
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        best_loss = 0.0
        for idx, data in enumerate(self.dataloader):
            if type(data) != dict:
                # get data according to user-defined get_data
                data = self.get_data(list(data))
            else:
                data = self.get_data(data)
            x = data[0]
            targets = data[1:]

            pred = self.model(x)
            # cal loss according to user defined loss_func
            loss = self.get_loss(pred, targets)
            if idx != 0:
                averaged_loss = (
                    self.mom * averaged_loss + (1 - self.mom) * loss.detach().cpu()
                )
            else:
                averaged_loss = loss.detach().cpu()
            smooth_loss = averaged_loss / (1 - self.mom ** (idx + 1))

            losses.append(averaged_loss.item())
            log_lrs.append(math.log10(lr))
            if idx == 0 or smooth_loss < best_loss:
                best_loss = smooth_loss
            if idx > 1 and smooth_loss > 4 * best_loss:
                return (losses, log_lrs)

            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            lr *= self.mult
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
            if idx > self.total_steps:
                break

        self.losses = losses
        self.log_lrs = log_lrs
        if return_lists:
            return (losses, log_lrs)

    def plot_lr(self):
        plt.plot(self.log_lrs[10:-5], self.losses[10:-5])
        plt.show()

    def recommend(self, plot=False):
        logs = np.array(self.log_lrs)
        losses = np.array(self.losses)
        deltaX = logs[:-1] - logs[1:]
        deltaY = losses[:-1] - losses[1:]
        grad = deltaY / deltaX
        max_index = np.argmin(grad)
        max_lr = logs[max_index]  # add overestimating heuristic backer
        print("Recommened Lr is: ", 10**max_lr)
        if plot:
            plt.plot(self.log_lrs[10:-5], self.losses[10:-5])
            plt.axvline(x=max_lr)
            plt.show()
            return 10**max_lr
        else:
            return 10**max_lr


# ------------------------------------------------------------------------SurfaceLoss

# All surface loss functions/Modified for binary seg
def p(*args):
    for x in args:
        plt.imshow(x)
        plt.show()


def uniq(a):
    return set(torch.unique(a.cpu()).numpy())


def sset(a, sub):
    return uniq(a).issubset(sub)


def one_hot(t, axis=1):
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg, C):
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    b, w, h = seg.shape  # type: Tuple[int, int, int]
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    #     assert one_hot(res)
    return res


def simplex(t, axis=1):
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot2dist(seg, normalise):
    # meant for binary seg, outputs mask to be dot producted with class 1 output
    # SurfaceLoss modified to: dc = dist_maps.type(torch.float32)
    # returns one channel mask that it -ive on class 1 region and +ve on class 0(np.abs for viewing)
    # the final loss is a dot product with a p-map which we try to minimize
    # hence class 0 regions will have min when their p->0 and class 1 regions when p->1

    # in internal normlaisation, fg and bg are normalised seperately so range is -1 to 1

    assert one_hot(torch.Tensor(seg), axis=0)
    C = len(seg)
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask

            x = distance(posmask)
            y = distance(negmask)

            if normalise:
                x = x / np.max(x)
                y = y / np.max(y)
                x[x > 0] = 1 - x[x > 0]
                y[y > 0] = 1 - y[y > 0]
            else:
                x[x > 0] = np.max(x) - x[x > 0]
                y[y > 0] = np.max(y) - y[y > 0]
            return x - y


class SurfaceLoss:
    def __init__(self, **kwargs):
        # modified to take dist_maps from modified one_hot2dist
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs["idc"]
        print(self.__class__.__name__, kwargs)

    def __call__(self, probs, dist_maps):
        assert simplex(probs)
        assert not one_hot(dist_maps)
        #         print(probs.size(), dist_maps.size())
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps.type(torch.float32).unsqueeze(1)
        #         print('2',pc.size(), dc.size())

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        #         if torch.isnan(loss):
        #             l=[pc,dc,multipled,loss]
        #             for e in l: print('_',torch.min(e),torch.max(e),torch.mean(e))

        return loss


class ce_dice_boundry:
    def forward(self, pred, tar, alpha):
        assert alpha <= 1
        dice_tar = on_hot_my(tar)
        soft_mask = soft_max(pred)

        mask_onehot = class2one_hot(tar, 2)[0]  # because the res is bchw
        mask_distmap = one_hot2dist(mask_onehot.cpu().numpy())
        mask_distmap = torch.tensor(mask_distmap).float()

        cross = ce(pred, tar)
        a, b = dice(soft_mask, dice_tar)
        boundry = surface_loss(soft_mask, dist_maps, None)

        loss = alpha * (cross + a + b) + (1 - alpha) * boundry
        return loss


# ------------------------------------------------------------------------SurfaceLoss


# ------------------------------------------------------------------------LRFinder


class SegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_list,
        mode="train",
        boundary=False,
        boundary_norm=True,
        augmentation=None,
    ):
        # train and val urls required

        assert mode in [
            "train",
            "val",
            "track",
        ], "DataLoaderMode Must be one of train,val,track"
        self.augmentation = augmentation
        self.boundary = boundary
        self.boundary_norm = boundary_norm
        self.mode = mode
        self.img_dir = img_list

        print(mode.capitalize() + "Set Stats: ", len(self.img_dir))

    def __len__(self):
        return len(self.img_dir)

    def get_mskname(self, x):

        if self.mode == "train":
            return (
                x.replace("/images", "/masks")
                .replace(".jpg", ".png")
                .replace(".jpeg", ".png")
            )
        elif self.mode == "val":
            return (
                x.replace("/val", "/masks")
                .replace(".jpg", ".png")
                .replace(".jpeg", ".png")
            )
        elif self.mode == "track":
            return (
                x.replace("/track", "/masks")
                .replace(".jpg", ".png")
                .replace(".jpeg", ".png")
            )

    def __getitem__(self, idx):
        x = self.img_dir[idx]
        mask_name = self.get_mskname(x)
        img_name = x.split("/")
        img_name = [img_name[-3], img_name[-1]]  # folder+imgname
        img_name = ("__").join(img_name).split(".")[0]

        image = plt.imread(self.img_dir[idx])

        if np.shape(image)[-1] == 4:
            image = image[:, :, :-1]
        if np.shape(np.shape(image)) == (2,):
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        #         Single Channel: mini_pm,SuperVislyDatasetCleaned_MoreClean
        #         Index 1:AllHairSitYoga,ethnic_4augs ,AnimalCleaned_DataFirstRmvbg, KomalEverything5MayImgMasks,AarmashEverything5MayImgMasks
        try:
            try:
                mask = plt.imread(mask_name, 0)[:, :, 1]
            except:
                mask = plt.imread(mask_name, 0)
        except:
            print("mask not found", x, img_name, mask_name)

        try:
            mask.setflags(write=1)
        except:
            pass

        if np.max(mask) > 1:
            mask = mask / 255.0
        mask[mask < 0.5] = 0
        mask[mask > 0.5] = 1

        if len(np.shape(mask)) == 2:
            mask = np.expand_dims(mask, axis=2)  # albu
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if np.max(image) > 1.2:
            image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        image = np.transpose(image, axes=(2, 0, 1))

        image = torch.tensor(image)
        mask = torch.tensor(mask).float()[:, :, 0]

        if self.boundary:
            mask_onehot = class2one_hot(mask, 2)[0]  # because the res is bchw
            mask_distmap = one_hot2dist(mask_onehot.cpu().numpy(), self.boundary_norm)
            sample = {
                "image": image,
                "dice_mask": one_hot_my(mask),
                "ce_mask": mask,
                "boundary": mask_distmap,
                "image_names": img_name,
            }
        else:
            sample = {
                "image": image,
                "dice_mask": one_hot_my(mask),
                "ce_mask": mask,
                "image_names": img_name,
            }

        return sample


#
# ------------------------------------------------------------------------CombineCos
# Cell


def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


class _Annealer:
    def __init__(self, f, start, end):
        self.f, self.start, self.end = f, start, end

    def __call__(self, pos):
        return self.f(self.start, self.end, pos)


def annealer(f):
    "Decorator to make `f` return itself partially applied."

    @functools.wraps(f)
    def _inner(start, end):
        return _Annealer(f, start, end)

    return _inner


def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


def SchedCos(start, end):
    return _Annealer(sched_cos, start, end)


def combine_scheds(pcts, scheds):
    "Combine `scheds` according to `pcts` in one function"
    assert sum(pcts) == 1.0
    pcts = torch.tensor([0] + pcts)
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        if pos == 1.0:
            return scheds[-1](1.0)  # last function called on 1
        # nonzero returns index of non-zero elemnts and we take last one(can also do [-1])
        idx = (pos >= pcts).nonzero().max()
        if idx == len(pcts) - 1:
            return scheds[-1](1.0)
        # if pos is 0.999999999232 then this comes >1 in python rounding messing up below steps
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        # pos-pcts[idx] is the relative position from start we are in. pcts[idx+1]-pcts[idx] is the length of the current cycle
        return scheds[idx](actual_pos.item())

    return _inner


# Cell


def combined_cos(pct, start, middle, end):
    "Return a scheduler with cosine annealing from `start`→`middle` & `middle`→`end`"
    return combine_scheds(
        [pct, 1 - pct], [SchedCos(start, middle), SchedCos(middle, end)]
    )


class CombineCos:
    def __init__(
        self,
        opt,
        pct_till_middle,
        start,
        middle,
        end,
        steps_per_epoch,
        epochs,
        last_epoch=-1,
    ):
        self.opt = opt
        self.func = combined_cos(pct_till_middle, start, middle, end)

        self.total_steps = steps_per_epoch * epochs + 0.0001
        self.counter = 0

    def step(self):
        self.counter += 1
        pct = self.counter / self.total_steps
        assert pct <= 1, "Total Steps Excedded in CombineCos Scheduler"
        curr_lr = self.func(pct)

        for groups in self.opt.param_groups:
            groups["lr"] = curr_lr


## ------------------------------------------------------------------------Opt, scheduler


def kwargs_check(kwargs, var, chosen_scheduler):
    statement = (
        " and ".join(var) + " needed for " + str(chosen_scheduler) + " scheduler"
    )
    for e in var:
        assert e in kwargs.keys(), statement


class EmptyScheduler:
    def step(self):
        pass


def get_opt(model, chosen_opt, opt_lr, weight_decay, momentum):
    if chosen_opt == "sgd":
        #         wandb.config.lr_init = opt_lr; wandb.config.momentum = 0
        opt = torch.optim.SGD(
            model.parameters(), lr=opt_lr, momentum=momentum, weight_decay=weight_decay
        )
    elif chosen_opt == "sgdm":
        #         wandb.config.lr_init = opt_lr; wandb.config.momentum = momentum
        opt = torch.optim.SGD(
            model.parameters(), lr=opt_lr, momentum=momentum, weight_decay=weight_decay
        )
    elif chosen_opt == "adam":
        #         wandb.config.lr_init = opt_lr; wandb.config.opt_betas=(0.9, 0.999)
        opt = torch.optim.Adam(model.parameters(), lr=opt_lr, weight_decay=weight_decay)
    elif chosen_opt == "adamw":
        #         wandb.config.lr_init = opt_lr; wandb.config.opt_betas=(0.9, 0.999)
        opt = torch.optim.AdamW(
            model.parameters(), lr=opt_lr, weight_decay=weight_decay
        )
    return opt


def get_schedule(opt, chosen_scheduler, trainloader, **kwargs):
    """chosen_scheduler options : steplr | reduceplateau | cycliclr | onecycle | warmrestarts | stitchcos | none"""
    if chosen_scheduler == "steplr":
        required = ["change_epochs", "gamma"]
        kwargs_check(kwargs, required, chosen_scheduler)  # std: 8,0.1
        change_epochs = kwargs["change_epochs"]
        gamma = kwargs["gamma"]
        stepsize = change_epochs * len(trainloader)  # steps
        #         wandb.config.opt_change_epochs = change_epochs; wandb.config.opt_gamma = gamma
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=stepsize, gamma=gamma
        )

    elif chosen_scheduler == "reduceplateau":
        required = ["patience", "factor"]
        # patience=10;factor=0.1
        kwargs_check(kwargs, required, chosen_scheduler)
        patience = kwargs["patience"]
        factor = kwargs["factor"]
        wandb.config.opt_patience = patience
        wandb.config.opt_factor = factor
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=factor, patience=patience
        )
    elif chosen_scheduler == "cycliclr":
        required = ["base_lr", "max_lr"]
        # base_lr=0.001; max_lr=0.01; mode='triangular'
        kwargs_check(kwargs, required, chosen_scheduler)
        if mode not in kwargs.keys():
            mode = "triangular"
        else:
            model = kwargs["mode"]
        base_lr = kwargs["base_lr"]
        max_lr = kwargs["max_lr"]
        #         wandb.config.opt_base_lr = base_lr; wandb.config.opt_max_lr = max_lr;wandb.config.opt_mode = mode
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=base_lr, max_lr=max_lr, mode=mode
        )
    elif chosen_scheduler == "onecycle":
        required = ["max_lr", "epochs"]
        # requires max_lr,epochs
        kwargs_check(kwargs, required, chosen_scheduler)
        pct_start = 0.3
        anneal_strategy = "cos"
        cycle_momentum = True
        base_momentum = 0.85
        max_momentum = 0.95
        div_factor = 25.0
        final_div_factor = 10000.0
        max_lr = kwargs["max_lr"]
        epochs = kwargs["epochs"]
        #         wandb.config.opt_max_lr = max_lr; wandb.config.opt_pct_start = pct_start;wandb.config.opt_anneal_strategy = anneal_strategy
        #         wandb.config.opt_cycle_momentum = cycle_momentum; wandb.config.opt_base_momentum = base_momentum;wandb.config.opt_max_momentum = max_momentum
        #         wandb.config.opt_div_factor = div_factor; wandb.config.opt_final_div_factor = final_div_factor

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max_lr,
            steps_per_epoch=len(trainloader),
            epochs=epochs,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
    elif chosen_scheduler == "warmrestarts":
        required = ["t0", "tmult"]
        # in epochs #t0=1; tmult=2
        kwargs_check(kwargs, required, chosen_scheduler)
        t0 = kwargs["t0"] * len(trainloader)
        tmult = kwargs["tmult"]  # number of steps
        #         wandb.config.opt_t0 = t0; wandb.config.opt_tmult = tmult
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, t0, tmult)
    elif chosen_scheduler == "stitchcos":
        required = ["pct", "start", "middle", "end", "epochs"]
        # in epochs #t0=1; tmult=2
        kwargs_check(kwargs, required, chosen_scheduler)
        scheduler = CombineCos(
            opt,
            kwargs["pct"],
            kwargs["start"],
            kwargs["middle"],
            kwargs["end"],
            len(trainloader),
            kwargs["epochs"],
        )
    elif chosen_scheduler == "none":
        #         wandb.config.scheduler = 'EmptyScheduler'
        scheduler = EmptyScheduler()
    return scheduler


# ------------------------------------------------------------------------Image_Tracker


class Image_Tracker:
    def __init__(self, model, dataloader, track_dir, save_initial):
        if track_dir[-1] != "/":
            track_dir += "/"
        self.model = model
        self.dataloader = dataloader
        self.track_dir = track_dir
        self.save_initial = save_initial
        self.counter = 0
        create = True

        if os.path.exists(track_dir + save_initial):
            overwrite = input(
                "ImageTracking Folder Already Exists: "
                + track_dir
                + save_initial
                + "   Overwrite? y/n"
            )
            overwrite = True if overwrite == "y" else False
            if overwrite:
                shutil.rmtree(track_dir + save_initial)
                os.mkdir(track_dir + save_initial)
            else:
                assert 1 == 0, "Tracker Stopped"

        else:
            os.mkdir(track_dir + save_initial)

        for idx, data in enumerate(dataloader):
            for i in range(np.shape(data["image_names"])[0]):
                # make image subfolders
                os.mkdir(track_dir + save_initial + "/" + data["image_names"][i])

    def __call__(self):
        self.counter += 1
        for idx, data in enumerate(self.dataloader):
            x = data["image"].float().to(device)
            image_names = data["image_names"]

            d0, d1, d2, d3, d4, d5, d6 = self.model(x)
            d0 = d0.detach().cpu().float().numpy()

            x = x.detach().permute(0, 2, 3, 1).cpu().float().numpy()
            for i in range(np.shape(image_names)[0]):
                matplotlib.image.imsave(
                    self.track_dir
                    + self.save_initial
                    + "/"
                    + image_names[i]
                    + "/"
                    + str(self.counter)
                    + ".png",
                    d0[i][0],
                )


# ------------------------------------------------------------------------Saver


class Saver:
    def __init__(
        self,
        model,
        opt,
        top_models,
        worst_val,
        save_dir,
        save_initial,
        epoch_saver=None,
    ):
        self.top_models = top_models
        self.worst_val = worst_val
        self.model = model
        self.opt = opt
        self.epoch_saver = epoch_saver
        self.save_dir = save_dir
        self.save_initial = save_initial

    def __call__(self, loss, epoch):
        # loss is metric to save on like avg val dice
        top_models = self.top_models
        worst_val = self.worst_val
        if loss < worst_val:
            print("------------- Model Saving -------------------", epoch)
            # sort maybe not needed
            top_models = sorted(top_models, key=lambda x: x[1])

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "opt": self.opt.state_dict(),
            }
            torch.save(
                checkpoint,
                self.save_dir
                + self.save_initial
                + "-"
                + str(round(loss, 3))
                + "|"
                + str(epoch + 1),
            )
            to_be_deleted = (
                self.save_dir
                + self.save_initial
                + "-"
                + str(round(top_models[-1][1], 3))
                + "|"
                + str(top_models[-1][0])
            )  # ...loss|epoch
            top_models.append((epoch + 1, loss))
            # sort after addition of new val
            top_models = sorted(top_models, key=lambda x: x[1])
            top_models.pop(-1)
            worst_val = top_models[-1][1]
            print("Top_models List", top_models)
            try:
                os.remove(to_be_deleted)
            except:
                print("Could not delete previous best. File not found:", to_be_deleted)

        self.top_models = top_models
        self.worst_val = worst_val

        if self.epoch_saver and epoch % self.epoch_saver == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "opt": self.opt.state_dict(),
            }
            torch.save(
                checkpoint,
                self.save_dir
                + self.save_initial
                + "-"
                + str(round(loss, 4))
                + "|"
                + str(epoch + 1)
                + "r",
            )


class SaverV2:
    def __init__(
        self,
        run_id,
        model,
        opt,
        top_models,
        worst_val,
        save_dir,
        save_initial,
        epoch_saver=None,
    ):
        self.top_models = top_models
        self.worst_val = worst_val
        self.model = model
        self.opt = opt
        self.epoch_saver = epoch_saver
        self.save_dir = save_dir
        self.save_initial = save_initial
        self.run_id = run_id

    def __call__(self, loss, epoch):
        # loss is metric to save on like avg val dice
        top_models = self.top_models
        worst_val = self.worst_val
        if loss < worst_val:
            print("------------- Model Saving -------------------", epoch)
            # sort maybe not needed
            top_models = sorted(top_models, key=lambda x: x[1])

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "opt": self.opt.state_dict(),
            }
            torch.save(
                checkpoint,
                self.save_dir
                + self.save_initial
                + "-"
                + str(round(loss, 3))
                + "|"
                + str(epoch + 1),
            )
            to_be_deleted = (
                self.save_dir
                + self.save_initial
                + "-"
                + str(round(top_models[-1][1], 3))
                + "|"
                + str(top_models[-1][0])
            )  # ...loss|epoch
            top_models.append((epoch + 1, loss))
            # sort after addition of new val
            top_models = sorted(top_models, key=lambda x: x[1])
            top_models.pop(-1)
            worst_val = top_models[-1][1]
            print("Top_models List", top_models)
            try:
                os.remove(to_be_deleted)
            except:
                print("Could not delete previous best. File not found:", to_be_deleted)

        self.top_models = top_models
        self.worst_val = worst_val

        if self.epoch_saver and epoch % self.epoch_saver == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "run_id": self.run_id,
            }
            torch.save(
                checkpoint,
                self.save_dir
                + self.save_initial
                + "-"
                + str(round(loss, 4))
                + "|"
                + str(epoch + 1)
                + "r",
            )


# ------------------------------------------------ Lap Loss
# change default device from cpu to cuda
# from FBA Matting issues


L1_nonreduce = torch.nn.L1Loss(reduction="none")
L1 = torch.nn.L1Loss()


def L1_mask(pred, tar, mask):
    # L1 between two images w mask
    loss = L1_nonreduce(pred, tar) * mask
    return loss.sum() / mask.sum()


def gauss_kernel(size=5, device=torch.device("cuda"), channels=3):
    kernel = torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ]
    )
    kernel /= 256.0
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x):
    cc = torch.cat(
        [
            x,
            torch.zeros(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device
            ),
        ],
        dim=3,
    )
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat(
        [
            cc,
            torch.zeros(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2, device=x.device
            ),
        ],
        dim=3,
    )
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    try:
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode="reflect")
    except:
        print("erro", img.size())
    out = torch.nn.functional.conv2d(img.float(), kernel.float(), groups=img.shape[1])
    return out


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr


class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=1, device=torch.device("cuda")):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device).cuda()

    def forward(self, input, target):

        alpha_gt = target[:, 0, :].unsqueeze(1)
        pyr_input = laplacian_pyramid(
            img=input, kernel=self.gauss_kernel, max_levels=self.max_levels
        )
        pyr_target = laplacian_pyramid(
            img=alpha_gt, kernel=self.gauss_kernel, max_levels=self.max_levels
        )
        return sum(
            torch.nn.functional.l1_loss(A[0], A[1]) * (2 ** (i))
            for i, A in enumerate(zip(pyr_input, pyr_target))
        )


def update_wandb_logs(metric_names, metric_vals):
    assert len(metric_names) == len(metric_vals)
    dict = {}
    for i in range(len(metric_names)):

        try:
            dict[metric_names[i]] = metric_vals[i].item()
        except:
            dict[metric_names[i]] = metric_vals[i]
    wandb.log(dict)


def update_lists(mets, l):
    for i in range(len(mets)):
        try:
            l[i].append(mets[i].item())
        except:
            l[i].append(mets[i])


bce_loss = torch.nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    labels_v = labels_v.double()

    loss0 = bce_loss(d0.double(), labels_v)
    loss1 = bce_loss(d1.double(), labels_v)
    loss2 = bce_loss(d2.double(), labels_v)
    loss3 = bce_loss(d3.double(), labels_v)
    loss4 = bce_loss(d4.double(), labels_v)
    loss5 = bce_loss(d5.double(), labels_v)
    loss6 = bce_loss(d6.double(), labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss0, loss


def shadow_effect(
    alpha: torch.Tensor,
    bg: torch.Tensor,
    aug_likelyhood: float = 0.3,
    n_shadow: int = 1,
) -> torch.Tensor:
    bg = bg.clone()
    for i in range(n_shadow):
        aug_shadow_idx = torch.rand(len(alpha)) < aug_likelyhood
        if aug_shadow_idx.any():
            aug_shadow = alpha[aug_shadow_idx].mul(random.uniform(0.1, 0.7))
            aug_shadow = T.RandomAffine(
                degrees=(-5, 5), translate=(0.0, 0.01), scale=(0.9, 1.1), shear=(-3, 3)
            )(aug_shadow)
            aug_shadow = kornia.filters.box_blur(
                aug_shadow, (random.choice(range(1, 15)),) * 2
            )
            bg[aug_shadow_idx] = bg[aug_shadow_idx].sub_(aug_shadow).clamp_(0, 1)
    return bg


def get_extreme_x(binary_mask: np.ndarray) -> np.ndarray:
    cnts = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv.contourArea)

    # Obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    return left, right


def shadow_lines(
    gt_alpha: torch.Tensor,
    bg: torch.Tensor,
    aug_likelyhood: float = 0.3,
    n_lines: int = 1,
) -> torch.Tensor:
    # create bg
    bg = bg.clone()
    try:
        left, right = get_extreme_x(
            (gt_alpha[0].numpy().transpose(2, 1, 0) * 255).astype(np.uint8)
        )
    except Exception as e:
        print(e)
        print("Countours not detected... Defaulting to whole image")
        left = (0, 0)
        right = (bg[0].shape[2], bg[0].shape[1])
    for i in range(n_lines):
        # alpha predefined as a arbitrary black line
        alpha = np.zeros(bg[0].shape).transpose(2, 1, 0).astype(np.uint8).copy()
        pt1_x = random.randint(left[0], right[0])
        pt1_y = 0
        pt2_x = pt1_x  # so that it's vertical
        pt2_y = alpha.shape[0]
        thickness = random.randint(1, 5)
        cv.line(
            alpha,
            pt1=(pt1_x, pt1_y),
            pt2=(pt2_x, pt2_y),
            color=(255, 255, 255),
            thickness=thickness,
        )
        alpha = torch.tensor(alpha.transpose(2, 1, 0)).unsqueeze(axis=0) / 255.0

        # create shadows
        aug_shadow_idx = torch.rand(len(alpha)) < aug_likelyhood
        if aug_shadow_idx.any():
            aug_shadow = alpha[aug_shadow_idx].mul(random.uniform(0.3, 0.7))
            aug_shadow = T.RandomAffine(
                degrees=(-5, 5), translate=(0.0, 0.01), scale=(1, 1.1), shear=(-5, 5)
            )(aug_shadow)
            aug_shadow = kornia.filters.box_blur(
                aug_shadow, (random.choice(range(5, 10)),) * 2
            )
            bg[aug_shadow_idx] = bg[aug_shadow_idx].sub_(aug_shadow).clamp_(0, 1)
    return bg


"""This module contains simple helper functions """


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def load_weights(model, states, strict=False):
    with torch.no_grad():
        for k, v in states.items():
            model_state = rgetattr(model, k)
            if v.shape == model_state.shape:
                model_state.copy_(v)
            else:
                if strict:
                    raise Exception(
                        f"Shape mismatch: {k} expected shape {model_state.shape}, got {v.shape} "
                    )
                else:
                    print(f"Skipping {k}: model {model_state.shape}, states{v.shape}")

    print("Loaded pretrained weights")
    return model
