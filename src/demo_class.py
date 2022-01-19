class Train:
    def __init__(self, epoch, lr, optimizer) -> None:
        self.lr = lr
        self.epoch = epoch
        self.optimizer = optimizer


class Dataset:
    def __init__(self, img_size, batch) -> None:
        self.img_size = img_size
        self.batch = batch
        