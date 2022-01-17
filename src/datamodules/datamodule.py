import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, train, val, test=None):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test 

    def train_dataloader(self):
        return self.train.dataloader

    def val_dataloader(self):
        return self.val.dataloader

    def test_dataloader(self):
        return self.test.dataloader