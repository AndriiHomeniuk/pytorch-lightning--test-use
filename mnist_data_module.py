import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir='./data', batch_size=128, num_workers=int(os.cpu_count() / 2)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1,), (0.3,))
        ])

        self.dl_dict = {'batch_size': self.batch_size, 'num_workers': self.num_workers}

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Validation data not strictly necessary for GAN but added for completeness
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    # For dataloaders, usually just wrap dataset defined in setup
    def train_dataloader(self):
        return DataLoader(self.mnist_train, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self.mnist_train, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.mnist_train, **self.dl_dict)
