from pytorch_lightning import Trainer

from gan import GAN
from mnist_data_module import MNISTDataModule


if __name__ == '__main__':
    trainer = Trainer(max_epochs=20)
    dm = MNISTDataModule()
    model = GAN()
    trainer.fit(model, dm)
