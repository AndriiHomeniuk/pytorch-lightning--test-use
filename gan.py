from collections import OrderedDict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid

from discriminator import Discriminator
from generator import Generator


class GAN(LightningModule):
    # Initialize. Define latent dim, learning rate, and Adam betas
    def __init__(self, latent_dim=100, lr=0.0002,

                 b1=0.5, b2=0.999, batch_size=128):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch

        # sample noise
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)

        # train generator
        if optimizer_idx == 0:
            self.generated_imgs = self(z)
            predictions = self.discriminator(self.generated_imgs)
            g_loss = self.adversarial_loss(predictions, torch.ones(real_imgs.size(0), 1))

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            real_preds = self.discriminator(real_imgs)
            real_loss = self.adversarial_loss(real_preds, torch.ones(real_imgs.size(0), 1))

            fake_preds = self.discriminator(self(z).detach())
            fake_loss = self.adversarial_loss(fake_preds, torch.zeros(real_imgs.size(0), 1))

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        # log sampled images
        sample_imgs = self(self.validation_z)
        grid = make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
