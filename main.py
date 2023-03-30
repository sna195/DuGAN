import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from ..geometry-score import gs
import matplotlib.pyplot as plt

"""
import easydict

opt = easydict.EasyDict({
    "n_epochs": 200,
    "batch_size": 64,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "n_cpu": 8,
    "latent_dim": 100,
    "img_size": 28,
    "channels": 1,
    "sample_interval": 5000,
    "alpha": 0.0005
})
"""

os.makedirs("images", exist_ok=True)
os.makedirs("images_dg", exist_ok=True)
os.makedirs("../images_trained", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5000, help="interval between image samples")
parser.add_argument("--alpha", type=float, default=0.0005, help="hyper parameter of l2 norm")
opt = parser.parse_args()
print(opt)

a = opt.n_cpu
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class DuplicatedGenerator(nn.Module):
    def __init__(self):
        super(DuplicatedGenerator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
duplicated_generator = DuplicatedGenerator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    duplicated_generator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_DG = torch.optim.Adam(duplicated_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

const_z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        torch.autograd.set_detect_anomaly(True)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z_gen = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z_gen)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # ---------------------------
        #  Train DuplicatedGenerator
        # ---------------------------

        optimizer_DG.zero_grad()

        # Generate a batch of images
        dgen_imgs = duplicated_generator(const_z)

        # Loss measures generator's ability to fool the discriminator
        dg_loss = torch.mean(torch.sum(torch.sum((dgen_imgs - gen_imgs.detach()) ** 2, 3), 2))

        g_loss = g_loss - opt.alpha * torch.mean(torch.sum(torch.sum((dgen_imgs.detach() - gen_imgs) ** 2, 3), 2))

        g_loss.backward()
        optimizer_G.step()

        # dg_loss.backward()
        # optimizer_DG.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            save_image(dgen_imgs.data[:25], "images_dg/%d.png" % batches_done, nrow=5, normalize=True)


# ------
# Test
# ------

os.makedirs("../data/mnist_test", exist_ok=True)
test_data = datasets.MNIST(
    "../data/mnist_test",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
)

os.makedirs("../images_trained/lambda%.4f" % opt.alpha, exist_ok=True)
z_test = Variable(Tensor(np.random.normal(0, 1, (len(test_data.data), opt.latent_dim))))
trained_imgs = generator(z_test)
for i, img in enumerate(trained_imgs):
    save_image(img, "../images_trained/lambda%.4f/%d.png" % (opt.alpha, i), normalize=True)

# -----------------
# Geometry score
# -----------------

test_data = np.array(np.reshape(test_data.data, (-1, opt.img_size ** 2)))
rlts = gs.rlts(test_data, gamma=1.0/128, n=100)
mrlt = np.mean(rlts, axis=0)
gs.fancy_plot(mrlt, label='MRLT of GAN')
plt.xlim([0, 30])
plt.legend()

trained_imgs = np.array(np.reshape(trained_imgs, (-1, opt.img_size ** 2)))
rlts_du = gs.rlts(trained_imgs, gamma=1.0/128, n=100)
mrlt_du = np.mean(rlts_du, axis=0)
gs.fancy_plot(mrlt_du, label='MRLT of DuGAN')
plt.xlim([0, 30])
plt.legend()

plt.savefig("../dugan_gs.png")
