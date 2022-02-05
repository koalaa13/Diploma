import argparse
import os

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.cuda
from torch.utils.data import DataLoader

from wgan.Discriminator import Discriminator
from wgan.Generator import Generator

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# for example for images = height * width * channels_count
parser.add_argument("--output_generator_dim", type=int, default=28 * 28 * 1, help="output size of generator")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="size of a batch to train")
parser.add_argument("--n_epochs", type=int, default=300, help="epochs count")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper bound for disc weights")
parser.add_argument("--n_critic", type=int, default=5, help="train generator every n_critic iterations")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
options = parser.parse_args()
print(options)

generator_dims = [options.latent_dim, 128, 256, 512, 1024, options.output_generator_dim]
discriminator_dims = [options.output_generator_dim, 512, 256]

generator = Generator(generator_dims).to(device)
discriminator = Discriminator(discriminator_dims).to(device)

os.makedirs("../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    ),
    batch_size=options.batch_size,
    shuffle=True
)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=options.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=options.lr)

batches_cnt = 0
for epoch in range(options.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_images = imgs.to(device)

        # --------------------
        #  Train Discriminator
        # --------------------
        optimizer_D.zero_grad()

        # Generate the same count of images as real ones
        z = torch.randn(imgs.shape[0], options.latent_dim)

        fake_imgs = generator(z).detach()
        loss_D = -torch.mean(discriminator(real_images)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-options.clip_value, options.clip_value)

        # Train generator every n_critic iterations
        if i % options.n_critic:
            # --------------------
            # Train Generator
            # --------------------

            optimizer_G.zero_grad()

            gen_imgs = generator(z)
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch, options.n_epochs, batches_cnt % len(dataloader), len(dataloader), loss_D.item(),
                    loss_G.item())
            )

        if batches_cnt % options.sample_interval:
            save_image(fake_imgs.data[:25], "images/%d.png" % batches_cnt, nrow=5, normalize=True)
        batches_cnt += 1
