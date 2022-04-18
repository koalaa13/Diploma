import argparse
import os

import torchvision
from torch.nn import BCELoss
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.cuda
from torch.utils.data import DataLoader
from time import process_time
import torch.nn.functional as F

from estimator.Estimator import Estimator
from utils.DatasetTransformer import Transformer
from wgan.Discriminator import Discriminator
from wgan.Generator import Generator
from wgan.NNEmbeddingDataset import NNEmbeddingDataset

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0') if cuda else torch.device('cpu')
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--embedding_width", type=int, default=500, help="width of an embedding")
    parser.add_argument("--embedding_height", type=int, default=500, help="height of an embedding")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="size of a batch to train")
    parser.add_argument("--n_epochs", type=int, default=100, help="epochs count")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper bound for disc weights")
    parser.add_argument("--n_critic", type=int, default=4, help="train generator every n_critic iterations")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    parser.add_argument("--batch_size_train", type=int, default=64,
                        help="batch size for training generated networks in estimator")
    parser.add_argument("--batch_size_test", type=int, default=1000,
                        help="batch size for testing generated networks in estimator")
    options = parser.parse_args()
    print(options)

    output_generator_dim = options.embedding_width * options.embedding_height
    obj_shape = (options.embedding_width, options.embedding_height)

    generator_dims = [options.latent_dim, 128, 256, 512, 1024, output_generator_dim]
    discriminator_dims = [output_generator_dim, 512, 256]

    generator = Generator(generator_dims).to(device)
    discriminator = Discriminator(discriminator_dims).to(device)

    train_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/mnist',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=options.batch_size_train,
        shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("../data/mnist",
                                   train=False,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=options.batch_size_test,
        shuffle=True)

    estimator = Estimator(options.embedding_width, options.embedding_height, train_dataloader, test_dataloader,
                          F.nll_loss, device)

    need_transform = False
    if need_transform:
        print("STARTED DATASET TRANSFORM")
        Transformer(options.embedding_width, options.embedding_height).transform()
        print("FINISHED DATASET TRANSFORM")

    dataloader = torch.utils.data.DataLoader(
        NNEmbeddingDataset("../data/nn_embedding"),
        batch_size=options.batch_size,
        shuffle=True,
        # num_workers=8,
        # pin_memory=True
    )

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=options.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=options.lr)

    batches_cnt = 0
    for epoch in range(options.n_epochs):
        for i, objs in enumerate(dataloader):
            real_objs = objs.to(device)

            # --------------------
            #  Train Discriminator
            # --------------------
            optimizer_D.zero_grad()

            z = torch.randn(objs.shape[0], options.latent_dim).to(device)

            fake_objs = generator(z, obj_shape).detach()
            loss_D = -torch.mean(discriminator(real_objs)) + torch.mean(discriminator(fake_objs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-options.clip_value, options.clip_value)

            # Train generator every n_critic iterations
            if i % options.n_critic == 0:
                # --------------------
                # Train Generator
                # --------------------

                optimizer_G.zero_grad()

                gen_objs = generator(z, obj_shape)
                estimator_feedbacks = []
                gen_objs_cnt = gen_objs.size(0)
                # TODO there is some questions about normalization and de-normalization of dataset
                for k in range(gen_objs_cnt):
                    estimator_feedbacks.append(int(estimator.check(gen_objs[k])))
                # mb minus feedback from TPE_Estimator
                loss_G = -torch.mean(discriminator(gen_objs)) + BCELoss(torch.tensor(estimator_feedbacks),
                                                                        torch.ones(gen_objs_cnt))

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (
                        epoch, options.n_epochs, batches_cnt % len(dataloader), len(dataloader), loss_D.item(),
                        loss_G.item())
                )

            # if batches_cnt % options.sample_interval == 0:
            #     save_image(fake_objs.data[:25], "images/%d.png" % batches_cnt, nrow=5, normalize=True)
            batches_cnt += 1

    need_example = False
    if need_example:
        generator.eval()
        torch.set_printoptions(profile="full")
        os.makedirs("examples", exist_ok=True)
        z = torch.randn(1, options.latent_dim).to(device)
        fake = generator(z, obj_shape).detach()
        with open("examples/generated_example", "w+") as f:
            print(fake, file=f)
