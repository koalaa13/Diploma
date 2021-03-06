import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy
import torchvision
from torch.nn import BCELoss
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.cuda
from torch.utils.data import DataLoader
from time import process_time
import torch.nn.functional as F

from embedding.graph import NODE_EMBEDDING_DIMENSION
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
    parser.add_argument("--latent_dim", type=int, default=5, help="dimensionality of the latent space")
    parser.add_argument("--embedding_width", type=int, default=1, help="width of an embedding")
    parser.add_argument("--embedding_height", type=int, default=14, help="height of an embedding")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="size of a batch to train")
    parser.add_argument("--n_epochs", type=int, default=1000, help="epochs count")
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
    obj_shape = (options.embedding_height, options.embedding_width)

    generator_dims = [options.latent_dim, 10, output_generator_dim]
    discriminator_dims = [output_generator_dim, 10, 5]

    generator = Generator(generator_dims, obj_shape).to(device)
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

    print("DATASET TRANSFORMATION STARTED")
    os.makedirs('../data/nn_super_small_embedding_transformed', exist_ok=True)
    transformer = Transformer(options.embedding_width, options.embedding_height)
    transformer.transform_dataset('../data/nn_super_small_embedding',
                                  '../data/nn_super_small_embedding_transformed')
    print("DATASET TRANSFORMATION FINISHED")

    dataloader = torch.utils.data.DataLoader(
        NNEmbeddingDataset('../data/nn_super_small_embedding_transformed'),
        batch_size=options.batch_size,
        shuffle=True,
        # num_workers=8,
        # pin_memory=True
    )

    # estimator = Estimator(options.embedding_width, options.embedding_height - Estimator.additional_ops_count,
    #                       train_dataloader, test_dataloader, device, '../estimator/saved_estimator')
    # print(len(estimator.bad_center))
    # print(len(estimator.good_center))
    # print(len(estimator.good_center[0]))
    # print(len(estimator.bad_center[0]))

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=options.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=options.lr)

    # estimator_loss = BCELoss()

    epochs = []
    losses_G = []
    losses_D = []
    batches_cnt = 0
    for epoch in range(options.n_epochs):
        for i, objs in enumerate(dataloader):
            real_objs = objs.to(device)

            # --------------------
            #  Train Discriminator
            # --------------------
            optimizer_D.zero_grad()

            z = torch.randn(objs.shape[0], options.latent_dim).to(device)

            fake_objs = generator(z).detach()
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

                gen_objs = generator(z)
                # estimator_feedbacks = []
                # gen_objs_cnt = gen_objs.size(0)
                # for k in range(gen_objs_cnt):
                #     estimator_feedbacks.append(float(int(estimator.check(gen_objs[k]))))
                # mb minus feedback from TPE_Estimator
                loss_G = -torch.mean(discriminator(gen_objs))\
                         # + estimator_loss(torch.tensor(estimator_feedbacks),
                         #                                                       torch.ones(gen_objs_cnt))

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (
                        epoch, options.n_epochs, batches_cnt % len(dataloader), len(dataloader), loss_D.item(),
                        loss_G.item())
                )
                epochs.append(epoch)
                losses_G.append(loss_G.item())
                losses_D.append(loss_D.item())

            if batches_cnt % options.sample_interval == 0:
                fake = fake_objs.data.cpu().numpy().tolist()
                for jj in range(len(fake)):
                    transformer.de_transform_embedding(fake[jj])
                with open('examples/generated_example' + str(batches_cnt), 'w+') as f:
                    f.write(json.dumps(fake))
            batches_cnt += 1

    with open('./epoches', 'w+') as f:
        f.write(json.dumps(epochs))
    with open('./losses_G', 'w+') as f:
        f.write(json.dumps(losses_G))
    with open('./losses_D', 'w+') as f:
        f.write(json.dumps(losses_D))
    legend = [plt.plot(epochs, losses_G, label='Generator Loss')[0],
              plt.plot(epochs, losses_D, label='Discriminator Loss')[0]]
    plt.xlabel('Epochs')
    plt.ylabel('GAN parts loss')
    plt.legend(handles=legend)
    plt.savefig('GAN parts losses')

    # need_example = True
    # if need_example:
    #     generator.eval()
    #     numpy.set_printoptions(threshold=sys.maxsize)
    #     os.makedirs("examples", exist_ok=True)
    #     examples_count = 10
    #     z = torch.randn(examples_count, options.latent_dim).to(device)
    #     fake = generator(z).detach().cpu().numpy()
    #     for i in range(examples_count):
    #         transformer.de_transform_embedding(fake[i])
    #         with open("examples/generated_example" + str(i), "w+") as f:
    #             print(fake[i], file=f)
    torch.save(generator.state_dict(), './generator_weights')
    torch.save(discriminator.state_dict(), './discriminator_weights')
