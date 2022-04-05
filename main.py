import torch.utils.data
from torchvision import datasets
from torchvision.transforms import transforms
from wgan.NNEmbeddingDataset import NNEmbeddingDataset

# dataset = datasets.MNIST(
#     "../data/mnist",
#     train=True,
#     download=True,
#     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# )

dataset = NNEmbeddingDataset("data/nn_embedding", 50, 50)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for i, img in enumerate(dataloader):
    print(img.shape)
