# import torch.utils.data
#
# from wgan.NNEmbeddingDataset import NNEmbeddingDataset
#
# dataset = NNEmbeddingDataset("data/nn_embedding", 50, 50)
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=2,
#     shuffle=False
# )
#
# for i, obj in enumerate(dataloader):
#     print(obj)
