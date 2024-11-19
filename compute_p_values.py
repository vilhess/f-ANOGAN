import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from model import ConvDiscriminator, ConvGenerator, Encoder
from utils import get_dataset_by_digit

DEVICE='cuda'

transform = Compose([
    ToTensor(),
    Normalize(0.5, 0.5)
])

trainset = MNIST(root="./", train=True, download=False, transform=transform)
testset = MNIST(root="./", train=False, download=False, transform=transform)

train_size = 40000
val_size = len(trainset) - train_size

generator = torch.Generator().manual_seed(42)
trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)

train_dic_dataset = get_dataset_by_digit(trainset)
val_dic_dataset = get_dataset_by_digit(valset)
test_dic_dataset = get_dataset_by_digit(testset)

LATENT_DIM=100

for anormal in range(10):

  final_results = {i:None for i in range(10)}

  ANORMAL= anormal
  NORMAL_DIGITS = [i for i in range(10)]
  NORMAL_DIGITS.remove(ANORMAL)

  normal_trainset = torch.cat([train_dic_dataset[i] for i in NORMAL_DIGITS])
  normal_val = torch.cat([val_dic_dataset[i] for i in NORMAL_DIGITS])

  encoder = Encoder(in_channels=1, hidden_channels=64, z_dim=LATENT_DIM).to(DEVICE)
  gen = ConvGenerator(z_dim=LATENT_DIM, hidden_dim=64).to(DEVICE)
  disc = ConvDiscriminator(im_channel=1, hidden_dim=64).to(DEVICE)

  checkpoint = torch.load(f"/content/drive/MyDrive/coding/ANOGAN/checkpoints/CGAN_checkpoint_{ANORMAL}.pkl", map_location=torch.device('cpu'))

  gen.load_state_dict(checkpoint["gen_state_dict"])
  disc.load_state_dict(checkpoint["disc_state_dict"])
  encoder.load_state_dict(checkpoint["encoder_state_dict"])

  gen.eval()
  disc.eval()
  encoder.eval()

  f = nn.Sequential(*list(disc.disc.children())[:-1])  # On exclut la dernière couche (Conv2d)
  f.append(nn.Flatten(start_dim=1))

  with torch.no_grad():
    encoded = encoder(normal_val.to(DEVICE))
    reconstructed = gen(encoded)
    features_decoded = f(reconstructed)
    features_batch = f(normal_val.to(DEVICE))

  reconstructed_loss = reconstructed.flatten(start_dim=1)
  normal_val_loss = normal_val.flatten(start_dim=1).to(DEVICE)
  features_decoded = features_decoded.flatten(start_dim=1)
  features_batch = features_batch.flatten(start_dim=1)

  loss_residual = ((reconstructed_loss - normal_val_loss)**2).sum(dim=1)
  loss_discriminator = ((features_decoded - features_batch)**2).sum(dim=1)

  complete_loss = loss_residual+loss_discriminator

  val_scores = - complete_loss

  val_scores_sorted, indices = val_scores.sort()

  final_results = {i:[None, None] for i in range(10)}

  for digit in range(10):
    inputs_test = test_dic_dataset[digit].to(DEVICE)

    with torch.no_grad():
      encoded = encoder(inputs_test)
      reconstructed = gen(encoded)
      features_decoded = f(reconstructed)
      features_batch = f(inputs_test)

    reconstructed_loss = reconstructed.flatten(start_dim=1)
    inputs_test_loss = inputs_test.flatten(start_dim=1).to(DEVICE)
    features_decoded = features_decoded.flatten(start_dim=1)
    features_batch = features_batch.flatten(start_dim=1)

    loss_residual = ((reconstructed_loss - inputs_test_loss)**2).sum(dim=1)
    loss_discriminator = ((features_decoded - features_batch)**2).sum(dim=1)

    complete_loss = loss_residual+loss_discriminator

    test_scores = - complete_loss

    test_p_values = (1 + torch.sum(test_scores.unsqueeze(1) >= val_scores_sorted, dim=1)) / (len(val_scores_sorted) + 1)

    final_results[digit][0] = test_p_values.tolist()
    final_results[digit][1] = len(inputs_test)

    with open(f"/content/drive/MyDrive/coding/ANOGAN/p_values/anogan_{ANORMAL}.json", "w") as file:
        json.dump(final_results, file)