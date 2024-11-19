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

  for i in range(10):
    test_inputs = test_dic_dataset[i].to(DEVICE)
    with torch.no_grad():
        encoded = encoder(test_inputs)
        decoded = gen(encoded)

    loss_residual = nn.functional.mse_loss(decoded, test_inputs)
    features_decoded = f(decoded).flatten(start_dim=1)
    features_batch = f(test_inputs).flatten(start_dim=1)

    loss_discriminator = nn.functional.mse_loss(features_decoded, features_batch)

    complete_loss = loss_residual + loss_discriminator
    final_results[i]=complete_loss.item()

  plt.bar(final_results.keys(), final_results.values())
  plt.title('Mean Loss for each digit')
  plt.savefig(f'/content/drive/MyDrive/coding/ANOGAN/figures/Mean_error_{ANORMAL}.jpg', bbox_inches='tight', pad_inches=0)
  plt.close()
