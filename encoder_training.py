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

EPOCHS=10
BATCH_SIZE=128

for anormal in range(10):
  print(f"Working on anomaly digit {anormal}")
  ANORMAL= anormal
  NORMAL_DIGITS = [i for i in range(10)]
  NORMAL_DIGITS.remove(ANORMAL)

  normal_trainset = torch.cat([train_dic_dataset[i] for i in NORMAL_DIGITS])
  normal_val = torch.cat([val_dic_dataset[i] for i in NORMAL_DIGITS])


  encoder = Encoder(in_channels=1, hidden_channels=64, z_dim=LATENT_DIM).to(DEVICE)

  trainloader = DataLoader(normal_trainset, batch_size=BATCH_SIZE, shuffle=True)

  optimizer = optim.Adam(encoder.parameters(), lr=1e-4)


  gen = ConvGenerator(z_dim=LATENT_DIM, hidden_dim=64).to(DEVICE)
  disc = ConvDiscriminator(im_channel=1, hidden_dim=64).to(DEVICE)

  checkpoint = torch.load(f"/content/drive/MyDrive/coding/ANOGAN/checkpoints/CGAN_checkpoint_{ANORMAL}.pkl", map_location=torch.device('cpu'))
  gen.load_state_dict(checkpoint["gen_state_dict"])
  disc.load_state_dict(checkpoint["disc_state_dict"])

  gen.eval()
  disc.eval()

  f = nn.Sequential(*list(disc.disc.children())[:-1])  # On exclut la dernière couche (Conv2d)
  f.append(nn.Flatten(start_dim=1))

  for epoch in range(EPOCHS):
      curr_loss = 0

      for batch in tqdm(trainloader):
          batch = batch.to(DEVICE)

          encoded = encoder(batch)
          decoded = gen(encoded)

          loss_residual = nn.functional.mse_loss(decoded, batch)

          features_decoded = f(decoded).flatten(start_dim=1)
          features_batch = f(batch).flatten(start_dim=1)

          loss_discriminator = nn.functional.mse_loss(features_decoded, features_batch)

          complete_loss = loss_residual + loss_discriminator

          optimizer.zero_grad()
          complete_loss.backward()
          optimizer.step()

          curr_loss+=complete_loss.item()
      print(f"For epoch {epoch+1}/{EPOCHS} ; loss : {curr_loss/len(trainloader)}")

  checkpoint["encoder_state_dict"] = encoder.state_dict()
  torch.save(checkpoint, f"/content/drive/MyDrive/coding/ANOGAN/checkpoints/CGAN_checkpoint_{ANORMAL}.pkl")

  test_batch = []
  for i in range(10):
    test_batch.append(test_dic_dataset[i][0])
  test_batch = torch.stack(test_batch)

  grid1 = make_grid(test_batch, nrow=10)

  inputs = test_batch.to(DEVICE)

  encoder.eval()

  with torch.no_grad():
    encoded = encoder(inputs)
    reconstructed = gen(encoded)

  grid2 = make_grid(reconstructed.detach().cpu(), nrow=10)

  grid = torch.cat([grid1, grid2], dim=1)
  plt.imshow(grid.permute(1, 2, 0))
  plt.title('Above : Example ; Below : Reconstruction')
  plt.axis('off')
  plt.savefig(f'/content/drive/MyDrive/coding/ANOGAN/figures/Encoded_reconstructed_{ANORMAL}.jpg', bbox_inches='tight', pad_inches=0)