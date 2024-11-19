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


from model import ConvDiscriminator, ConvGenerator
from utils import get_dataset_by_digit

DEVICE="mps"

transform = Compose([
    ToTensor(),
    Normalize(0.5, 0.5)
])

trainset = MNIST(root="../../../coding/Dataset/", train=True, download=True, transform=transform)
testset = MNIST(root="../../../coding/Dataset/", train=False, download=True, transform=transform)

EPOCHS=50
BATCH_SIZE=128
LR=0.0002

LATENT_DIM=100

beta_1 = 0.5
beta_2 = 0.999

train_size = 40000
val_size = len(trainset) - train_size

generator = torch.Generator().manual_seed(42)
trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)

train_dic_dataset = get_dataset_by_digit(trainset)
val_dic_dataset = get_dataset_by_digit(valset)
test_dic_dataset = get_dataset_by_digit(testset)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


for digit in range(10):

  print(f"digit {digit}")

  ANORMAL= digit
  NORMAL_DIGITS = [i for i in range(10)]
  NORMAL_DIGITS.remove(ANORMAL)

  normal_trainset = torch.cat([train_dic_dataset[i] for i in NORMAL_DIGITS])
  normal_val = torch.cat([val_dic_dataset[i] for i in NORMAL_DIGITS])

  trainloader = DataLoader(normal_trainset, batch_size=BATCH_SIZE, shuffle=True)

  disc = ConvDiscriminator(im_channel=1, hidden_dim=64).to(DEVICE)
  gen = ConvGenerator(z_dim=LATENT_DIM, hidden_dim=64).to(DEVICE)

  gen = gen.apply(weights_init)
  disc = disc.apply(weights_init)

  optim_gen = optim.Adam(gen.parameters(), lr=LR, betas=(beta_1, beta_2))
  optim_disc = optim.Adam(disc.parameters(), lr=LR, betas=(beta_1, beta_2))

  criterion_disc = nn.BCEWithLogitsLoss()
  criterion_gen = nn.BCEWithLogitsLoss()

  for epoch in tqdm(range(EPOCHS)):
      epoch_loss_gen = 0
      epoch_loss_disc = 0

      for i, inputs in enumerate(trainloader):

          inputs = inputs.to(DEVICE)
          batch_size = inputs.size(0)

          # Discriminator

          optim_disc.zero_grad()

          ones = torch.ones(batch_size, 1).to(DEVICE)

          pred_disc_true = disc(inputs)
          loss_disc_true = criterion_disc(pred_disc_true, ones)


          zeros = torch.zeros(batch_size, 1).to(DEVICE)

          z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
          fake = gen(z)
          pred_disc_false = disc(fake.detach())
          loss_disc_fake = criterion_disc(pred_disc_false, zeros)

          loss_disc_true.backward()
          loss_disc_fake.backward()

          loss_disc = (loss_disc_true + loss_disc_fake) /2
          optim_disc.step()

          # Generator :
          optim_gen.zero_grad()

          pred_disc_false = disc(fake)
          loss_gen = criterion_gen(pred_disc_false, ones)

          loss_gen.backward()
          optim_gen.step()

          epoch_loss_gen+=loss_gen.item()
          epoch_loss_disc+=loss_disc_fake.item() + loss_disc_true.item()

  z = torch.randn(100, LATENT_DIM).to(DEVICE)
  with torch.no_grad():
      generated = gen(z)
  gen_images = generated.reshape(-1, 1, 28, 28)
  grid_image = make_grid(gen_images, nrow=10, normalize=True)

  plt.imshow(grid_image.permute(1, 2, 0).detach().cpu())
  plt.axis('off')
  plt.savefig(f'figures/Generated_Anomaly_{ANORMAL}.jpg', bbox_inches='tight', pad_inches=0)
  plt.close()

  checkpoint = {
      "gen_state_dict":gen.state_dict(),
      "disc_state_dict":disc.state_dict()
  }
  torch.save(checkpoint, f"checkpoints/CGAN_checkpoint_{ANORMAL}.pkl")
    