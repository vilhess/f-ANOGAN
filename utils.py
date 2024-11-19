import torch 
from torchvision.utils import make_grid

def get_dataset_by_digit(dataset):
    dic = {i:[] for i in range(10)}
    for img, lab in dataset:
        dic[lab].append(img)
    dic = {i:torch.stack(dic[i]) for i in range(10)}
    return dic