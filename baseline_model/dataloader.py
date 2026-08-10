import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

#Preprocessing data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

#Generate the ASL dataset
asl_data = datasets.ImageFolder(
    root = "data/raw",
    transform=transform
)

#Split sizes 80/10/10
train_size = int(0.8*len(asl_data))
val_size = int(0.1*len(asl_data))
test_size = len(asl_data) - train_size - val_size

#Generate training, validation, and test sets
train_data, val_data, test_data = random_split(
    asl_data,
    [train_size, val_size, test_size]
)

#DataLoaders for training, validation, and test sets
train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_data, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_data, batch_size = 32, shuffle = True)
