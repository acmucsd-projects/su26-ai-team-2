import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
 
# Letters that involve motion and can't be represented by a single static
# frame get excluded from the dataset entirely.
EXCLUDED_CLASSES = ["J", "Z"]
 
 
class FilteredImageFolder(datasets.ImageFolder):
    """ImageFolder that skips a set of class subfolders entirely.
 
    Overriding find_classes() means the excluded folders are never scanned
    or added to samples/targets, so the resulting dataset simply has no
    knowledge that J and Z ever existed (rather than filtering them out
    after the fact, which would leave gaps in the class index mapping).
    """
    def __init__(self, root, transform=None, exclude_classes=None):
        self.exclude_classes = set(exclude_classes or [])
        super().__init__(root, transform=transform)
 
    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        classes = [c for c in classes if c not in self.exclude_classes]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
 
 
#Preprocessing data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])
 
#Generate the ASL dataset (J and Z excluded — see EXCLUDED_CLASSES above)
asl_data = FilteredImageFolder(
    root = "data/raw",
    transform=transform,
    exclude_classes=EXCLUDED_CLASSES
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
 
