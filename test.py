import os

dataset_path = os.path.expanduser("~/Documents/Kaggle/asl_dataset/asl_dataset")

class_names = sorted(os.listdir(dataset_path))
print(class_names)

print(len(class_names))
print(class_names[35])