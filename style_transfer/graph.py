import torch
import torch.nn as nn
from vgg import vgg16
import os

from torchvision import datasets, transforms, utils as tvutils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def transform_back(data, save_dir):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.detach().clone().cpu().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(save_dir)


vgg = vgg16()

transform_style = transforms.Compose([  
    transforms.Resize(600),           
    transforms.CenterCrop(600),      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(512),           
    transforms.CenterCrop(512),      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



device = torch.device('mps')

clock = Image.open("./test_image/clock.jpg")
clock = transform_test(clock)
clock = clock.to(device)
transform_back(clock, "./report_image/clock.jpg")

cake = Image.open("./test_image/cake.jpg")
cake = transform_test(cake)
cake = cake.to(device)
transform_back(cake, "./report_image/cake.jpg")

man = Image.open("./test_image/man.jpg")
man = transform_test(man)
man = man.to(device)
transform_back(man, "./report_image/man.jpg")
