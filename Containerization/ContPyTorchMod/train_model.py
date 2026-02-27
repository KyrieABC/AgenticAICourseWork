import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Simple pretrained model
model = models.resnet18(pretrained=True)
model.fc=nn.Linear(model.fc.in_features,10) #10 classes

#Fake training: just save weights for demo
torch.save(model,"model.pt")
print("Model saved as model.pt")
