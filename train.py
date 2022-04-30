#%%
import torch
from torchvision import models as tv_models
from lightly import models, loss

path = './danbooru-images/danbooru-images'

# %%
import pytorch_lightning as pl

# %%
from torchvision.transforms import (
    Compose, 
    RandomRotation, 
    RandomResizedCrop, 
    ToTensor, 
    Normalize, 
    RandomApply, 
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
)
from torch.utils.data import DataLoader
from danbooru import Danbooru, BiaoQingBao, DianZiBaoJiang, GaussianBlur, JPEGCompression
from torch import nn

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = Compose([
    RandomApply([BiaoQingBao()], p=0.5),
    RandomApply([DianZiBaoJiang()], p=0.5),
    RandomApply([JPEGCompression()], p=0.5),
    RandomGrayscale(),
    RandomApply([
        ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.5),
    # RandomApply([RandomRotation(degrees=20)], p=0.1),
    RandomResizedCrop((224, 224), scale=(0.08, 0.5)),
    RandomApply([BiaoQingBao((20, 30))], p=0.1),
    RandomApply([DianZiBaoJiang((4, 8))], p=0.1),
    RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize
])

pure_transform = Compose([
    RandomApply([
        ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.5),
    RandomGrayscale(),
    RandomResizedCrop((224, 224)),
    RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize
])

# %%
dataset = Danbooru(path, transform, pure_transform)
train_loader = DataLoader(dataset, 64, shuffle=True, num_workers=6)
max_epochs = 5

from lightly.models.modules.heads import SimCLRProjectionHead

class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = tv_models.resnet34()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 512)

        self.criterion = loss.NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        x0, x1 = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        # optim = torch.optim.AdamW(self.parameters(), lr=1e-4)
        # return [optim]
        optim = torch.optim.SGD(
            self.parameters(), lr=8e-2, momentum=0.9, weight_decay=5e-4
        )
        return [optim]
#%%
model = SimCLRModel()
#%%
trainer = pl.Trainer(
    max_epochs=max_epochs, 
    gpus=1, 
)
trainer.fit(model=model, train_dataloaders=train_loader)
# %%
import time
torch.save(nn.Sequential(
    model.backbone,
    nn.Flatten(start_dim=1),
    model.projection_head
), f"simclr_{int(time.time())}.pth")
# %%
