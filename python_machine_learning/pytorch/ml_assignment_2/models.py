from typing import Tuple
from lightning.pytorch import LightningModule
from torch import nn
import torch
from torch import Tensor
import torchmetrics


class BaseModel(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=num_classes)
        self.model = self.build_model()

    def build_model(self):
        raise Exception("Not yet implemented")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.model(x)

    def loss(self, logits, target):
        return nn.functional.cross_entropy(logits, target)

    def shared_step(self, mode: str, batch: Tuple[Tensor, Tensor], batch_index: int):
        x, target = batch
        output = self.forward(x)
        loss = self.loss(output, target)
        self.accuracy(output, target)
        self.log(f"{mode}_step_acc", self.accuracy, prog_bar=True)
        self.log(f"{mode}_step_loss", loss, prog_bar=False)
        return loss

    def training_step(self, batch, batch_index):
        return self.shared_step('train', batch, batch_index)

    def validation_step(self, batch, batch_index):
        return self.shared_step('val', batch, batch_index)

    def test_step(self, batch, batch_index):
        return self.shared_step('test', batch, batch_index)


class LinearModel(BaseModel):
    def build_model(self):
        return nn.Linear(784, self.num_classes)

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        return self.model(inputs)


class MLP(BaseModel):
    def __init__(self, classes, hidden_units):
        self.hidden_units = hidden_units
        super().__init__(classes)

    def build_model(self):
        return nn.Sequential(
            nn.Linear(784, self.hidden_units),
            nn.ReLU(),
            nn.Linear(self.hidden_units, self.num_classes)
        )

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        return self.model(inputs)


class ConvNet(BaseModel):
    def __init__(self, classes, filters, kernel, pool):
        self.filters = filters
        self.kernel = kernel
        self.pool = pool
        super().__init__(classes)

    def build_model(self):
        return nn.Sequential(
            nn.Conv2d(1, self.filters, kernel_size=self.kernel),
            nn.ReLU(),
            nn.MaxPool2d(self.pool),
            nn.Flatten(),
            nn.Linear(self.filters * ((28 - self.kernel + 1) // self.pool) ** 2, self.num_classes)
        )


class DeepConvNet(BaseModel):
    def build_model(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )