import pytorch_lightning as pl
import torch

from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import coral_loss
from efficientnet_pytorch import EfficientNet
from torch import nn, optim

class OrdRegressor(pl.LightningModule):

    def __init__(self, num_classes, lr=0.001) -> None:
        super().__init__()
        self.features = EfficientNet.from_pretrained(
                                model_name='efficientnet-b0',
                                weights_path='models\efficientnet_b0.pth', 
                                include_top=False)
        
        self.fc = CoralLayer(size_in=1280, num_classes=num_classes)
        self.learning_rate = lr
        self.num_classes = num_classes
        self.save_hyperparameters()

    def forward(self, x):
        x = self.features(x).squeeze()
        logits =  self.fc(x)
        probas = torch.sigmoid(logits)
        return logits, probas

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def shared_step(self, batch):
        x, y = batch
        levels = levels_from_labelbatch(y, num_classes=self.num_classes)
        logits, probas = self(x)
        loss = coral_loss(logits, levels)
        
        predicted_labels = proba_to_label(probas).float()

        n_samples = x.shape[0]
        mae = torch.sum(torch.abs(predicted_labels - y))
        mse = torch.sum((predicted_labels - y)**2)
        return loss, mae, mse, n_samples

    def shared_epoch(self, outputs):
        total_samples = sum([o['n_samples'] for o in outputs])
        total_mae = sum([o['mae'] for o in outputs]) / total_samples
        total_mse = sum([o['mae'] for o in outputs]) / total_samples
        total_loss = sum([o['loss'] for o in outputs]) / total_samples
        return total_loss, total_mae, total_mse

    def training_step(self, batch, batch_idx):
        loss, mae, mse, n_samples = self.shared_step(batch)
        return {'loss': loss, 'mae': mae, 'mse':mse, 'n_samples': n_samples}

    def training_epoch_end(self, outputs) -> None:
        total_loss, total_mae, total_mse = self.shared_epoch(outputs)
        self.log('train_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_mae', total_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_mse', total_mse, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        loss, mae, mse, n_samples = self.shared_step(batch)
        return {'loss': loss, 'mae': mae, 'mse':mse, 'n_samples': n_samples}

    def training_epoch_end(self, outputs) -> None:
        total_loss, total_mae, total_mse = self.shared_epoch(outputs)
        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mae', total_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mse', total_mse, prog_bar=True, on_step=False, on_epoch=True)