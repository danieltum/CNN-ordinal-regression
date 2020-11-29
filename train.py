from argparse import ArgumentParser
from pandas.core.algorithms import mode

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from modeling import OrdRegressor
from data import CustomDataModule


def main(args):
    seed_everything(seed=42)
    dm = CustomDataModule(dims=(224,224), batch_size=32)
    dm.prepare_data()
    dm.setup()

    model = OrdRegressor(lr=0.002, num_classes = dm.num_classes)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath='checkpoints')
    early_stopping = EarlyStopping(moinitor='val_loss', mode='min', patience=5)
    trainer = Trainer(max_epochs=2, callbacks=[checkpoint_callback, early_stopping],
                      auto_lr_find=True)

    trainer.tune(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
