import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src_torch import RuPosReader, RuPosIndexer, RuPosDataset, load_embeddings, SimpleTagger


class Tagger(pl.LightningModule):
    def __init__(self, data_path: str, embeddings_path: str, batch_size: int = 256):
        super().__init__()
        reader = RuPosReader()
        self.indexer = RuPosIndexer()
        sentences = reader.read(data_path)
        self.indexer.index_dataset(sentences)
        embeddings = load_embeddings(self.indexer.token_vocab, embeddings_path)
        self.train_sentences = sentences[:-1000]
        self.dev_sentences = sentences[-1000:]
        self.batch_size = batch_size
        self.model = SimpleTagger(output_dim=len(self.indexer.pos_vocab), embedding_matrix=embeddings)
        self.accuracy = torchmetrics.Accuracy()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        pred = self.forward(x)
        loss = self.calculate_loss(pred, y)
        acc = self.accuracy(pred.view(-1, pred.size(-1)), y.view(-1))
        self.log('train.loss', loss)
        self.log('train.acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        pred = self.forward(x)
        loss = self.calculate_loss(pred, y)
        acc = self.accuracy(pred.view(-1, pred.size(-1)), y.view(-1))
        self.log('val.loss', loss)
        self.log('val.acc', acc)
        return loss

    def calculate_loss(self, pred, target):
        loss = self.criterion( pred.view(-1, pred.size(-1)), target.view(-1))
        return loss

    def train_dataloader(self):
        train_dataset = RuPosDataset(self.train_sentences, self.indexer, device)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                            collate_fn=train_dataset.collate_fn)
        return loader

    def val_dataloader(self):
        test_dataset = RuPosDataset(self.dev_sentences, self.indexer, device)
        loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                            collate_fn=test_dataset.collate_fn)
        return loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    device = torch.device('cuda')
    system = Tagger(data_path='data/train.csv', embeddings_path='data/cc.ru.300.vec')
    wandb_logger = WandbLogger(name='pl', project='rupos2018')
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val.acc',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback], max_epochs=20, gpus=1)
    trainer.fit(system, system.train_dataloader(), system.test_dataloader())

