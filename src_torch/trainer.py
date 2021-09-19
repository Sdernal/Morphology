import torch
import torch.nn as nn
import wandb


class Trainer:
    def __init__(self, model: nn.Module, train_iterator, dev_iterator, lr=2e-5):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
        self.model = model

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (tokens, pos_tags, _) in enumerate(self.train_iterator):
            self.optimizer.zero_grad()
            logits = self.model(tokens)
            loss = self.criterion(logits.view(-1, logits.size(-1)), pos_tags.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            mask = (tokens != 0).to(torch.long)
            pred = torch.argmax(logits, dim=-1)
            correct += ((pred == pos_tags)*mask).sum().item()
            total += mask.sum().item()

            print('\rLoss: %4f, Accuracy: %4f, Batch: %d of %d' % (
                total_loss / (batch_idx + 1), correct / total, batch_idx + 1, len(self.train_iterator)
            ), end='')
        print()
        loss, accuracy = total_loss / (batch_idx + 1), correct / total
        return loss, accuracy

    def test_epoch(self):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            total = 0
            correct = 0
            for batch_idx, (tokens, pos_tags, _) in enumerate(self.dev_iterator):
                logits = self.model(tokens)
                loss = self.criterion(logits.view(-1, logits.size(-1)), pos_tags.view(-1))
                total_loss += loss.item()

                mask = (tokens != 0).to(torch.long)
                pred = torch.argmax(logits, dim=-1)
                correct += ((pred == pos_tags) * mask).sum().item()
                total += mask.sum().item()

                print('\rLoss: %4f, Accuracy: %4f, Batch: %d of %d' % (
                    total_loss / (batch_idx + 1), correct / total, batch_idx + 1, len(self.dev_iterator)
                ), end='')
            print()
            loss, accuracy = total_loss / (batch_idx + 1), correct / total
            return loss, accuracy

    def fit(self, max_epochs: int = 20):
        for epoch in range(max_epochs):
            train_loss, train_accuracy = self.train_epoch()
            test_loss, test_accuracy = self.test_epoch()
            wandb.log({
                'train': {
                    'loss': train_loss,
                    'acc': train_accuracy
                },
                'val': {
                    'loss': test_loss,
                    'acc': test_accuracy
                },
                'epoch': epoch
            })

