from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.indexer import RuPosIndexer
from src.reader import Sentence


class RuPosDataset(Dataset):
    """
    Торчевый датасет, который возвращает тензор по индексу предложения
    Для того, чтобы с ним моглиработать торчевые итераторы, нужно реализовать:
    __len__ - получение длины датасета
    __getitem__ - получение тензора по индексу
    collate_fn - получение тензора для батча
    """
    def __init__(self, sentences: List[Sentence], indexer: RuPosIndexer, device: torch.device):
        self.sentences = sentences
        self.indexer = indexer
        self.device = device

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        tokens, pos_tags, grammemes = self.indexer.sentence_to_indexes(self.sentences[item])
        # просто переводим индексы в торчевые
        return torch.LongTensor(tokens), torch.LongTensor(pos_tags), torch.LongTensor(grammemes)

    def collate_fn(self, batch):
        tokens, pos_tags, grammemes = list(zip(*batch))
        # не забываем про паддинги
        tokens = pad_sequence(tokens, batch_first=True).to(self.device)
        pos_tags = pad_sequence(pos_tags, batch_first=True).to(self.device)
        grammemes = pad_sequence(grammemes, batch_first=True).to(self.device)

        return tokens, pos_tags, grammemes


