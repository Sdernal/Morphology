from typing import List, Tuple

from src.reader import Sentence
from src.vocab import Vocab


class RuPosIndexer:
    """Индексирует датасет и хранит словари"""
    def __init__(self):
        self.token_vocab = Vocab(lowercase=True, paddings=True)
        self.pos_vocab = Vocab(paddings=True)
        self.gram_vocab = Vocab(paddings=True)

    def index_dataset(self, dataset: List[Sentence]):
        """
        Заполняет словари по датасету
        """
        for sentence in dataset:
            self.token_vocab.fill(sentence.tokens)
            self.pos_vocab.fill(sentence.pos_tags)
            self.gram_vocab.fill(sentence.grammems)

    def sentence_to_indexes(self, sentence: Sentence) -> Tuple[List[int], List[int], List[int]]:
        """
        Переводит предложение в индексы
        """
        tokens = [self.token_vocab[token] for token in sentence.tokens]
        pos_tags = [self.pos_vocab[pos] for pos in sentence.pos_tags]
        grammemes = [self.gram_vocab[gram] for gram in sentence.grammems]

        return tokens, pos_tags, grammemes