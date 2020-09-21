from typing import Dict, List, Iterator

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField


@DatasetReader.register("RuPosReader")
class RuPosReader(DatasetReader):
    """
    Считывает файл с датасетом и возвращает список "предложений"
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, lazy=False) -> None:
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(lowercase_tokens=True)
        }

    def text_to_instance(self, tokens: List[Token], tags: List[str]) -> Instance:
        if tags:
            assert len(tokens) == len(tags)
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["tags"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r', encoding='utf-8') as f:
            columns = next(f).strip().split("\t")  # первая строка для названий колонок
            tokens, labels = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if tokens:
                        yield self.text_to_instance(tokens, labels)
                    tokens, labels = [], []
                    continue
                if len(columns) == 4:
                    # для train.csv
                    _, _, word, gram = line.split('\t')
                    pos, _ = gram.split('#')
                    labels.append(pos)
                else:
                    # для test.csv
                    _, _, word = line.split('\t')
                tokens.append(Token(word))

            if tokens:
                yield self.text_to_instance(tokens, labels)
