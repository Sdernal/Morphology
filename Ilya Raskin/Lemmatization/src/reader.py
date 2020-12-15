import os
from conllu import parse
from typing import Dict, List, Iterator

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField


class DependencyReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, target_indexers: Dict[str, TokenIndexer] = None, lazy=False) -> None:
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(lowercase_tokens=True)
        }
        self.target_indexers = target_indexers or {
            "target": SingleIdTokenIndexer(lowercase_tokens=True)
        }

    def text_to_instance(self, tokens: List[Token], target: List[Token]) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        target_field = TextField(target, self.target_indexers)
        
        fields = {'source_tokens': sentence_field, 'target_tokens': target_field}
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r', encoding='utf-8') as f:
            parsed = parse(f.read())
        for i in range(len(parsed)):
            sentence = [dict(word) for word in parsed[i]]
            for j in range(len(sentence)):
                context = []
                if j > 1:
                    context += [char for char in sentence[j - 2]['form']] + ['#']
                if j > 0:
                    context += [char for char in sentence[j - 1]['form']]
                context += ['##'] + [char for char in sentence[j]['form']] + ['##']
                if j < len(sentence) - 1:
                    context += [char for char in sentence[j + 1]['form']]
                if j < len(sentence) - 2:
                    context += ['#'] + [char for char in sentence[j + 2]['form']]
                context += [sentence[j]['upos'], sentence[j]['xpos'] if sentence[j]['xpos'] else '*']
                if sentence[j]['feats']:
                    context += [sentence[j]['feats'][k] for k in sentence[j]['feats']]
                context = [Token(i) for i in context]
                target = [Token(char) for char in sentence[j]['lemma']]
                yield self.text_to_instance(context, target)
