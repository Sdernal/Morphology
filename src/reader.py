from typing import List


class Sentence:
    """
    Контейнер для хранения токенов и тэгов в сыром виде
    """
    def __init__(self):
        self.tokens = []  # type: List[str]
        self.pos_tags = []  # type: List[str]
        self.grammems = []  # type: List[str]


class RuPosReader:
    """
    Считывает файл с датасетом и возвращает список "предложений"
    """
    def __init__(self):
        pass

    def read(self, data_file: str) -> List[Sentence]:
        sentences = []  # type: List[Sentence]
        with open(data_file, 'r', encoding='utf-8') as f:
            next(f)  # первая строка для названий колонок
            sentence = Sentence()
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    split = line.split('\t')
                    if len(split) == 4:
                        # для train.csv
                        _, _, token, gram = split
                        pos, gram = gram.split('#')
                    else:
                        # для test.csv
                        _, _, token = split
                        pos, gram = '<UNK>', '<UNK>'
                    sentence.tokens.append(token)
                    sentence.pos_tags.append(pos)
                    sentence.grammems.append(gram)
                else:
                    if len(sentence.tokens) > 0:
                        sentences.append(sentence)
                        sentence = Sentence()

            if len(sentence.tokens) > 0:
                sentences.append(sentence)

        return sentences
