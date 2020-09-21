import torch
from tqdm import tqdm

from src_torch.vocab import Vocab


def load_embeddings(tokens_vocab: Vocab, embeddings_file: str):
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        vocab_size, embeddings_size = next(f).split()
        print('Vocab size: %s\tEmbeddings size: %s' % (vocab_size, embeddings_size))
        embeddings_size = int(embeddings_size)
        embeddings_matrix = torch.rand((len(tokens_vocab), embeddings_size))
        paddings = torch.zeros(embeddings_size)
        embeddings_matrix[0] = paddings
        for line in tqdm(f):
            word, *weights = line.split()
            if word in tokens_vocab:
                weights = torch.FloatTensor(list(map(float, weights)))
                embeddings_matrix[tokens_vocab[word]] = weights

    return embeddings_matrix
