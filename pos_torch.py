import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

from src_torch import RuPosReader, RuPosIndexer, RuPosDataset, load_embeddings, SimpleTagger, Trainer

if __name__ == '__main__':
    device = torch.device('cuda')
    reader = RuPosReader()
    indexer = RuPosIndexer()
    sentences = reader.read('data/train.csv')
    indexer.index_dataset(sentences)

    train_sentences = sentences[:-1000]
    dev_sentences = sentences[-1000:]

    train_dataset = RuPosDataset(train_sentences, indexer, device)
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=256, sampler=train_sampler,
                                collate_fn=train_dataset.collate_fn)

    dev_dataset = RuPosDataset(dev_sentences, indexer, device)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_iterator = DataLoader(dev_dataset, batch_size=256, sampler=dev_sampler,
                              collate_fn=dev_dataset.collate_fn)

    embeddings = load_embeddings(indexer.token_vocab, 'data/cc.ru.300.vec')
    model = SimpleTagger(output_dim=len(indexer.pos_vocab), embedding_matrix=embeddings)
    model.to(device)

    trainer = Trainer(model, train_iterator, dev_iterator)

    for i in range(20):
        print('Epoch: %d' % (i + 1))
        trainer.train_epoch()
        trainer.test_epoch()
