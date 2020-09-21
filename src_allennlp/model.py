from typing import Dict

import torch
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules import FeedForward
from allennlp.nn import Activation
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import FBetaMeasure, CategoricalAccuracy


@Model.register("SimpleTagger")
class SimpleTagger(Model):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2SeqEncoder, dropout: float = 0.1,
                 ff_dim: int = 100):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder

        assert self.embedder.get_output_dim() == self.encoder.get_input_dim()

        self.feedforward = FeedForward(encoder.get_output_dim(), 1, hidden_dims=100,
                                       activations=Activation.by_name('relu')(), dropout=dropout )
        self.out = torch.nn.Linear(in_features=self.feedforward.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.f1 = FBetaMeasure(average='micro')
        self.accuracy = CategoricalAccuracy()
        self.idx_to_label = vocab.get_index_to_token_vocabulary('labels')

    def forward(self, tokens: Dict[str, torch.Tensor], tags: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.embedder(tokens)
        encoder_out = self.encoder(embeddings, mask)
        encoder_out = self.feedforward(encoder_out)
        logits = self.out(encoder_out)
        output = {
            "logits": logits,
            "mask": mask
        }
        if tags is not None:
            self.accuracy(logits, tags, mask)
            self.f1(logits, tags, mask)
            output['loss'] = sequence_cross_entropy_with_logits(logits, tags, mask)
        return output

    def decode(self, output_dict: Dict[str, torch.Tensor]):

        logits = output_dict["logits"]
        mask = output_dict["mask"]
        tag_logits = torch.argmax(logits, dim=2).tolist()
        lengths = torch.sum(mask, dim=1).tolist()
        all_labels = []
        for sample_num in range(len(tag_logits)):
            labels = []
            for label_idx in range(lengths[sample_num]):
                labels.append(self.idx_to_label[tag_logits[sample_num][label_idx]])
            all_labels.append(labels)
        return {"labels": all_labels}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        out = {"accuracy": self.accuracy.get_metric(reset)}
        out.update(self.f1.get_metric(reset))
        return out