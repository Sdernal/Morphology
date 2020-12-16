from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import FloatTensor, LongTensor

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

def do_epoch(model, criterion, data_iter, optimizer=None, name=None):
    epoch_loss = 0
    
    is_train = not optimizer is None
    name = name or ''
    model.train(is_train)
    
    batches_count = len(data_iter)
    
    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, batch in enumerate(data_iter):                
                logits, _, _ = model(batch.source[0], batch.source[1], batch.target)
                
                target = torch.cat((batch.target[1:], batch.target.new_ones((1, batch.target.shape[1]))))
                loss = criterion(logits.view(-1, logits.shape[-1]), target.view(-1))

                epoch_loss += loss.item()

                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(name, loss.item(), 
                                                                                         math.exp(loss.item())))
                
            progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                name, epoch_loss / batches_count, math.exp(epoch_loss / batches_count))
            )
            progress_bar.refresh()
    return epoch_loss / batches_count


def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None):
    best_val_loss = None
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss = do_epoch(model, criterion, train_iter, optimizer, name_prefix + 'Train:')
        
        if not val_iter is None:
            val_loss = do_epoch(model, criterion, val_iter, None, name_prefix + '  Val:')
            print('\nVal BLEU = {:.2f}'.format(evaluate_model(model, val_iter)))
            
            
def evaluate_model(model, iterator):
    model.eval()
    refs, hyps = [], []
    bos_index = iterator.dataset.fields['target'].vocab.stoi[BOS_TOKEN]
    eos_index = iterator.dataset.fields['target'].vocab.stoi[EOS_TOKEN]
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            encoder_output, encoder_hidden = model.encoder(batch.source[0], batch.source[1])
            mask = batch.source[0] == 1.
            
            output = encoder_output
            hidden = encoder_hidden
            result = [LongTensor([bos_index]).expand(1, batch.target.shape[1])]
            
            for _ in range(30):
                step, hidden, _ = model.decoder(result[-1], encoder_output, mask, encoder_hidden, output, hidden)
                step = step.argmax(-1)
                result.append(step)
            
            targets = batch.target.data.cpu().numpy().T
            eos_indices = (targets == eos_index).argmax(-1)
            eos_indices[eos_indices == 0] = targets.shape[1]

            targets = [target[:eos_ind] for eos_ind, target in zip(eos_indices, targets)]
            refs.extend(targets)
            
            result = torch.cat(result)
            result = result.data.cpu().numpy().T
            eos_indices = (result == eos_index).argmax(-1)
            eos_indices[eos_indices == 0] = result.shape[1]

            result = [res[:eos_ind] for eos_ind, res in zip(eos_indices, result)]
            hyps.extend(result)
            
    return corpus_bleu([[ref] for ref in refs], hyps) * 100