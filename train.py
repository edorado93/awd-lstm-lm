import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from title_data import TitlesAndAbstracts
from gensim.models import KeyedVectors
import model

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=1,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--pretrained', type=str, default=None,
                    help='Pretrained word embeddings file to use')
parser.add_argument('--encoder', type=str, default="BOW",
                    help="Encoder model to use (LSTM, BOW)")
parser.add_argument('--title_abstract_concat', action='store_true',
                    help="Should tie title's word embedding with each word of the abstract before decoding.")
parser.add_argument('--title_abstract_concat_type', type=str, default="sum",
                    help="Type of concatenation between title's word embedding and words in the abstract. (sum, mean, learned)")
args = parser.parse_args()

print("Arguments entered are:", args)

if args.title_abstract_concat and args.encoder == "BOW":
    raise Exception("Cannot concatenate title abstract word embeddings in BOW model")

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

titles_abstracts_corpus = TitlesAndAbstracts(args.data)

""" Title-Abstracts Model Parameters START Here ........................... """

eval_batch_size = 10
test_batch_size = 1
title_train, title_valid, title_test, abstracts_train, abstracts_valid, abstracts_test = titles_abstracts_corpus.cudify(args.batch_size)
num_tokens = len(titles_abstracts_corpus.dictionary.word2idx.keys())
nlg_model = model.Seq2Seq(args, num_tokens)
if args.cuda:
    nlg_model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in nlg_model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)

""" Title-Abstracts Model Parameters END Here ........................... """

criterion = nn.CrossEntropyLoss()

def evaluate(titles, abstracts, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    nlg_model.eval()
    if args.model == 'QRNN': nlg_model.reset()
    total_loss = 0
    for i in range(0, len(titles)):
        title, abstract =  titles[i], abstracts[i] # Considering one at a time. No batching here
        output, hidden = nlg_model(title, abstract)
        targets = Variable(abstract[1:].view(-1))
        output_flat = output.view(-1, num_tokens)
        total_loss += criterion(output_flat, targets).data


    return total_loss[0] / len(titles)

def load_word_embeddings(model, filename, word2index):
    print("Copying pretrained word embeddings from ", filename)
    en_model = KeyedVectors.load_word2vec_format(filename)
    """ Fetching all of the words in the vocabulary. """
    pretrained_words = set()
    for word in en_model.vocab:
        pretrained_words.add(word)

    arr = [0] * len(word2index)
    for word in word2index:
        index = word2index[word]
        if word in pretrained_words:
            arr[index] = en_model[word]
        else:
            arr[index] = np.random.uniform(-1.0, 1.0, args.emsize)

    """ Creating a numpy dictionary for the index -> embedding mapping """
    arr = np.array(arr)
    """ Add the word embeddings to the empty PyTorch Embedding object """
    model.load_word_embeddings(torch.from_numpy(arr))
if args.pretrained:
    load_word_embeddings(nlg_model, args.pretrained, titles_abstracts_corpus.dictionary.word2idx)

def train():
    global title_train
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': nlg_model.reset()
    total_loss = 0
    start_time = time.time()
    batch, i = 0, 0
    loss = None; assign_loss = True
    while i < len(title_train):
        seq_len = 1

        if batch > 0 and batch % args.batch_size == 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm(nlg_model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            assign_loss = True

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        nlg_model.train()
        title, abstract =  title_train[i], abstracts_train[i] # One at a time
        targets = Variable(abstract[1:].view(-1))

        output, hidden, rnn_hs, dropped_rnn_hs = nlg_model(title, abstract, return_h=True)
        raw_loss = criterion(output.view(-1, num_tokens), targets)

        loss = raw_loss if assign_loss else (loss + raw_loss)
        # Activiation Regularization
        loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(title_train) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += 1

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(nlg_model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in nlg_model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(title_valid, abstracts_valid)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2) if val_loss2 <= 10 else -1), flush=True)
            print('-' * 89)

            if val_loss2 < stored_loss:
                with open(args.save, 'wb') as f:
                    torch.save(nlg_model, f)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in nlg_model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(title_valid, abstracts_valid, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss) if val_loss <= 10 else -1), flush=True)
            print('-' * 89)

            if val_loss < stored_loss:
                with open(args.save, 'wb') as f:
                    torch.save(nlg_model, f)
                print('Saving Normal!')
                stored_loss = val_loss

            if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching!')
                optimizer = torch.optim.ASGD(nlg_model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                #optimizer.param_groups[0]['lr'] /= 2.
            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    nlg_model = torch.load(f)

# Run on test data.
test_loss = evaluate(title_test, abstracts_test, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

