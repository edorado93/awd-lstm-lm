import argparse
import torch
from torch.autograd import Variable
import numpy as np
from title_data import TitlesAndAbstracts
import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--title', type=str, default=None,
                    help='The title for which the abstract is to be generated')
parser.add_argument('--encoder', type=str, default="BOW",
                    help='The encoder model that was used during training')

args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)

model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

titles_abstracts_corpus = TitlesAndAbstracts(args.data)

encoder = model.encoder
decoder = model.decoder
title_tensor = titles_abstracts_corpus.tokenize_test_title(args.title)
ntokens = len(titles_abstracts_corpus.dictionary.word2idx.keys())
input_tensor = torch.rand(1, 1).mul(ntokens).long()

if args.cuda:
    title_tensor = title_tensor.cuda()
    input_tensor = input_tensor.cuda()

input = Variable(input_tensor, volatile=True)


if args.encoder == "LSTM":
    encoder_output, hidden_context = encoder(Variable(title_tensor.view(len(title_tensor), -1)), encoder.init_hidden(1), return_h=False)
    h1, h2 = hidden_context[-1]
    hidden_context_BOW = decoder.init_hidden(1)
    hidden_context_BOW[0] = (h1, h2)
else:
    hidden_context_BOW = encoder(title_tensor)
    hidden_context_BOW = decoder.package_hidden(1, hidden_context_BOW)

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        output, hidden_context_BOW = decoder(input, hidden_context_BOW)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = titles_abstracts_corpus.dictionary.idx2word[word_idx]

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
