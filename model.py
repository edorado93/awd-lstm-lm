import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    """ Assumes that "hidden" is a Variable containing a tensor of dimension (1 * X). Basically this is the output of the encoder. """
    def package_hidden(self, bsz, hidden):
        hidden_states_default = self.init_hidden(bsz)
        hidden_states_default[0] = (hidden.view(1, bsz, hidden.size(1)), hidden.view(1, bsz, hidden.size(1)))
        return hidden_states_default

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]

class Encoder(nn.Module):
    def __init__(self, embeddings, emdim, hidden_size, is_cuda):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(emdim, hidden_size)
        self.tanh = nn.Tanh()
        self.embeddings = embeddings
        self.is_cuda = is_cuda

    def forward(self, sentence):
        x = self.combine_word_embeddings(sentence)
        out = self.fc1(x)
        out = self.tanh(out)
        return out

    def combine_word_embeddings(self, sentence):
        embeddings = [self.embeddings(Variable((torch.LongTensor([w])).cuda()) if self.is_cuda else Variable(torch.LongTensor([w]))) for w in sentence]
        stacked_embedding = torch.cat(embeddings)
        return torch.mean(stacked_embedding, 0, True)


class Seq2Seq(nn.Module):
    def __init__(self, rnn_type, encoder, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1,
                 wdrop=0, tie_weights=False, cuda=False):
        super(Seq2Seq, self).__init__()
        self.endoder_model = encoder
        if encoder == "BOW":
            self.decoder = RNNModel(rnn_type, ntoken, ninp, nhid, nlayers, dropout, dropouth, dropouti, dropoute, wdrop,
                                    tie_weights)
            self.encoder = Encoder((self.decoder).encoder, ninp, nhid, cuda)
        else:
            self.decoder = RNNModel(rnn_type, ntoken, ninp, nhid, nlayers, dropout, dropouth, dropouti, dropoute, wdrop,
                                    tie_weights)
            self.encoder = RNNModel(rnn_type, ntoken, ninp, nhid, nlayers, dropout, dropouth, dropouti, dropoute, wdrop,
                                tie_weights=False)

    def load_word_embeddings(self, new_embeddings):
        self.decoder.encoder.weight.data.copy_(new_embeddings)

    def forward(self, title, abstract, return_h=False):
        if self.endoder_model == "LSTM":
            encoder_output, hidden_context = self.encoder(Variable(title.view(len(title), -1)), self.encoder.init_hidden(1), return_h=False)
            h1, h2 = hidden_context[-1]
            hidden_layer = self.decoder.init_hidden(1)
            hidden_layer[0] = (h1, h2)
        else:
            hidden_layer = self.encoder(title)
            hidden_layer = self.decoder.package_hidden(1, hidden_layer)
        data, targets = Variable(abstract[:-1].view(len(abstract) - 1, -1)), Variable(abstract[1:].view(-1))
        return_values = self.decoder(data, hidden_layer, return_h)
        return return_values

