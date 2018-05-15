import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class LSTMEncoder(nn.Module):
    def __init__(self, system_args, num_tokens):
        super(LSTMEncoder, self).__init__()
        rnn_type = system_args.model
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(system_args.dropouti)
        self.hdrop = nn.Dropout(system_args.dropouth)
        self.drop = nn.Dropout(system_args.dropout)
        self.embedding = nn.Embedding(num_tokens, system_args.emsize)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(system_args.emsize if l == 0 else system_args.nhid, system_args.nhid if l != system_args.nlayers - 1 else (system_args.emsize if system_args.tied else system_args.nhid), 1, dropout=0) for l in range(system_args.nlayers)]
        if rnn_type == "GRU":
            self.rnns = [torch.nn.GRU(system_args.emsize if l == 0 else system_args.nhid, system_args.nhid if l != system_args.nlayers - 1 else system_args.emsize, 1, dropout=0) for l in range(system_args.nlayers)]
        if system_args.wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=system_args.wdrop) for rnn in self.rnns]
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.final = nn.Linear(system_args.nhid, num_tokens)

        if system_args.tied:
            self.final.weight = self.embedding.weight

        self.init_weights()
        self.dropout = system_args.dropout
        self.dropouti = system_args.dropouti
        self.dropouth = system_args.dropouth
        self.dropoute = system_args.dropoute
        self.rnn_type = rnn_type
        self.ninp = system_args.emsize
        self.nhid = system_args.nhid
        self.nlayers = system_args.nlayers
        self.tie_weights = system_args.tied

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.final.bias.data.fill_(0)
        self.final.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.embedding, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        decoded = self.final(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]

class LSTMDecoder(nn.Module):
    def __init__(self, system_args, num_tokens):
        super(LSTMDecoder, self).__init__()
        rnn_type = system_args.model
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(system_args.dropouti)
        self.hdrop = nn.Dropout(system_args.dropouth)
        self.drop = nn.Dropout(system_args.dropout)
        self.embedding = nn.Embedding(num_tokens, system_args.emsize)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(system_args.emsize if l == 0 else system_args.nhid, system_args.nhid if l != system_args.nlayers - 1 else (system_args.emsize if system_args.tied else system_args.nhid), 1, dropout=0) for l in range(system_args.nlayers)]
        if rnn_type == "GRU":
            self.rnns = [torch.nn.GRU(system_args.emsize if l == 0 else system_args.nhid, system_args.nhid if l != system_args.nlayers - 1 else system_args.emsize, 1, dropout=0) for l in range(system_args.nlayers)]
        if system_args.wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=system_args.wdrop) for rnn in self.rnns]
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.final = nn.Linear(system_args.nhid, num_tokens)

        if system_args.tied:
            self.final.weight = self.embedding.weight

        self.init_weights()
        self.dropout = system_args.dropout
        self.dropouti = system_args.dropouti
        self.dropouth = system_args.dropouth
        self.dropoute = system_args.dropoute
        self.rnn_type = rnn_type
        self.ninp = system_args.emsize
        self.nhid = system_args.nhid
        self.nlayers = system_args.nlayers
        self.tie_weights = system_args.tied

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.final.bias.data.fill_(0)
        self.final.weight.data.uniform_(-initrange, initrange)

    def context_apply(self, context, abstract, type):
        if type == "sum":
            return context + abstract
        else:
            return torch.div(context + abstract, 2.0)

    def forward(self, input, hidden, return_h=False, context=None, is_context_available=False, concat_type="sum"):
        emb = embedded_dropout(self.embedding, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        if is_context_available:
            emb = self.context_apply(context, emb, concat_type)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        decoded = self.final(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]

class Seq2Seq(nn.Module):
    def __init__(self, system_args, num_tokens):
        super(Seq2Seq, self).__init__()
        self.title_abstract_concat = system_args.title_abstract_concat
        self.title_abstract_concat_type = system_args.title_abstract_concat_type
        self.decoder = LSTMDecoder(system_args, num_tokens)
        self.encoder = LSTMEncoder(system_args, num_tokens)
        self.encoder_model = system_args.encoder

    def load_word_embeddings(self, new_embeddings):
        self.decoder.embedding.weight.data.copy_(new_embeddings)

    def forward(self, title, abstract, return_h=False):
        encoder_context = None
        """ The encoder context would only be available in case of the LSTM based encoder.  """
        if self.encoder_model == "LSTM":
            encoder_output, encoder_context = self.encoder(Variable(title.view(len(title), -1)), self.encoder.init_hidden(1), return_h=False)
            h1, h2 = encoder_context[-1]
            hidden_layer = self.decoder.init_hidden(1)
            encoder_context = h1
        data, targets = Variable(abstract[:-1].view(len(abstract) - 1, -1)), Variable(abstract[1:].view(-1))
        return_values = self.decoder(data, hidden_layer, return_h, encoder_context,
                                     is_context_available=self.title_abstract_concat, concat_type=self.title_abstract_concat_type)
        return return_values

