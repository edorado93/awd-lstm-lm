import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout, LockedDropoutLSTMCell
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

class LSTMDecoderCell(nn.Module):
    def __init__(self, system_args, num_tokens):
        super(LSTMDecoderCell, self).__init__()
        rnn_type = system_args.model
        self.lockdropLSTM = LockedDropoutLSTMCell()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(system_args.dropouti)
        self.hdrop = nn.Dropout(system_args.dropouth)
        self.drop = nn.Dropout(system_args.dropout)
        self.embedding = nn.Embedding(num_tokens, system_args.emsize)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTMCell(system_args.emsize if l == 0 else system_args.nhid, system_args.nhid if l != system_args.nlayers - 1 else (system_args.emsize if system_args.tied else system_args.nhid)) for l in range(system_args.nlayers)]
        if rnn_type == "GRU":
            self.rnns = [torch.nn.GRU(system_args.emsize if l == 0 else system_args.nhid, system_args.nhid if l != system_args.nlayers - 1 else system_args.emsize, 1, dropout=0) for l in range(system_args.nlayers)]
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
        self.attention = SoftDotAttention(system_args.emsize) if system_args.attention else None

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

    def forward(self, input, hidden, encoder_outs=None, return_h=False, context=None, is_context_available=False, concat_type="sum"):
        emb = embedded_dropout(self.embedding, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        if is_context_available:
            emb = self.context_apply(context, emb, concat_type)

        """ The number of words in the abstract """
        seqlen = input.size()[0]

        raw_outputs = {l : [] for l in range(self.nlayers)}
        outputs = {l : [] for l in range(self.nlayers)}

        for j in range(seqlen):

            """ The embedding for each word to be fed into the lstm """
            rnn_input = emb[j, :, :]

            for l, rnn in enumerate(self.rnns):
                rnn_hidden, rnn_cell = rnn(rnn_input, hidden[l])

                """ Updated hidden states of the LSTMCell according to the word currently fed """
                hidden[l] = (rnn_hidden, rnn_cell)

                """ Append the raw output for this layer and timestep i.e. word"""
                raw_outputs[l].append(rnn_hidden.unsqueeze(0))

                """ Apply dropoutH for this layer if this isn't the last layer and append to outputs. """
                if l != self.nlayers - 1:
                    rnn_input = self.lockdropLSTM(rnn_hidden, self.dropouth)
                    outputs[l].append(rnn_input.unsqueeze(0))

            if self.attention:
                out = self.attention(rnn_hidden, encoder_outs)
            else:
                out = rnn_hidden
            output = self.lockdropLSTM(out, self.dropout)
            outputs[self.nlayers - 1].append(output.unsqueeze(0))

        """ Concatenating outputs- both raw and dropped ones across time steps """
        raw = []
        op = []
        for l in range(self.nlayers):
            raw.append(torch.cat(raw_outputs[l], dim=0))
            op.append(torch.cat(outputs[l], dim=0))
            hidden[l] = (hidden[l][0].unsqueeze(0), hidden[l][1].unsqueeze(0))

        output = op[-1]
        decoded = self.final(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw, op
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
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
        self.attention = LSTMAttention(system_args.emsize) if system_args.attention else None

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

    def forward(self, input, hidden, encoder_outs=None, return_h=False, context=None, is_context_available=False, concat_type="sum"):
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

        if self.attention:
            output = self.attention(output, encoder_outs)

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
        self.encoder = LSTMEncoder(system_args, num_tokens)
        if system_args.use_cell:
            self.decoder = LSTMDecoderCell(system_args, num_tokens)
        else:
            self.decoder = LSTMDecoder(system_args, num_tokens)
        self.encoder_model = system_args.encoder
        self.dim_change = nn.Linear(system_args.emsize, system_args.nhid)
        self.system_args = system_args

    def get_decoder_hidden_state(self, encoder_hidden):
        h1, h2 = encoder_hidden[-1]
        hidden_layer = self.decoder.init_hidden(1)
        hidden_layer[0] = (self.dim_change(h1), self.dim_change(h2))
        if self.system_args.use_cell:
           hidden_layer[0] =  (hidden_layer[0][0].squeeze(0),  hidden_layer[0][1].squeeze(0))
        return hidden_layer

    def load_word_embeddings(self, new_embeddings):
        self.decoder.embedding.weight.data.copy_(new_embeddings)

    def forward(self, title, abstract, return_h=False):
        """ The encoder context would only be available in case of the LSTM based encoder.  """
        _, encoder_hidden, _, encoder_outputs = self.encoder(Variable(title.view(len(title), -1)), self.encoder.init_hidden(1), return_h=True)
        decoder_hidden = self.get_decoder_hidden_state(encoder_hidden)
        encoder_context = encoder_hidden[-1][0] # Last hidden state. encoder hidden is a tuple of hidden state and cell state
        data, targets = Variable(abstract[:-1].view(len(abstract) - 1, -1)), Variable(abstract[1:].view(-1))
        return_values = self.decoder(data, decoder_hidden, encoder_outputs[-1], return_h, encoder_context,
                                     is_context_available=self.title_abstract_concat, concat_type=self.title_abstract_concat_type)
        return return_values

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        context = context.transpose(0, 1) # sourceL * batch * dim

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde

class LSTMAttention(nn.Module):
    def __init__(self, dim):
        super(LSTMAttention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, output, context):

        # batch_size * abstract_length * embedding_dim
        output = output.transpose(0, 1)

        # batch_size * title_length * embedding_dim
        context = context.transpose(0, 1)

        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        mask = torch.eq(attn, 0).data.byte()
        attn.data.masked_fill_(mask, -float('inf'))
        attn = nn.functional.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)

        # output -> (batch, out_len, dim)
        output = nn.functional.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)


        if not output.is_contiguous():
            output = output.contiguous()

        return output