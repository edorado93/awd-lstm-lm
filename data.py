import os
import torch

from collections import Counter

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, min_freq=5):
        self.dictionary = Dictionary()
        self.dictionary.add_word('<unk>')
        self.min_freq = min_freq
        self.train = self.tokenize(os.path.join(path, 'train.txt'), create_dict=True)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path, create_dict=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        all_words = Dictionary()
        tokens = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    all_words.add_word(word)

        for index in all_words.counter:
            if all_words.counter[index] >= self.min_freq:
                # ONLY consider words of the training corpus for vocab. Not the validation and train sets.
                if create_dict:
                    self.dictionary.add_word(all_words.idx2word[index])

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx.get(word, self.dictionary.word2idx['<unk>'])
                    token += 1

        return ids
