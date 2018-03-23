import os
import torch
import nltk

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

class TitlesAndAbstracts(object):

    unknown = 'UNK'
    end_of_sequence = '<eos>'

    def __init__(self, path):
        self.dictionary = Dictionary()

        self.title_train = self.tokenize_titles(os.path.join(path, 'titles_train.txt'))
        self.title_valid = self.tokenize_titles(os.path.join(path, 'titles_valid.txt'))
        self.title_test = self.tokenize_titles(os.path.join(path, 'titles_test.txt'))

        self.corpus_train = self.tokenize_abstracts(os.path.join(path, 'corpus_train.txt'))
        self.corpus_valid = self.tokenize_abstracts(os.path.join(path, 'corpus_valid.txt'))
        self.corpus_test = self.tokenize_abstracts(os.path.join(path, 'corpus_test.txt'))

    def tokenize_test_title(self, title):
        words = nltk.word_tokenize(title)
        title_tensor = torch.LongTensor(len(words))
        for i, word in enumerate(words):
            if word in self.dictionary.word2idx:
                title_tensor[i] = self.dictionary.word2idx[word]
            else:
                title_tensor[i] = self.dictionary.word2idx[TitlesAndAbstracts.unknown]
        return title_tensor

    """ Returns a list of LongTensors each representing one title """
    def tokenize_titles(self, path):
        assert os.path.exists(path)
        titles = []
        with open(path, 'r') as f:
            for line in f:
                words = nltk.word_tokenize(line)
                for word in words:
                    self.dictionary.add_word(word)

        self.dictionary.add_word(TitlesAndAbstracts.unknown)

        with open(path, 'r') as f:
            for line in f:
                words = nltk.word_tokenize(line)
                title = torch.LongTensor(len(words))
                for i, word in enumerate(words):
                    title[i] = self.dictionary.word2idx[word]
                titles.append(title)

        return titles

    """ Returns a list of LongTensors each representing one abstract """
    def tokenize_abstracts(self, path):
        assert os.path.exists(path)
        abstracts = []
        with open(path, 'r') as f:
            for line in f:
                words = nltk.word_tokenize(line)
                if '----------' in line:
                    self.dictionary.add_word(TitlesAndAbstracts.end_of_sequence)
                    continue
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            single_abstract = []
            for line in f:
                words = nltk.word_tokenize(line)
                if '----------' in line:
                    single_abstract.append(self.dictionary.word2idx[TitlesAndAbstracts.end_of_sequence])
                    abstracts.append(torch.LongTensor(single_abstract))
                    single_abstract = []
                    continue
                for word in words:
                    single_abstract.append(self.dictionary.word2idx[word])

        return abstracts

    def cudify(self, batch_size):
        cuda_title_train = []
        for tensor in self.title_train:
            cuda_title_train.append(tensor.cuda())

        cuda_title_valid = []
        for tensor in self.title_valid:
            cuda_title_valid.append(tensor.cuda())

        cuda_title_test = []
        for tensor in self.title_test:
            cuda_title_test.append(tensor.cuda())

        cuda_abstracts_train = []
        for tensor in self.corpus_train:
            cuda_abstracts_train.append(tensor.cuda())

        cuda_abstracts_valid = []
        for tensor in self.corpus_valid:
            cuda_abstracts_valid.append(tensor.cuda())

        cuda_abstracts_test = []
        for tensor in self.corpus_test:
            cuda_abstracts_test.append(tensor.cuda())

        return cuda_title_train, cuda_title_valid, cuda_title_test, cuda_abstracts_train, cuda_abstracts_valid,\
                cuda_abstracts_test

if __name__ == "__main__":
    corpus = TitlesAndAbstracts('abstracts/')
    print(corpus.title_train[0], corpus.corpus_train[0])