from os import path

from nltk.corpus import brown
from nltk import WhitespaceTokenizer
import nltk

import dill
import torch
from torch.utils.data import Dataset

from symbols import EOS, PAD



class RegularDataset(Dataset):

    def __init__(self, data_dir: str, split: str):
        assert split in {"train", "test"}
        self.split = split

        self.prepare_data(data_dir)

    def prepare_data(self, data_dir: str):

        with open(path.join(data_dir, self.split + ".txt"), "r") as f:
            self.D = f.read().split(",")

        with open(path.join(data_dir, "alphabet.txt"), "r") as f:
            self.alphabet = list(f.read().split(",")) + [EOS, PAD]

        self.sym2idx = {c: i for i, c in enumerate(self.alphabet) if c != PAD}
        self.sym2idx[PAD] = -1
        self.idx2sym = {i: c for i, c in enumerate(self.alphabet) if c != PAD}
        self.idx2sym[-1] = PAD

        lens = torch.tensor([len(d) for d in self.D])
        self.maxlen = torch.max(lens)

        self.X = torch.stack(
            [
                torch.tensor(
                    [self.sym2idx[s] for s in y]
                    + [self.sym2idx[EOS]]
                    + (self.maxlen - len(y)) * [self.sym2idx[PAD]]
                )
                for y in self.D
            ]
        )
        self.Y = torch.stack(
            [
                torch.tensor(
                    [self.sym2idx[s] for s in y]
                    + [self.sym2idx[EOS]]
                    + (self.maxlen - len(y) + 1) * [self.sym2idx[PAD]]
                )
                for y in self.D
            ]
        )[:, 1:]

    def __len__(self):
        return len(self.X)

    def get_vocab_size(self):
        return len(self.alphabet)

    def get_block_size(self):
        return self.maxlen + 1

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.Y[idx]

        return x, y

    def decode(self, y):
        return "".join([self.idx2sym[i] for i in y])
    


class BrownDataset(Dataset):
    def __init__(self, split: str, n_train : int = 2000, n_test : int = 500):
        nltk.download('brown') 
        sentences =  [' '.join(map(str.lower, sentence)) for sentence in brown.sents()]
        assert split in {"train", "test"}
        assert len(sentences) >= n_train + n_test
        if split == "train":
            sentences = sentences[:n_train]
        else:
            sentences = sentences[n_train:n_train+n_test]
        self.prepare_data(sentences,  WhitespaceTokenizer())


    def prepare_data(self, sentences, tokenizer):
        self.toknizer = tokenizer
        self.sentences = sentences
        self.tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in self.sentences]
        self.tokens = list({token for sentence in self.tokenized_sentences for token in sentence}) + [EOS, PAD]
        self.sym2idx = {word: i for i, word in enumerate(self.tokens)}
        self.sym2idx[PAD] = -1
        self.idx2word = {i: word for i, word in enumerate(self.tokens)}
        self.idx2word[-1] = PAD

        lens = torch.tensor([len(s) for s in self.tokenized_sentences])
        self.maxlen = torch.max(lens)

        self.X = torch.stack(
            [
                torch.tensor(
                    [self.sym2idx[s] for s in y]
                    + [self.sym2idx[EOS]]
                    + (self.maxlen - len(y)) * [self.sym2idx[PAD]]
                )
                for y in self.tokenized_sentences
            ]
        )
        self.Y = torch.stack(
            [
                torch.tensor(
                    [self.sym2idx[s] for s in y]
                    + [self.sym2idx[EOS]]
                    + (self.maxlen - len(y) + 1) * [self.sym2idx[PAD]]
                )
                for y in self.tokenized_sentences
            ]
        )[:, 1:] 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y
    
    def encode(self, sentence, add_eos=False, pad=False):
        encoded_sentence = [self.sym2idx[s] for s in self.toknizer.tokenize(sentence)]
        if add_eos:
            encoded_sentence.append(self.sym2idx[EOS])
        if pad:
            encoded_sentence += (self.maxlen - len(encoded_sentence)) * [self.sym2idx[PAD]]
        return torch.tensor(encoded_sentence, dtype=torch.long)

    def decode(self, encoded_sentence):
        if isinstance(encoded_sentence, torch.Tensor):
            encoded_sentence = encoded_sentence.cpu().numpy()
        return ' '.join([self.idx2word[i] for i in encoded_sentence])