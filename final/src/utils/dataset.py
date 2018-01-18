from functools import partial

import nltk
import spacy
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.utils import read_train_json, read_dev_json, read_test_json, tokenized_by_answer, sort_idx
# 
from utils.utils import read_test_pickle, read_mao_pickle, read_error_analyze_pickle
from utils.utils import pos_tokenized_by_answer
import numpy as np
import re
## zh-TW jieba
import sys
sys.path.insert(0, "./jieba-zh_TW")
import jieba
import jieba.posseg as pseg
print(jieba)
# scikit
from sklearn.preprocessing import OneHotEncoder
import pickle
def char_cut(s):
    return [char for char in s]

def sentence_cut(s):
    result = []
    if False:
        split = s.split("。")
        for index, token in enumerate(split):
            if index != len(split) - 1:
                result.append(token + "。")
            else:
                if len(token) > 0:
                    # do not append delimiter in the end
                    result.append(token)
    else:
        split = re.split("(。|，|、)", s)
        for index, token in enumerate(split):
            if len(token) > 0:
                if  len(token) == 1 and (token[0] == "。" or token[0] == "，") and len(result) > 0:
                    # TODO: ADD delimiter to previous sentence
                    result[-1] += token
                else:
                    result.append(token)

    return result

def sliding_window(s):
    result = []
    window_size = 7
    quotient = len(s) // window_size
    remainder = len(s) % window_size
    for i in range(quotient):
        result.append(s[i*window_size:(i+1)*window_size])
    if remainder > 0:
        result.append(s[-remainder:])
    return result



def char_padding(seqs, wv_vec, batch_first=False):
    #import pdb
    #pdb.set_trace()
    batch_size = len(seqs)
    lengths = [len(s) for s in seqs]
    batch_length = max(lengths)
    # for each training example
    # pad 300-dimension zero vector for padding
    fake_tensor = torch.LongTensor(batch_length, batch_size).zero_()
    seq_tensor = torch.FloatTensor(batch_length, batch_size, 300).zero_()
    for i, seq in enumerate(seqs): # batch
        for j, sentence in enumerate(seq): # lengths
            # compute mean vector of sentence
            mean_vector = torch.FloatTensor(300).zero_()
            for encoded_word in sentence:
                # WARNING: UNK WORD WILL WRONG
                mean_vector += wv_vec[encoded_word]
            # take mean
            mean_vector /= len(sentence)
            # put it into tensor
            seq_tensor[j, i] = mean_vector

    if batch_first:
        # (T, B, dim) -> (B, T, dim)
        seq_tensor = seq_tensor.transpose(0, 1)
        fake_tensor = fake_tensor.t()
    return (fake_tensor, seq_tensor, lengths)


def padding(seqs, pad, batch_first=False):
    """

    :param seqs: tuple of seq_length x dim
    :return: seq_length x Batch x dim
    """
    lengths = [len(s) for s in seqs]

    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(pad)
    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])
    if batch_first:
        seq_tensor = seq_tensor.t()
    return (seq_tensor, lengths)

# ADDED: appearance bit and term frequency
def self_defined_padding(seqs, pad, appearance_bit, term_frequency=None, batch_first=False):
    """

    :param seqs: tuple of seq_length x dim
    :return: seq_length x Batch x dim
    """
    USE_APPEARANCE = False
    USE_TF = False
    if appearance_bit is not None:
        USE_APPEARANCE = True
    if term_frequency is not None:
        USE_TF = True
    lengths = [len(s) for s in seqs]

    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(pad)
    if USE_APPEARANCE:
        if USE_TF:
            # appearance bit
            appearance_bit_tensor = torch.FloatTensor(batch_length, len(seqs)).fill_(pad)
            appearances = [torch.Tensor(a) for a in appearance_bit]
            # term frequency
            term_frequency_tensor = torch.FloatTensor(batch_length, len(seqs)).fill_(pad)
            tfs = [torch.Tensor(tf) for tf in term_frequency]
            for i, (s, a, tf) in enumerate(zip(seqs, appearances, tfs)):
                end_seq = lengths[i]
                seq_tensor[:end_seq, i].copy_(s[:end_seq])
                assert end_seq == len(a)
                appearance_bit_tensor[:end_seq, i].copy_(a[:end_seq])
                # term frequency
                term_frequency_tensor[:end_seq, i].copy_(tf[:end_seq])

            if batch_first:
                seq_tensor = seq_tensor.t()
                appearance_bit_tensor = appearance_bit_tensor.t()
                term_frequency_tensor = term_frequency_tensor.t()
            return (seq_tensor, lengths, appearance_bit_tensor, term_frequency_tensor)
        else:
            # appearance bit
            appearance_bit_tensor = torch.FloatTensor(batch_length, len(seqs)).fill_(pad)
            appearances = [torch.Tensor(a) for a in appearance_bit]
            for i, (s, a) in enumerate(zip(seqs, appearances)):
                end_seq = lengths[i]
                seq_tensor[:end_seq, i].copy_(s[:end_seq])
                assert end_seq == len(a)
                appearance_bit_tensor[:end_seq, i].copy_(a[:end_seq])

            if batch_first:
                seq_tensor = seq_tensor.t()
                appearance_bit_tensor = appearance_bit_tensor.t()
            return (seq_tensor, lengths, appearance_bit_tensor)
    else:
        for i, s in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])
        if batch_first:
            seq_tensor = seq_tensor.t()
        return (seq_tensor, lengths)

# ADDED: appearance bit and pos tag
def self_defined_pos_padding(seqs, pad, appearance_bit, pos, batch_first=False):
    """

    :param seqs: tuple of seq_length x dim
    :return: seq_length x Batch x dim
    """
    
    lengths = [len(s) for s in seqs]

    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(pad)
    
    # appearance bit
    appearance_bit_tensor = torch.FloatTensor(batch_length, len(seqs)).fill_(pad)
    appearances = [torch.Tensor(a) for a in appearance_bit]
    # pos one-hot
    pos_tag_dim = len(pos[0][0])
    pos_tensor = torch.FloatTensor(batch_length, len(seqs), pos_tag_dim).fill_(pad)
    poses = [torch.from_numpy(p).float() for p in pos]
    for i, (s, a, p) in enumerate(zip(seqs, appearances, poses)):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])
        assert end_seq == len(a)
        appearance_bit_tensor[:end_seq, i].copy_(a[:end_seq])
        pos_tensor[:end_seq, i].copy_(p[:end_seq])

    if batch_first:
        seq_tensor = seq_tensor.t()
        appearance_bit_tensor = appearance_bit_tensor.t()
        pos_tensor = torch.transpose(pos_tensor, 0, 1)
    return (seq_tensor, lengths, appearance_bit_tensor, pos_tensor)
class Documents(object):
    """
        Helper class for organizing and sorting seqs

        should be batch_first for embedding
    """

    def __init__(self, tensor, lengths, appearance_bit_tensor, pos_tensor=None, term_frequency_tensor=None):
        
        self.original_lengths = lengths
        sorted_lengths_tensor, self.sorted_idx = torch.sort(torch.LongTensor(lengths), dim=0, descending=True)

        self.tensor = tensor.index_select(dim=0, index=self.sorted_idx)

        self.lengths = list(sorted_lengths_tensor)
        self.original_idx = torch.LongTensor(sort_idx(self.sorted_idx))

        #self.mask_original = torch.zeros(*self.tensor.size())
        # WARNING: CHANGED
        self.mask_original = torch.zeros(*self.tensor.size()[:2]) # use first two dimension
        for i, length in enumerate(self.original_lengths):
            self.mask_original[i][:length].fill_(1)

        # ADDED: whether a passage token is appeared in question 
        # (one bit, 1 denotes it has appear in question)
        USE_APPEARANCE = True
        self.appearance_bit_tensor = None
        if USE_APPEARANCE and appearance_bit_tensor is not None:
            self.appearance_bit_tensor = appearance_bit_tensor.index_select(dim=0, index=self.sorted_idx)
        # 
        USE_POS = False
        if USE_POS:
            self.pos_tensor = pos_tensor.index_select(dim=0, index=self.sorted_idx)
        # tf
        USE_TF = True
        if USE_TF:
            self.term_frequency_tensor = term_frequency_tensor.index_select(dim=0, index=self.sorted_idx)

    def variable(self, volatile=False):
        self.tensor = Variable(self.tensor, volatile=volatile)
        self.sorted_idx = Variable(self.sorted_idx, volatile=volatile)
        self.original_idx = Variable(self.original_idx, volatile=volatile)
        self.mask_original = Variable(self.mask_original, volatile=volatile)
        USE_APPEARANCE = True
        if USE_APPEARANCE:
            self.appearance_bit_tensor = Variable(self.appearance_bit_tensor, volatile=volatile)
        USE_POS = False
        if USE_POS:
            self.pos_tensor = Variable(self.pos_tensor, volatile=volatile)
        USE_TF = True
        if USE_TF:
            self.term_frequency_tensor = Variable(self.term_frequency_tensor, volatile=volatile)

        return self

    def cuda(self, *args, **kwargs):
        if torch.cuda.is_available():
            self.sorted_idx = self.sorted_idx.cuda(*args, **kwargs)
            self.original_idx = self.original_idx.cuda(*args, **kwargs)
            self.mask_original = self.mask_original.cuda(*args, **kwargs)
            if self.appearance_bit_tensor is not None:
                self.appearance_bit_tensor = self.appearance_bit_tensor.cuda(*args, **kwargs)
        return self

    def restore_original_order(self, sorted_tensor, batch_dim):
        return sorted_tensor.index_select(dim=batch_dim, index=self.original_idx)

    def to_sorted_order(self, original_tensor, batch_dim):
        return original_tensor.index_select(dim=batch_dim, index=self.sorted_idx)


class SQuAD(Dataset):
    def __init__(self, path, itos, stoi, itoc, ctoi, tokenizer="jieba", split="train",
                 debug_mode=False, debug_len=50, wv_vec=None):
        #self.wv_vec = wv_vec
        self.insert_start = stoi.get("<SOS>", None)
        self.insert_end = stoi.get("<EOS>", None)
        self.UNK = stoi.get("<UNK>", None)
        self.PAD = stoi.get("<PAD>", None)

        self.stoi = stoi
        self.ctoi = ctoi
        self.itos = itos
        self.itoc = itoc
        self.split = split
        self._set_tokenizer(tokenizer)
        # ADDED: Term frequency
        self.term_frequency_dict = pickle.load(open("./data/tf/tf_dict.pickle", "rb"))

        # Read and parsing raw data from json
        # Tokenizing with answer may result in different tokenized passage even the passage is the same one.
        # So we tokenize passage for each question in train split
        USE_MAO =True
        USE_APPEARANCE = True
        USE_POS = False
        USE_TF = True
        if self.split == "train":
            
            if USE_MAO:
                self.examples = read_mao_pickle("train")
            else:
                self.examples = read_train_json(path, debug_mode, debug_len)
            # Training need to split in different way
            
            self._tokenize_passage_with_answer_for_train()
        elif self.split == "dev":
        
            if USE_MAO == True:
                self.examples = read_mao_pickle("dev")
            else:
                self.examples = read_dev_json(path, debug_mode, debug_len)
            if USE_POS:
                
                postag_dict = pickle.load(open("./data/postag/postag_dict.pickle", "rb"))
                pos_enc = OneHotEncoder(n_values=len(postag_dict))
                for e in self.examples:
                    e.tokenized_passage = []
                    e.passage_pos = []
                    for w in self.tokenizer(e.passage):
                        e.tokenized_passage.append(w.word)
                        e.passage_pos.append(postag_dict[w.flag])
                    e.passage_pos = pos_enc.fit_transform(np.array(e.passage_pos).reshape(-1, 1)).toarray()
                
            else:
                for e in self.examples:
                    e.tokenized_passage = list(self.tokenizer(e.passage))

        elif self.split == "test":
            self.examples = read_test_pickle()
            #self.examples = read_test_json(path, debug_mode, debug_len)
            # Using POS tagging
            if USE_POS:
            
                postag_dict = pickle.load(open("./data/postag/postag_dict.pickle", "rb"))
                pos_enc = OneHotEncoder(n_values=len(postag_dict))
                for e in self.examples:
                    e.tokenized_passage = []
                    e.passage_pos = []
                    for w in self.tokenizer(e.passage):
                        e.tokenized_passage.append(w.word)
                        e.passage_pos.append(postag_dict[w.flag])
                    e.passage_pos = pos_enc.fit_transform(np.array(e.passage_pos).reshape(-1, 1)).toarray()
                   
            else:
                for e in self.examples:
                    e.tokenized_passage = list(self.tokenizer(e.passage))
        elif self.split == "error":
            self.examples = read_error_analyze_pickle()
            if USE_POS:
            
                postag_dict = pickle.load(open("./data/postag/postag_dict.pickle", "rb"))
                pos_enc = OneHotEncoder(n_values=len(postag_dict))
                for e in self.examples:
                    e.tokenized_passage = []
                    e.passage_pos = []
                    for w in self.tokenizer(e.passage):
                        e.tokenized_passage.append(w.word)
                        e.passage_pos.append(postag_dict[w.flag])
                    e.passage_pos = pos_enc.fit_transform(np.array(e.passage_pos).reshape(-1, 1)).toarray()
                   
            else:
                for e in self.examples:
                    e.tokenized_passage = list(self.tokenizer(e.passage))

        else:
            raise NotImplementedError
        unknown_word = 0
        total_word = 0
        if USE_POS:
            postag_dict = pickle.load(open("./data/postag/postag_dict.pickle", "rb"))
            pos_enc = OneHotEncoder(n_values=len(postag_dict))
        for e in self.examples:
            # Question using word-level, passage use sentence level 
            if USE_POS:
                e.tokenized_question = []
                e.question_pos = []
                for w in self.tokenizer(e.question):
                    e.tokenized_question.append(w.word)
                    e.question_pos.append(postag_dict[w.flag])

                e.question_pos = pos_enc.fit_transform(np.array(e.question_pos).reshape(-1, 1)).toarray()
                
            else:
                e.tokenized_question = list(jieba.cut(e.question))
            # e.tokenized_question is sentences
            #e.numeralized_question = self._numeralize_word_seq(e.tokenized_question, self.stoi)
            #e.numeralized_passage = self._sentence_level_numeralize(e.tokenized_passage)
            # 
            if USE_TF:
                e.numeralized_question, temp_unknown_word, temp_total_word, e.question_tf = \
                    self._numeralize_word_seq_and_term_frequency(e.tokenized_question, self.stoi)
            else:
                e.numeralized_question, temp_unknown_word, temp_total_word = \
                    self._numeralize_word_seq(e.tokenized_question, self.stoi)
            unknown_word += temp_unknown_word
            total_word += temp_total_word

            if USE_TF:
                e.numeralized_passage, temp_unknown_word, temp_total_word, e.passage_tf = \
                     self._numeralize_word_seq_and_term_frequency(e.tokenized_passage, self.stoi)
            else:
                e.numeralized_passage, temp_unknown_word, temp_total_word = \
                    self._numeralize_word_seq(e.tokenized_passage, self.stoi)
            unknown_word += temp_unknown_word
            total_word += temp_total_word
            # REMOVE char embedding
            e.numeralized_question_char = None
            e.numeralized_passage_char = None

            #e.numeralized_question_char = self._char_level_numeralize(e.tokenized_question)
            #e.numeralized_passage_char = self._char_level_numeralize(e.tokenized_passage)

            # ADDED: 20180115 appearance_bit
            if USE_APPEARANCE:
                e.question_apperance_bit, e.passage_appearance_bit = self._appearance_bit(e.numeralized_question, e.numeralized_passage)
        print("Total word token is: {}, unknown_word has: {}, ratio: {}".format(
                total_word, unknown_word, unknown_word / total_word))

    # ADDED: appearance_bit for passage and question
    def _appearance_bit(self, numeralized_question, numeralized_passage):
        # question index set
        
        question_index_set = set(numeralized_question)
        passage_index_set = set(numeralized_passage)
        question_apperance_bit = []
        passage_appearance_bit = []
        # search for every word whether the word appears in question set
        for word_index in numeralized_passage:
            if word_index in question_index_set and word_index != self.UNK:
                passage_appearance_bit.append(1)
            else:
                passage_appearance_bit.append(0)
        # question
        for word_index in numeralized_question:
            if word_index in passage_index_set and word_index != self.UNK:
                question_apperance_bit.append(1)
            else:
                question_apperance_bit.append(0)
        return question_apperance_bit , passage_appearance_bit

    def _tokenize_passage_with_answer_for_train(self):
        #import pdb
        #pdb.set_trace()
        USE_POS = False
        if USE_POS:
            
            postag_dict = pickle.load(open("./data/postag/postag_dict.pickle", "rb"))
            pos_enc = OneHotEncoder(n_values=len(postag_dict))
            for example in self.examples:
                
                example.tokenized_passage, answer_start, answer_end, example.passage_pos = \
                    pos_tokenized_by_answer(example.passage, example.answer_text, \
                    example.answer_start, self.tokenizer)
                # convert to numerical data
                example.passage_pos = list(map(lambda x: postag_dict[x], example.passage_pos))
                example.passage_pos = pos_enc.fit_transform(np.array(example.passage_pos).reshape(-1, 1)).toarray()
                
                example.answer_position = (answer_start, answer_end)
        else:
            for example in self.examples:
                example.tokenized_passage, answer_start, answer_end = tokenized_by_answer(example.passage, example.answer_text,
                                                                                  example.answer_start, self.tokenizer)
                example.answer_position = (answer_start, answer_end)
    # WARNING: DEPRECIATED
    def _sentence_level_numeralize(self, sentences):
        result = []
        for sentence in sentences:
            result.append(self._numeralize_word_seq(jieba.cut(sentence), self.stoi))
        return result

    def _char_level_numeralize(self, tokenized_doc):
        result = []
        for word in tokenized_doc:
            result.append(self._numeralize_word_seq(word, self.ctoi))
        return result

    def _numeralize_word_seq(self, seq, stoi, insert_sos=False, insert_eos=False):
        if self.insert_start is not None and insert_sos:
            result = [self.insert_start]
        else:
            result = []
        # count unknow_word
        unknown_word = 0
        total_word = len(seq)
        for word in seq:
            if word not in stoi:
                unknown_word += 1
            result.append(stoi.get(word, 0))

        if self.insert_end is not None and insert_eos:
            result.append(self.insert_end)
        return result, unknown_word, total_word

    # ADDED: add term frequency
    def _numeralize_word_seq_and_term_frequency(self, seq, stoi, insert_sos=False, insert_eos=False):
        # WARNING: SOS EOS NOT ADDED 
        if self.insert_start is not None and insert_sos:
            result = [self.insert_start]
        else:
            result = []
        term_frequency = []
        # count unknow_word
        unknown_word = 0
        total_word = len(seq)
        for word in seq:
            if word not in stoi:
                unknown_word += 1
            result.append(stoi.get(word, 0))
            # tf: if exist, get frequency, if not get 0. frequency
            term_frequency.append(self.term_frequency_dict.get(word, 0.))


        if self.insert_end is not None and insert_eos:
            result.append(self.insert_end)
        return result, unknown_word, total_word, term_frequency

    def __getitem__(self, idx):
        item = self.examples[idx]
        USE_APPEARANCE = True
        USE_POS = False
        USE_TF = True
        # WARNING: Complex control flow here
        if USE_APPEARANCE:
            if USE_POS:
                if self.split == "train":
                    return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                            item.numeralized_passage, item.numeralized_passage_char,
                            item.answer_position, item.answer_text, item.tokenized_passage, item.question_apperance_bit, item.passage_appearance_bit, item.question_pos, item.passage_pos)
                else:
                    return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                            item.numeralized_passage, item.numeralized_passage_char,
                            item.tokenized_passage, item.question_apperance_bit, item.passage_appearance_bit, item.question_pos, item.passage_pos)
            else:
                if USE_TF:
                    if self.split == "train":
                        return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                                item.numeralized_passage, item.numeralized_passage_char,
                                item.answer_position, item.answer_text, item.tokenized_passage, 
                                item.question_apperance_bit, item.passage_appearance_bit, item.question_tf, item.passage_tf)
                    else:
                        return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                                item.numeralized_passage, item.numeralized_passage_char,
                                item.tokenized_passage, item.question_apperance_bit, item.passage_appearance_bit, 
                                item.question_tf, item.passage_tf)
                else:
                    if self.split == "train":
                        return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                                item.numeralized_passage, item.numeralized_passage_char,
                                item.answer_position, item.answer_text, item.tokenized_passage, item.question_apperance_bit, item.passage_appearance_bit)
                    else:
                        return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                                item.numeralized_passage, item.numeralized_passage_char,
                                item.tokenized_passage, item.question_apperance_bit, item.passage_appearance_bit)
        else:
            if self.split == "train":
                return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                        item.numeralized_passage, item.numeralized_passage_char,
                        item.answer_position, item.answer_text, item.tokenized_passage)
            else:
                return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                        item.numeralized_passage, item.numeralized_passage_char,
                        item.tokenized_passage)

    def __len__(self):
        return len(self.examples)

    def _set_tokenizer(self, tokenizer):
        """
        Set tokenizer

        :param tokenizer: tokenization method
        :return: None
        """
        
        if tokenizer == "jieba":
            self.tokenizer = jieba.cut
        elif tokenizer == "jieba_pos":
            self.tokenizer = pseg.cut
        elif tokenizer == "sentence_cut":
            # tokenizer(self define)
            self.tokenizer = sentence_cut
        elif tokenizer == "char_cut":
            self.tokenizer = char_cut
        elif tokenizer == "sliding_window":
            self.tokenizer = sliding_window
        elif tokenizer == "nltk":
            self.tokenizer = nltk.word_tokenize
        elif tokenizer == "spacy":
            spacy_en = spacy.load("en")

            def spacy_tokenizer(seq):
                return [w.text for w in spacy_en(seq)]

            self.tokenizer = spacy_tokenizer
        else:
            raise ValueError("Invalid tokenizing method %s" % tokenizer)

    def _create_collate_fn(self, batch_first=True):

        def collate(examples, this):
            USE_APPEARANCE = True
            USE_POS = False
            USE_TF = True
            if USE_APPEARANCE:
                if USE_POS:
                    if this.split == "train":
                        items, question_ids, questions, questions_char, passages, passages_char, answers_positions, answer_texts, passage_tokenized, question_apperance_bit, passage_appearance_bit, question_pos, passage_pos = zip(
                            *examples)
                    else:
                        items, question_ids, questions, questions_char, passages, passages_char, passage_tokenized, question_apperance_bit, passage_appearance_bit, question_pos, passage_pos = zip(*examples)
                else:
                    if USE_TF:
                        if this.split == "train":
                            items, question_ids, questions, questions_char, passages, passages_char, answers_positions, answer_texts, passage_tokenized, question_apperance_bit, passage_appearance_bit, question_tf, passage_tf = zip(
                                *examples)
                        else:
                            items, question_ids, questions, questions_char, passages, passages_char, passage_tokenized, question_apperance_bit, passage_appearance_bit, question_tf, passage_tf = zip(*examples)
                    else:
                        if this.split == "train":
                            items, question_ids, questions, questions_char, passages, passages_char, answers_positions, answer_texts, passage_tokenized, question_apperance_bit, passage_appearance_bit = zip(
                                *examples)
                        else:
                            items, question_ids, questions, questions_char, passages, passages_char, passage_tokenized, question_apperance_bit, passage_appearance_bit = zip(*examples)
            else:
                if this.split == "train":
                    items, question_ids, questions, questions_char, passages, passages_char, answers_positions, answer_texts, passage_tokenized = zip(
                        *examples)
                else:
                    items, question_ids, questions, questions_char, passages, passages_char, passage_tokenized = zip(*examples)
            # ADDED:
            
            if USE_APPEARANCE:
                if USE_POS:
                    questions_tensor, question_lengths, question_apperance_bit_tensor, question_pos_tensor = \
                        self_defined_pos_padding(questions, this.PAD, question_apperance_bit, question_pos,batch_first=batch_first)
                    passages_tensor, passage_lengths, passage_appearance_bit_tensor, passage_pos_tensor =\
                        self_defined_pos_padding(passages, this.PAD, passage_appearance_bit, passage_pos, batch_first=batch_first)
                else:
                    if USE_TF:
                        questions_tensor, question_lengths, question_apperance_bit_tensor, question_tf_tensor = \
                            self_defined_padding(questions, this.PAD, question_apperance_bit, question_tf, batch_first=batch_first)
                        passages_tensor, passage_lengths, passage_appearance_bit_tensor, passage_tf_tensor =\
                            self_defined_padding(passages, this.PAD, passage_appearance_bit, passage_tf, batch_first=batch_first)
                    else:
                        questions_tensor, question_lengths, question_apperance_bit_tensor = \
                            self_defined_padding(questions, this.PAD, question_apperance_bit, batch_first=batch_first)
                        passages_tensor, passage_lengths, passage_appearance_bit_tensor =\
                            self_defined_padding(passages, this.PAD, passage_appearance_bit, batch_first=batch_first)
            else:
                questions_tensor, question_lengths = padding(questions, this.PAD, batch_first=batch_first)
                passages_tensor, passage_lengths = padding(passages, this.PAD, batch_first=batch_first)

            # TODO: implement char level embedding
            # questions char tensor is (B, T, dim) or (T, B, dim)
            #questions_fake_tensor, questions_char_tensor, questions_char_lengths = \
            #        char_padding(questions, self.wv_vec, batch_first=batch_first)
            #passages_fake_tensor, passages_char_tensor, passages_char_lengths = \
             #       char_padding(passages, self.wv_vec, batch_first=batch_first)

            #question_document = Documents(questions_tensor, question_lengths)
            #passages_document = Documents(passages_char_tensor, passages_char_lengths)
            # ADDED: 20180115 appearance_bit
            
            if USE_APPEARANCE:
                if USE_POS:
                    question_document = Documents(questions_tensor, question_lengths, question_apperance_bit_tensor, question_pos_tensor)
                    passages_document = Documents(passages_tensor, passage_lengths, passage_appearance_bit_tensor, passage_pos_tensor)
                else:
                    if USE_TF:
                        question_document = Documents(questions_tensor, question_lengths, question_apperance_bit_tensor, term_frequency_tensor=question_tf_tensor)
                        passages_document = Documents(passages_tensor, passage_lengths, passage_appearance_bit_tensor, term_frequency_tensor=passage_tf_tensor)
                    else:
                        question_document = Documents(questions_tensor, question_lengths, question_apperance_bit_tensor)
                        passages_document = Documents(passages_tensor, passage_lengths, passage_appearance_bit_tensor)
            else:
                question_document = Documents(questions_tensor, question_lengths)
                passages_document = Documents(passages_tensor, passage_lengths)
            
            if this.split == "train":
                return question_ids, question_document, passages_document, torch.LongTensor(answers_positions), answer_texts

            else:

                return question_ids, question_document, passages_document, passage_tokenized, items

        return partial(collate, this=self)

    def get_dataloader(self, batch_size, num_workers=20, shuffle=True, batch_first=True, pin_memory=False):
        """

        :param batch_first:  Currently, it must be True as nn.Embedding requires batch_first input
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory)
