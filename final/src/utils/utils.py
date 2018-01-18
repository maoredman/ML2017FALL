import array
import errno
import json
import os
import random
import zipfile
from argparse import ArgumentParser
from collections import Counter
from os.path import dirname, abspath

import nltk
import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import trange, tqdm

URL = {
    'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
    'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
    'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
    'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip'
}



def get_args():
    parser = ArgumentParser(description='PyTorch R-net')

    parser.add_argument('--name', type=str, default="r-net")
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch_num', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--batch_size_dev', type=int, default=64)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--train_json', type=str, default="./data/squad/split_train-v0114.json")
    parser.add_argument('--dev_json', type=str, default="./data/squad/split_dev-v0114.json")
    parser.add_argument('--update_word_embedding', type=bool, default=False)
    parser.add_argument('--update_char_embedding', type=bool, default=True)
    parser.add_argument('--hidden_size', type=int, default=75)
    parser.add_argument('--attention_size', type=int, default=75)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--app_path', type=str, default=dirname(dirname(abspath(__file__))))
    parser.add_argument('--pin_memory', type=bool, default=False)

    args = parser.parse_args()
    return args


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def load_word_vectors(root, wv_type, dim):
    """

    From https://github.com/pytorch/text/

    BSD 3-Clause License

    Copyright (c) James Bradbury and Soumith Chintala 2016,
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        return torch.load(fname_pt)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    
    else:
        raise RuntimeError('unable to load word vectors %s from %s' % (wv_type, root))

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        print("Loading word vectors from {}".format(fname_txt))
        for line in trange(len(cm)):
            
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret


class RawExample(object):
    pass


def make_dirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def maybe_download(url, download_path, filename):
    if not os.path.exists(os.path.join(download_path, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url, os.path.join(download_path, filename), reporthook=t.update_to)
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e


def read_train_json(path, debug_mode, debug_len, delete_long_context=True, delete_long_question=True,
                    longest_context=400, longest_question=100):
    import sys
    sys.path.insert(0, "./jieba-zh_TW")
    import jieba
    print(jieba)
    with open(path) as fin:
        data = json.load(fin)
    examples = []
    deleted_context = []
    deleted_question = []
    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            passage = p['context']
            if delete_long_context and len(list(jieba.cut(passage))) > longest_context:
                # dump long context into pickle
                deleted_context.append(passage)
                continue
            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]

                if delete_long_question and len(list(jieba.cut(question))) > longest_question:
                    deleted_question.append(question)
                    continue

                question_id = qa["id"]
                for ans in answers:
                    answer_start = int(ans["answer_start"])
                    answer_text = ans["text"]
                    e = RawExample()
                    e.title = title
                    e.passage = passage
                    e.question = question
                    e.question_id = question_id
                    e.answer_start = answer_start
                    e.answer_text = answer_text
                    examples.append(e)

                    if debug_mode and len(examples) >= debug_len:
                        return 
    # dump long file
    import pickle
    #pickle.dump(deleted_context, open("deleted_context", "wb"))
    #pickle.dump(deleted_question, open("deleted_question", "wb"))
    print("train examples :%s" % len(examples))
    return examples


def get_counter(*seqs):
    word_counter = {}
    char_counter = {}
    for seq in seqs:
        for doc in seq:
            for word in doc:
                word_counter.setdefault(word, 0)
                word_counter[word] += 1
                for char in word:
                    char_counter.setdefault(char, 0)
                    char_counter[char] += 1
    return word_counter, char_counter


def read_dev_json(path, debug_mode, debug_len):
    with open(path) as fin:
        data = json.load(fin)
    examples = []

    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']

            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                question_id = qa["id"]
                answer_start_list = [ans["answer_start"] for ans in answers]
                c = Counter(answer_start_list)
                most_common_answer, freq = c.most_common()[0]
                answer_text = None
                answer_start = None
                if freq > 1:
                    for i, ans_start in enumerate(answer_start_list):
                        if ans_start == most_common_answer:
                            answer_text = answers[i]["text"]
                            answer_start = answers[i]["answer_start"]
                            break
                else:
                    answer_text = answers[random.choice(range(len(answers)))]["text"]
                    answer_start = answers[random.choice(range(len(answers)))]["answer_start"]

                e = RawExample()
                e.title = title
                e.passage = context
                e.question = question
                e.question_id = question_id
                e.answer_start = answer_start
                e.answer_text = answer_text 
                examples.append(e)

                if debug_mode and len(examples) >= debug_len:
                    return examples

    return examples


def tokenized_by_answer(context, answer_text, answer_start, tokenizer):
    """
    Locate the answer token-level position after tokenizing as the original location is based on
    char-level

    snippet modified from: https://github.com/haichao592/squad-tf/blob/master/dataset.py

    :param context:  passage
    :param answer_text:     context/passage
    :param answer_start:    answer start position (char level)
    :param tokenizer: tokenize function
    :return: tokenized passage, answer start index, answer end index (inclusive)
    """
    # WARNING: it force answer to be split
    fore = context[:answer_start]
    mid = context[answer_start: answer_start + len(answer_text)]
    after = context[answer_start + len(answer_text):]

    tokenized_fore = list(tokenizer(fore))
    tokenized_mid = list(tokenizer(mid))
    tokenized_after = list(tokenizer(after))
    tokenized_text = list(tokenizer(answer_text))

    '''
    for i, j in zip(tokenized_text, tokenized_mid):
        if i != j:
            return None
    '''
    words = []
    words.extend(tokenized_fore)
    words.extend(tokenized_mid)
    words.extend(tokenized_after)
    answer_start_token, answer_end_token = len(tokenized_fore), len(tokenized_fore) + len(tokenized_mid) - 1
    return words, answer_start_token, answer_end_token
# NEW ADDED: POS Tagging
def pos_tokenized_by_answer(context, answer_text, answer_start, tokenizer):
    """
    Locate the answer token-level position after tokenizing as the original location is based on
    char-level

    snippet modified from: https://github.com/haichao592/squad-tf/blob/master/dataset.py

    :param context:  passage
    :param answer_text:     context/passage
    :param answer_start:    answer start position (char level)
    :param tokenizer: tokenize function
    :return: tokenized passage, answer start index, answer end index (inclusive)
    """
    # WARNING: it force answer to be split
    fore = context[:answer_start]
    mid = context[answer_start: answer_start + len(answer_text)]
    after = context[answer_start + len(answer_text):]

    passage_pos = []
    tokenized_fore = []
    tokenized_mid = []
    tokenized_after = []
    tokenized_text = []
    # 
    
    for w in tokenizer(fore):
        tokenized_fore.append(w.word)
        passage_pos.append(w.flag)
    for w in tokenizer(mid):
        tokenized_mid.append(w.word)
        passage_pos.append(w.flag)
    for w in tokenizer(after):
        tokenized_after.append(w.word)
        passage_pos.append(w.flag)
    for w in tokenizer(answer_text):
        tokenized_text.append(w.word)

    for i, j in zip(tokenized_text, tokenized_mid):
        if i != j:
            return None

    words = []
    words.extend(tokenized_fore)
    words.extend(tokenized_mid)
    words.extend(tokenized_after)
    answer_start_token, answer_end_token = len(tokenized_fore), len(tokenized_fore) + len(tokenized_mid) - 1
    return words, answer_start_token, answer_end_token, passage_pos

def truncate_word_counter(word_counter, max_symbols):
    words = [(freq, word) for word, freq in word_counter.items()]
    words.sort()
    return {word: freq for freq, word in words[:max_symbols]}


def read_embedding(root, word_type, dim):
    wv_dict, wv_vectors, wv_size = load_word_vectors(root, word_type, dim)
    return wv_dict, wv_vectors, wv_size


def get_rnn(rnn_type):
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        network = torch.nn.GRU
    elif rnn_type == "lstm":
        network = torch.nn.LSTM
    else:
        raise ValueError("Invalid RNN type %s" % rnn_type)
    return network


def sort_idx(seq):
    """

    :param seq: variable
    :return:
    """
    return sorted(range(seq.size(0)), key=lambda x: seq[x])


def prepare_data():
    make_dirs("data/cache")
    make_dirs("data/embedding/char")
    make_dirs("data/embedding/word")
    make_dirs("data/squad")
    make_dirs("data/trained_model")
    make_dirs("checkpoint")
    # TODO: CHANGED
    nltk.download("punkt")

    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"
    squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

    train_url = os.path.join(squad_base_url, train_filename)
    dev_url = os.path.join(squad_base_url, dev_filename)

    download_prefix = os.path.join("data", "squad")
    #maybe_download(train_url, download_prefix, train_filename)
    #maybe_download(dev_url, download_prefix, dev_filename)
    # Remove char embedding

    #char_embedding_pretrain_url = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt"
    #char_embedding_filename = "glove_char.840B.300d.txt"
    #maybe_download(char_embedding_pretrain_url, "data/embedding/char", char_embedding_filename)


def read_test_json(path, debug_mode, debug_len):
    with open(path) as fin:
        data = json.load(fin)
    examples = []

    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']

            for qa in qas:
                question = qa["question"]
                
                question_id = qa["id"]
                # No answer in test_json
                answer_text = None
                answer_start = None
                
                e = RawExample()
                e.title = title
                e.passage = context
                e.question = question
                e.question_id = question_id
                e.answer_start = answer_start
                e.answer_text = answer_text
                # ADDED: To fit new testing set
                e.offset = 0
                examples.append(e)

                if debug_mode and len(examples) >= debug_len:
                    return examples

    return examples


def read_test_pickle():
    import pickle
    examples = []
    qid = pickle.load(open("./data/test_2sentence_pickles/question_id.pickle", "rb"))
    q_text = pickle.load(open("./data/test_2sentence_pickles/question_text.pickle", "rb"))
    seg_start = pickle.load(open("./data/test_2sentence_pickles/segment_start_position.pickle", "rb"))
    seg_text = pickle.load(open("./data/test_2sentence_pickles/segment_text.pickle", "rb"))
    for id_num, question, offset, passage in zip(qid, q_text, seg_start, seg_text):
        e = RawExample()
        e.passage = passage
        e.question = question
        e.question_id = id_num
        e.offset = offset
        examples.append(e)
    return examples

def read_error_analyze_pickle():
    import pickle
    examples = []
    qid = pickle.load(open("./data/error_analyze/question_id.pickle", "rb"))
    q_text = pickle.load(open("./data/error_analyze/question_text.pickle", "rb"))
    seg_start = pickle.load(open("./data/error_analyze/segment_start_position.pickle", "rb"))
    seg_text = pickle.load(open("./data/error_analyze/segment_text.pickle", "rb"))
    for id_num, question, offset, passage in zip(qid, q_text, seg_start, seg_text):
        e = RawExample()
        e.passage = passage
        e.question = question
        e.question_id = id_num
        e.offset = offset
        examples.append(e)
    return examples


def read_mao_pickle(split):
    '''
        split = "train" or "dev"
    '''
    # WARNING: question id is all -1
    import pickle
    examples = []
    # load train.pickle
    

    if split == "train":
        q_text, ans_start, ans_text, seg_start, seg_text, qid = pickle.load( \
        open("./data/train_1sentence_pickles/train.pickle", "rb"))
    elif split == "dev":
        q_text, ans_start, ans_text, seg_start, seg_text, qid = pickle.load( \
        open("./data/train_1sentence_pickles/dev.pickle", "rb"))
    # filter the segment which
    for question, passage, seg_offset, ans_offset, ans, id_num in \
            zip(q_text, seg_text, seg_start, ans_start, ans_text, qid):
        #
        e = RawExample()
        e.passage = passage
        e.question = question
        e.answer_start = ans_offset - seg_offset # because original start is full passage's position
        e.answer_text = ans
        e.question_id = id_num
        examples.append(e)
    return examples

def spilt_mao_train_dev():
    '''
        Use Mao's tran_2setence_pickles
    '''
    import pdb
    import pickle
    import numpy as np
    
    # passage
    q_text = pickle.load(open("./data/train_1sentence_pickles/question_text.pickle", "rb"))
    # answer
    ans_start = pickle.load(open("./data/train_1sentence_pickles/answer_start_position.pickle", "rb"))
    ans_text = pickle.load(open("./data/train_1sentence_pickles/answer_text.pickle", "rb"))
    # segment
    seg_start = pickle.load(open("./data/train_1sentence_pickles/segment_start_position.pickle", "rb"))
    seg_text = pickle.load(open("./data/train_1sentence_pickles/segment_text.pickle", "rb"))
    # WARNING: fake qid
    N = q_text.shape[0]
    qid = np.array([-0.1 for _ in range(N)])
    # sample train and dev
    
    # split it 8 : 2
    split_point = N * 8 // 10
    permutation = np.random.permutation(N)
    # 
    train_q_text, dev_q_text = q_text[permutation[:split_point]], q_text[permutation[split_point:]]
    train_ans_start, dev_ans_start = ans_start[permutation[:split_point]], ans_start[permutation[split_point:]]
    train_ans_text, dev_ans_text = ans_text[permutation[:split_point]], ans_text[permutation[split_point:]]
    train_seg_start, dev_seg_start = seg_start[permutation[:split_point]], seg_start[permutation[split_point:]]
    train_seg_text, dev_seg_text = seg_text[permutation[:split_point]], seg_text[permutation[split_point:]]
    train_qid, dev_qid = qid[permutation[:split_point]], qid[permutation[split_point:]]
    # dump all tuple to the pickle files
    pickle.dump((train_q_text, train_ans_start, train_ans_text, train_seg_start, train_seg_text, train_qid), \
            open("./data/train_1sentence_pickles/train.pickle", "wb"))
    pickle.dump((dev_q_text, dev_ans_start, dev_ans_text, dev_seg_start, dev_seg_text, dev_qid), 
            open("./data/train_1sentence_pickles/dev.pickle", "wb"))