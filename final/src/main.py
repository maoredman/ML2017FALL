import os
import pickle

import torch

from trainer import Trainer
from utils.utils import prepare_data, get_args, read_embedding, spilt_mao_train_dev


# TODO: read vocab into a cpu embedding layer
def read_vocab(vocab_config):
    """
    :param counter: counter of words in dataset
    :param vocab_config: word_embedding config: (root, word_type, dim)
    :return: itos, stoi, vectors
    """
    print("Using {}".format(vocab_config["embedding_type"]))
    wv_dict, wv_vectors, wv_size = read_embedding(vocab_config["embedding_root"],
                                                  vocab_config["embedding_type"],
                                                  vocab_config["embedding_dim"])

    # embedding size = glove vector size
    embed_size = wv_vectors.size(1)
    print("word embedding size: %d" % embed_size)

    itos = vocab_config['specials'][:]
    stoi = {}

    itos.extend(list(w for w, i in sorted(wv_dict.items(), key=lambda x: x[1])))

    for idx, word in enumerate(itos):
        stoi[word] = idx

    vectors = torch.zeros([len(itos), embed_size])

    for word, idx in stoi.items():
        if word not in wv_dict or word in vocab_config['specials']:
            continue
        vectors[idx, :wv_size].copy_(wv_vectors[wv_dict[word]])
    return itos, stoi, vectors


def main():
    args = get_args()
    prepare_data()
    word_vocab_config = {
        "<UNK>": 0,
        "<PAD>": 1,
        "<start>": 2,
        "<end>": 3,
        "insert_start": "<SOS>",
        "insert_end": "<EOS>",
        "tokenization": "nltk",
        "specials": ["<UNK>", "<PAD>", "<SOS>", "<EOS>"],
        "embedding_root": os.path.join(args.app_path, "data", "embedding", "word"),
        "embedding_type": "gensim_mincount0_size128",
        "embedding_dim": 128
    }
    print("Reading Vocab", flush=True)
    '''
    char_vocab_config = word_vocab_config.copy()
    char_vocab_config["embedding_root"] = os.path.join(args.app_path, "data", "embedding", "char")
    char_vocab_config["embedding_type"] = "glove_char.840B"
    '''
    # TODO: build vocab out of dataset
    # build vocab
    itos, stoi, wv_vec = read_vocab(word_vocab_config)
    itoc, ctoi, cv_vec = itos, stoi, wv_vec
    #itoc, ctoi, cv_vec = read_vocab(char_vocab_config)

    char_embedding_config = {"embedding_weights": cv_vec,
                             "padding_idx": word_vocab_config["<UNK>"],
                             "update": args.update_char_embedding,
                             "bidirectional": args.bidirectional,
                             "cell_type": "gru", "output_dim": 300}

    word_embedding_config = {"embedding_weights": wv_vec,
                             "padding_idx": word_vocab_config["<UNK>"],
                             "update": args.update_word_embedding}

    sentence_encoding_config = {"hidden_size": args.hidden_size,
                                "num_layers": args.num_layers,
                                "bidirectional": True,
                                "dropout": args.dropout, }

    pair_encoding_config = {"hidden_size": args.hidden_size,
                            "num_layers": args.num_layers,
                            "bidirectional": args.bidirectional,
                            "dropout": args.dropout,
                            "gated": True, "mode": "GRU",
                            "rnn_cell": torch.nn.GRUCell,
                            "attn_size": args.attention_size,
                            "residual": args.residual}

    self_matching_config = {"hidden_size": args.hidden_size,
                            "num_layers": args.num_layers,
                            "bidirectional": args.bidirectional,
                            "dropout": args.dropout,
                            "gated": True, "mode": "GRU",
                            "rnn_cell": torch.nn.GRUCell,
                            "attn_size": args.attention_size,
                            "residual": args.residual}

    pointer_config = {"hidden_size": args.hidden_size,
                      "num_layers": args.num_layers,
                      "dropout": args.dropout,
                      "residual": args.residual,
                      "rnn_cell": torch.nn.GRUCell}

    print("DEBUG Mode is ", "On" if args.debug else "Off", flush=True)
    USE_MAO = True 
    PREDICT_MODE = True 
    if USE_MAO and not PREDICT_MODE:
        spilt_mao_train_dev()
        train_cache = "./data/cache/SQuAD_seg_0118_1006_tf%s.pkl" % ("_debug" if args.debug else "")
        dev_cache = "./data/cache/SQuAD_dev_seg_0118_1006_tf%s.pkl" % ("_debug" if args.debug else "")
        test_cache = "./data/cache/SQuAD_test_seg_0118_1039_tf_1sent+predict_1sent.pkl%s.pkl" % ("_debug" if args.debug else "")
    else:
        train_cache = "./data/cache/SQuAD_0114%s.pkl" % ("_debug" if args.debug else "")
        dev_cache = "./data/cache/SQuAD_dev_0114%s.pkl" % ("_debug" if args.debug else "")
        test_cache = "./data/cache/SQuAD_test%s.pkl" % ("_debug" if args.debug else "")
    # json
    train_json = args.train_json
    dev_json = args.dev_json
    test_json = "./data/squad/test-v1.1.json"
    # read dataset
    
    if not PREDICT_MODE:
        train = read_dataset(train_json, itos, stoi, itoc, ctoi, train_cache, args.debug, wv_vec=wv_vec)
        dev = read_dataset(dev_json, itos, stoi, itoc, ctoi, dev_cache, args.debug, split="dev", wv_vec=wv_vec)
    
    test = read_dataset(test_json, itos, stoi, itoc, ctoi, test_cache, args.debug, split="test", wv_vec=wv_vec)
    if not PREDICT_MODE:
        print("Training pair {}".format(len(train)))
        print("Dev pair {}".format(len(dev)))
    print("Test pair {}".format(len(test)))
    # 
    if not PREDICT_MODE:
        dev_dataloader = dev.get_dataloader(args.batch_size_dev)
        train_dataloader = train.get_dataloader(args.batch_size, shuffle=True, pin_memory=args.pin_memory)
    else:
        dev_dataloader = None
        train_dataloader = None
    test_dataloader = test.get_dataloader(args.batch_size_dev, shuffle=False)
    # ADD Test data
    trainer = Trainer(args, train_dataloader, dev_dataloader,
                      char_embedding_config, word_embedding_config,
                      sentence_encoding_config, pair_encoding_config,
                      self_matching_config, pointer_config)
    trainer.train(args.epoch_num)
    # Predict result
    #_, f1 = trainer.eval()
    #print("dev f1 {}".format(f1))
    make_prediction = True
    if make_prediction:
        pred_result = trainer.predict(test_dataloader)

        # print to csv 
        import pandas as pd
        from IPython import embed
        
        df = pd.DataFrame(pred_result, columns=["id", "answer"])
        
        df.to_csv("R_Net_output", index=False)
    
    # error analysis
    '''
    import socket
    import subprocess
    TCP_IP = "0.0.0.0"
    TCP_PORT = 9487
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    
    BUFFER_SIZE = 50000
    while False:
        
        print("Accepting..........")
        conn, addr = s.accept()
        print("Accepting Succeed")
        data = conn.recv(BUFFER_SIZE)
        if not data: break
        # a list ["question", "passage"]
        print("Parse Client data.....")
        question_and_passage = pickle.loads(data)
        # dump pickl into error folder
        print("Dump to Pickle")
        subprocess.call(["python3", "error_analysis.py", str(question_and_passage[0]), str(question_and_passage[1])])
        error_cache = "./data/cache/SQuAD_error_cache%s.pkl" % ("_debug" if args.debug else "")
        error = read_dataset(test_json, itos, stoi, itoc, ctoi, error_cache, args.debug, split="error", wv_vec=wv_vec)
        error_dataloader = error.get_dataloader(1, shuffle=False)
        print("Predicting")
        error_result = trainer.predict(error_dataloader, print_it=True)
        # return answer to client
        conn.send(pickle.dumps(error_result))
        conn.close()
    '''



def read_dataset(json_file, itos, stoi, itoc, ctoi, cache_file, is_debug=False, split="train", wv_vec=None):
    '''
    if os.path.isfile(cache_file):
        print("Read built %s dataset from %s" % (split, cache_file), flush=True)
        dataset = pickle.load(open(cache_file, "rb"))
        print("Finished reading %s dataset from %s" % (split, cache_file), flush=True)

    else:
	'''
    print("building %s dataset" % split, flush=True)
    from utils.dataset import SQuAD
    dataset = SQuAD(json_file, itos, stoi, itoc, ctoi, debug_mode=is_debug, split=split, wv_vec=wv_vec)
    '''
    if split != "error":
        pickle.dump(dataset, open(cache_file, "wb"))
    '''
    return dataset


if __name__ == "__main__":
    import sys, traceback, pdb
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
