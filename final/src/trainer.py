import datetime
import json
import os
import shutil
import sys
import time
from tqdm import tqdm
import torch
from tensorboard_logger import configure, log_value
from torch import optim
from torch.autograd import Variable

import models.r_net as RNet
from utils.squad_eval import evaluate
from utils.utils import make_dirs


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    checkpoint_regular = os.path.join(path, filename)
    checkpint_best = os.path.join(path, best_filename)
    torch.save(state, checkpoint_regular)
    if is_best:
        shutil.copyfile(checkpoint_regular, checkpint_best)

def load_checkpoint(path, filename='model_best.pth.tar'):
    checkpint_best = os.path.join(path, filename)
    state = torch.load(checkpint_best)
    return state

class Trainer(object):
    def __init__(self, args, dataloader_train, dataloader_dev, char_embedding_config, word_embedding_config,
                 sentence_encoding_config, pair_encoding_config, self_matching_config, pointer_config):

        # for validate
        '''
            expected_version = "1.1"
            with open(args.dev_json) as dataset_file:
                dataset_json = json.load(dataset_file)
                if dataset_json['version'] != expected_version:
                    print('Evaluation expects v-' + expected_version +
                          ', but got dataset with v-' + dataset_json['version'],
                          file=sys.stderr)
                self.dev_dataset = dataset_json['data']
        '''
        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev
        
        self.model = RNet.Model(args, char_embedding_config, word_embedding_config, sentence_encoding_config,
                                pair_encoding_config, self_matching_config, pointer_config)
        self.parameters_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = optim.RMSprop(self.parameters_trainable, lr=0.001)
        #self.optimizer = optim.Adadelta(self.parameters_trainable, rho=0.95)
        self.best_f1 = 0
        self.step = 0
        self.start_epoch = args.start_epoch
        self.name = args.name
        self.start_time = datetime.datetime.now().strftime('%b-%d_%H-%M')

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_f1 = checkpoint['best_f1']
                self.name = checkpoint['name']
                self.step = checkpoint['step']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_time = checkpoint['start_time']

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
        else:
            self.name += "_" + self.start_time

        # use which device
        if torch.cuda.is_available():
            self.model = self.model.cuda(args.device_id)
            # CHANGED: put embedding in cpu
            self.model.embedding.word_embedding_word_level.cpu()
        else:
            self.model = self.model.cpu()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        configure("log/%s" % (self.name), flush_secs=5)
        self.checkpoint_path = os.path.join(args.checkpoint_path, self.name)
        make_dirs(self.checkpoint_path)

    def train(self, epoch_num, load_model=True, load_model_and_keep_training=False):
        if load_model:
            state = load_checkpoint("checkpoint/r-net_Jan-17_12-12/", filename="model_best.pth.tar")
            self.model.load_state_dict(state["state_dict"])
            self.optimizer.load_state_dict(state["optimizer"])
            print("best f1 {}".format(state["best_f1"]))
            if not load_model_and_keep_training:
                return 

        for epoch in range(self.start_epoch, epoch_num):
            global_loss = 0.0
            global_acc = 0.0
            last_step = self.step - 1
            last_time = time.time()
            
            try:

                for batch_idx, batch_train in tqdm(enumerate(self.dataloader_train)):
                    loss, acc = self._forward(batch_train)
                    global_loss += loss.data[0]
                    global_acc += acc
                    self._update_param(loss)

                    if self.step % 10 == 0:
                        used_time = time.time() - last_time
                        step_num = self.step - last_step
                        print("step %d / %d of epoch %d)" % (batch_idx, len(self.dataloader_train), epoch), flush=True)
                        print("loss: ", global_loss / step_num, flush=True)
                        print("acc: ", global_acc / step_num, flush=True)
                        print("dev best_f1 :", self.best_f1, flush=True)
                        speed = self.dataloader_train.batch_size * step_num / used_time
                        print("speed: %f examples/sec \n\n" %
                              (speed), flush=True)

                        log_value('train/EM', global_acc / step_num, self.step)
                        log_value('train/loss', global_loss / step_num, self.step)
                        log_value('train/speed', speed, self.step)

                        global_loss = 0.0
                        global_acc = 0.0
                        last_step = self.step
                        last_time = time.time()
                    self.step += 1
                    #############
                    
                    #############
            except KeyboardInterrupt:
                pass
            exact_match, f1 = self.eval()
            print("(exact_match: %f)" % exact_match, flush=True)
            print("(f1: %f)" % f1, flush=True)

            log_value('dev/f1', f1, self.step)
            log_value('dev/EM', exact_match, self.step)

            if f1 > self.best_f1:
                is_best = True
                self.best_f1 = f1
            else:
                is_best = False

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'step': self.step + 1,
                'best_f1': self.best_f1,
                'name': self.name,
                'optimizer': self.optimizer.state_dict(),
                'start_time': self.start_time
            }, is_best, self.checkpoint_path)
        # Load best model to predict
        state = load_checkpoint(self.checkpoint_path)
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])



    def eval(self):
        # special control signal
        USE_MAO = True
        # 
        self.model.eval()
        # qid map to tuple (start, end)
        if USE_MAO:
            f1 = 0.
            f1_divisor = 0.
        pred_result = {}
        import pdb
        
        for _, batch in tqdm(enumerate(self.dataloader_dev)):
            
            question_ids, questions, passages, passage_tokenized, items = batch
            questions.variable(volatile=True)
            passages.variable(volatile=True)
            begin_, end_ = self.model(questions, passages)  # batch x seq

            _, pred_begin = torch.max(begin_, 1)
            _, pred_end = torch.max(end_, 1)

            pred = torch.stack([pred_begin, pred_end], dim=1)

            for i, (begin, end) in enumerate(pred.cpu().data.numpy()):
                if begin > end:
                    max_value = float("-inf")
                    end = -1
                    for j, value in enumerate(end_[i].cpu().data.numpy()):
                        if j >= begin:
                            if value > max_value:
                                max_value = value
                                end = j
                # accumulate start (char by char)
                ans_start =  sum([len(passage_tokenized[i][j]) for j in range(0, begin)])
                # accumulate end (char by char)
                ans_end = sum([len(passage_tokenized[i][j]) for j in range(0, end+1)])-1
                # concatenatation ex. 5-8 => 5 6 7 8
                ans_indices = " ".join([str(j) for j in range(ans_start, ans_end+1)])
                if USE_MAO:
                    ground_start = items[i].answer_start
                    ground_end = items[i].answer_start + len(items[i].answer_text) - 1
                    left = min(ans_start, ground_start)
                    right = max(ans_end, ground_end)
                    span = right - left + 1
                    left_residual = max(ans_start, ground_start) - left
                    right_residual = right -  min(ans_end, ground_end)
                    overlap = span - left_residual - right_residual
                    if overlap > 0:
                        precision = float(overlap) / (ans_end - ans_start + 1)
                        recall = float(overlap) / (ground_end - ground_start + 1)
                    else:
                        precision = 0.
                        recall = 0.
                    assert precision >= 0 and recall >= 0 

                    f1 += (2 * precision * recall) / (precision + recall) if (overlap > 0.) else 0
                    f1_divisor += 1
                else:
                    qid = question_ids[i]
                    pred_result[qid] = (ans_start, ans_end) 
                '''
                ans = passage_tokenized[i][begin:end + 1]
                qid = question_ids[i]
                pred_result[qid] = " ".join(ans)
                '''
        self.model.train()
        
        if USE_MAO:
            return -1, f1 / f1_divisor
        else:
            return evaluate(self.dev_dataset, pred_result)

    def predict(self, test_dataloader, print_it=False):
        import pdb
        print("Length of test_json: {}".format(len(test_dataloader)))
        self.model.eval()
        pred_result = []
        RETREIVE_PROB = True
        if RETREIVE_PROB:
            retreive_prob = []
            qid_list = []
            passage_list = []
            offset_list = []
        for _, batch in tqdm(enumerate(test_dataloader)):
            question_ids, questions, passages, passage_tokenized, items = batch
            questions.variable(volatile=True)
            passages.variable(volatile=True)
            begin_, end_ = self.model(questions, passages)  # batch x seq
            #
            _, pred_begin = torch.max(begin_, 1)
            _, pred_end = torch.max(end_, 1)
            pred = torch.stack([pred_begin, pred_end], dim=1)
            for i, (begin, end) in enumerate(pred.cpu().data.numpy()):
                #
                if RETREIVE_PROB:
                    retreive_prob.append((begin_[i].cpu().data.numpy(), end_[i].cpu().data.numpy()))
                    qid_list.append(question_ids[i])
                    passage_list.append(passage_tokenized[i])
                    offset_list.append(items[i].offset)
                #
                # WARNING:
                if begin > end:
                    max_value = float("-inf")
                    end = -1
                    for j, value in enumerate(end_[i].cpu().data.numpy()):
                        # BUG: If begin is length of sequence length
                        if j >= begin:
                            if value > max_value:
                                max_value = value
                                end = j
                
                #
                # accumulate start (char by char)
                ans_start = sum([len(passage_tokenized[i][j]) for j in range(0, begin)])
                ans_start += items[i].offset
                # accumulate end (char by char)
                ans_end = sum([len(passage_tokenized[i][j]) for j in range(0, end+1)])-1
                ans_end += items[i].offset
                # concatenatation ex. 5-8 => 5 6 7 8
                if print_it:
                    print("".join(passage_tokenized[i]))
                    print("".join(passage_tokenized[i])[ans_start-items[i].offset:ans_end-items[i].offset+1])
                    
                    return ("".join(passage_tokenized[i])[ans_start-items[i].offset:ans_end-items[i].offset+1], passage_tokenized[i], begin_[i].cpu().data.numpy(), end_[i].cpu().data.numpy())
                ans_indices = " ".join([str(j) for j in range(ans_start, ans_end+1)])

                qid = question_ids[i]
                
                pred_result.append((qid, ans_indices))
        self.model.train()
        import numpy as np
        import pickle
        if RETREIVE_PROB:
            pickle.dump(retreive_prob, open("retreive_prob.pkl", "wb"))
            # No need to produce these
            # pickle.dump(qid_list, open("qid_list.pkl", "wb"))
            # pickle.dump(passage_list, open("passage_list.pkl", "wb"))
            # pickle.dump(offset_list, open("offset_list.pkl", "wb"))
        return np.array(pred_result)

    def _forward(self, batch):
        #import pdb
        #pdb.set_trace()
        _, questions, passages, answers, answers_texts = batch
        batch_num = questions.tensor.size(0)

        questions.variable()
        passages.variable()

        begin_, end_ = self.model(questions, passages)  # batch x seq
        assert begin_.size(0) == batch_num

        answers = Variable(answers)
        if torch.cuda.is_available():
            answers = answers.cuda()
        begin, end = answers[:, 0], answers[:, 1]
        #
        loss = self.loss_fn(begin_, begin) + self.loss_fn(end_, end) 

        _, pred_begin = torch.max(begin_, 1)
        _, pred_end = torch.max(end_, 1)

        exact_correct_num = torch.sum(
            (pred_begin == begin) * (pred_end == end))

        em = exact_correct_num.data[0] / batch_num

        return loss, em

    def _update_param(self, loss):
        self.model.zero_grad()
        loss.backward()
        # TODO: Gradient Clipping
        _ = torch.nn.utils.clip_grad_norm(self.parameters_trainable, 50)
        self.optimizer.step()
