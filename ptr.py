import os
import json
import time
import random
from pprint import pprint, pformat


from anikattu.logger import CMDFilter
import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from config import Config

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.trainer.seq2seq import Trainer, Feeder, Predictor
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.utilz import tqdm, ListTable

from functools import partial

from collections import namedtuple, defaultdict
import itertools

from utilz import SequenceSample as Sample
from utilz import PAD,  word_tokenize
from utilz import VOCAB
from anikattu.utilz import pad_seq

from anikattu.utilz import logger
from anikattu.vocab import Vocab
from anikattu.tokenstring import TokenString
from anikattu.utilz import LongVar, Var, init_hidden
import numpy as np


SELF_NAME = os.path.basename(__file__).replace('.py', '')

def load_squad_data(data_path, ids, max_para_len=600, max_ans_len=10):
    dataset = json.load(open(data_path,'r'))
    samples = []
    qn, an = 0, 0
    skipped = 0

    vocabulary = defaultdict(int)
    
    def __(s):
        import unicodedata
        s =  ''.join(
            c for c in unicodedata.normalize('NFKD', s)
            if unicodedata.category(c) != 'Mn'
        )
        return s.replace("``", '"').replace("''", '"')

    try:
        for aid, article in enumerate(tqdm(dataset['data'])):
            for pid, paragraph in enumerate(article['paragraphs']):

                context = TokenString(__(paragraph['context']), word_tokenize).delete_whitespace()
                questions = paragraph['qas']

                for token in context:
                    vocabulary[token] += 1
                    
                
                for qid, qa in enumerate(questions):
                    log.debug('processing: {}.{}.{}'.format(aid, pid, qid))
                    q = TokenString(__(qa['question']), word_tokenize).delete_whitespace()
                    a = TokenString(__(qa['answers'][0]['text']), word_tokenize).delete_whitespace()  #simply ignore other answers
                    squad_id = qa['id']
                    for token in q:
                        vocabulary[token] += 1

                    indices = context.index(a)
                    if not indices:
                        log.debug(pformat(paragraph['context']))
                        log.debug(pformat(paragraph['qas'][qid]))
                        log.error('{}.{}.{} - "{}" not found in \n"{}"'
                                  .format(aid, pid, qid, a.tokenized_string,
                                          context.tokenized_string))
                        skipped += 1
                        continue
                        
                    a_start, a_end = indices
                    fields = (aid, pid, qid, squad_id, context, q, a, list(range(a_start, a_end)))
                    _id = tuple( fields[i-1] for i in ids )
                    samples.append(Sample(_id, *fields))
    except:
        skipped += 1
        log.exception('{}'.format(aid))

    print('skipped {} samples'.format(skipped))
    return samples, vocabulary


# ## Loss and accuracy function
def loss(decoding_index, output, batch, loss_function=nn.NLLLoss(), scale=1, *args, **kwargs):
    indices, input_, (target, ) = batch
    target = target.transpose(0,1)[decoding_index]
    log.debug('i, o sizes: {} {}'.format(target.size(), output.size()))
    return loss_function(output, target)

def accuracy(decoding_index, output, batch, *args, **kwargs):    
    batch_size = output.size(1)
    indices, input_, (target, ) = batch
    target = target.transpose(0,1)[decoding_index]
    log.debug('i, o sizes: {} {}'.format(target.size(), output.size()))
    return (output.max(1)[1] ==  target).float().sum()/output.size(0)
        

def f1score(decoding_index, output, batch, *args, **kwargs):
    p, r, f1 = 0.0, 0.0, 0.0

    batch_size = output.size(1)
    indices, input_, (target, ) = batch
    output = output.transpose(0,1).transpose(1, 2).max(2)[1].data.tolist()
    
    for index, (o, t) in enumerate(zip(output, target)):
        tp = sum([oi in t for oi in o])
        fp = sum([oi not in t for oi in o])
        fn = sum([ti not in o for ti in t])

        if tp > 0:
            p  += tp/ (tp + fp)
            r  += tp/ (tp + fn)

    p, r = p/batch_size, r/batch_size
    if p + r > 0:
        f1 = 2*p*r/(p+r)
        
    return p, r, f1


def repr_function(output, feed, batch_index, VOCAB, raw_samples):
    results = []
    output = output.transpose(0, 1).data.tolist()
    indices, __, __ = feed.nth_batch(batch_index)

    for idx, op in zip(indices, output):
        sample = feed.data_dict[idx]
        _op = []
        for i in op:
            if i < len(sample.context):
                _op.append(i)
                
        results.append([ ' '.join(sample.context),
                         ' '.join(sample.q),
                         ' '.join(sample.context[i] for i in sample.a_positions),
                         ' '.join(sample.context[i] for i in _op),
        ])

    return results

def batchop(datapoints, WORD2INDEX, *args, **kwargs):
    indices = [d.id for d in datapoints]
    context = []
    question = []
    answer_positions = []
    answer_lengths = []
    for d in datapoints:
        context.append([WORD2INDEX[w] for w in d.context] + [WORD2INDEX['EOS']])
        question.append([WORD2INDEX[w] for w in d.q])
        
        answer_length = len(d.a_positions) + 1
        answer_positions.append([i for i in d.a_positions] + [len(d.context)])
        answer_lengths.append(answer_length)
        
    context = LongVar(pad_seq(context))
    question = LongVar(pad_seq(question))
    answer_positions = LongVar(pad_seq(answer_positions))
    answer_lengths = LongVar(answer_lengths)

    batch = indices, (context, question, answer_lengths), (answer_positions,)
    return batch

class Base(nn.Module):
    def __init__(self, Config, name):
        super(Base, self).__init__()
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
        self.log.setLevel(logging.INFO)
        self.size_log.setLevel(logging.INFO)
       
    def cpu(self):
        super(Base, self).cpu()
        return self
    
    def cuda(self):
        super(Base, self).cuda()
        return self
    
    def __(self, tensor, name=''):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))

        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
                
        return tensor

    def name(self, n):
        return '{}.{}'.format(self._name, n)
    
class Encoder(Base):
    def __init__(self, Config, name, input_vocab_size):
        super(Encoder, self).__init__(Config, name)

        self.embed_size = Config.embed_size
        self.hidden_size = Config.hidden_size
        self.embed = nn.Embedding(input_vocab_size, self.embed_size)
        
        self.encode_context = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True)
        self.encode_question = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        if Config.cuda:
            self.cuda()
            
    def forward(self, input_):
        idxs, inputs, targets = input_
        context, question, _ = inputs
        context = self.__( context, 'context')
        question = self.__(question, 'question')

        batch_size, context_size  = context.size()
        batch_size, question_size = question.size()
        
        context  = self.__( self.embed(context),  'context_emb')
        question = self.__( self.embed(question), 'question_emb')

        context  = context.transpose(1,0)
        C, _  = self.__(  self.encode_context(context, init_hidden(batch_size, self.encode_context)), 'C')
        
        question  = question.transpose(1,0)
        Q, _ = self.__(  self.encode_question(question, init_hidden(batch_size, self.encode_question)), 'Q')
        return F.tanh(C), F.tanh(Q)
                


class PtrDecoder(Base):
    def __init__(self, Config, name, embedding, initial_decoder_input, output_vocab_size):
        super(PtrDecoder, self).__init__(Config, name)

        self.hidden_size  = Config.hidden_size
        self.pooling_size = Config.pooling_size
        self.max_iter     = Config.max_iter
        self.output_vocab_size = output_vocab_size
        
        self.embed = embedding
        self.decode = nn.GRUCell(self.embed.embedding_dim, 2 * self.hidden_size)

        self.initial_decoder_input = initial_decoder_input
        self.dropout = nn.Dropout(Config.dropout)

        self.project_query = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size)
        self.attn = nn.Parameter(torch.zeros(2 * self.hidden_size, 2 * self.hidden_size))

        
    def initial_input(self, batch_size):
        decoder_input = LongVar([self.initial_decoder_input]).expand(batch_size)
        return decoder_input        
        
    def forward(self, encoder_output, decoder_input, input_):
        context_states, question_states = self.__( encoder_output, 'encoder_output')
        seq_len, batch_size, hidden_size = context_states.size()
        dropout = self.dropout
        decoder_input, hidden = decoder_input
        
        decoder_input  = self.__(  self.embed(decoder_input), 'decoder_input')
        if not isinstance(hidden, torch.Tensor):
            hidden = context_states[-1]
            
        hidden = self.__( self.decode(decoder_input, hidden), 'decoder_output')
        hidden = F.tanh(hidden)
        #combine question and current hidden state and project
        query = self.__( torch.cat([question_states[-1], hidden], dim=-1), 'query')
        query = self.__( self.project_query(query), 'projected query')
        query = F.tanh(query)
        
        sentinel = self.__( Var(torch.zeros(batch_size, context_states.size(2))), 'sentinel')
        pointer_predistribution = self.__(
            torch.cat([
                context_states,
                sentinel.unsqueeze(0),
            ], dim=0), 'pointer_predistribution').transpose(0,1)

        attn = self.__( self.attn.unsqueeze(0), 'attn')
        attn = self.__( torch.bmm(query.unsqueeze(1), attn.expand(batch_size, *self.attn.size())), 'attn')
        attn = self.__( torch.bmm(attn, pointer_predistribution.transpose(1, 2)), 'attn').squeeze(1)

        ret = self.__( F.log_softmax(attn, dim=-1), 'ret')
        return ret, hidden
    
class Decoder(Base):
    def __init__(self, Config, name, embedding, initial_decoder_input, output_vocab_size):
        super(PtrDecoder, self).__init__(Config, name)

        self.hidden_size  = Config.hidden_size
        self.pooling_size = Config.pooling_size
        self.max_iter     = Config.max_iter
        self.output_vocab_size = output_vocab_size
        
        self.embed = embedding
        self.decode = nn.GRUCell(self.embed.embedding_dim, 2 * self.hidden_size)

        self.initial_decoder_input = initial_decoder_input
        self.dropout = nn.Dropout(Config.dropout)
        self.linear = nn.Linear(2 * self.hidden_size, self.output_vocab_size)

    def initial_input(self, batch_size):
        decoder_input = LongVar([self.initial_decoder_input]).expand(batch_size)
        return decoder_input        
        
    def forward(self, encoder_output, decoder_input, input_):
        #batch_size, seq_len, hidden_size = encoder_output.size()
        context_states, question_states = encoder_output
        dropout = self.dropout
        decoder_input, hidden = decoder_input
        self.__(decoder_input, 'decoder_input')
        if not isinstance(hidden, torch.Tensor):
            hidden = context_states[-1]
            
        self.__(hidden, 'hidden')
            
        decoder_input = self.__(  self.embed(decoder_input), 'decoder_input')
        decoder_output = self.__( self.decode(decoder_input, hidden), 'decoder_output')
        ret = self.__( F.log_softmax(self.linear(dropout(decoder_output))), 'ret')
        return ret, decoder_output
        
def experiment(VOCAB, raw_samples, datapoints=[[], []], eons=1000, epochs=10, checkpoint=5):
    try:
        encoder =  Encoder(Config(), 'encoder', len(VOCAB))
        decoder =  PtrDecoder(Config(), 'decoder', encoder.embed, VOCAB['GO'], len(VOCAB))
        try:
            encoder.load_state_dict(torch.load('{}.{}.{}'.format(SELF_NAME, 'encoder', 'pth')))
            decoder.load_state_dict(torch.load('{}.{}.{}'.format(SELF_NAME, 'decoder', 'pth')))
            log.info('loaded the old image for the model')
        except:
            log.exception('failed to load the model')

        if Config().cuda:
            log.info('cuda the model...')
            encoder.cuda()
            decoder.cuda()

        model = (encoder, decoder)
        print('**** the model', model)

        name = os.path.basename(__file__).replace('.py', '')
        
        _batchop = partial(batchop, WORD2INDEX=VOCAB)
        train_feed     = DataFeed(name, datapoints[0], batchop=_batchop, batch_size=16)
        test_feed      = DataFeed(name, datapoints[1], batchop=_batchop, batch_size=16)
        predictor_feed = DataFeed(name, datapoints[1], batchop=_batchop, batch_size=16)

        _loss = partial(loss, loss_function=nn.NLLLoss())
        trainer = Trainer(name=name,
                          model=(encoder, decoder),
                          loss_function=_loss, accuracy_function=accuracy, #f1score_function=f1score,
                          checkpoint=checkpoint, epochs=epochs,
                          feeder = Feeder(train_feed, test_feed))

        _repr_function=partial(repr_function, VOCAB=VOCAB, raw_samples=raw_samples)
        predictor = Predictor(model=(encoder, decoder),
                              feed=predictor_feed, repr_function=_repr_function)

        dump = open('results/experiment_attn.csv', 'w')        
        for e in range(eons):
            log.info('on {}th eon'.format(e))

            dump.write('#========================after eon: {}\n'.format(e))
            results = ListTable()
            for ri in tqdm(range(predictor_feed.num_batch//10)):
                output, _results = predictor.predict(ri)
                results.extend(_results)
                
            dump.write(repr(results))
            dump.flush()
            if not trainer.train():
                raise Exception

    except :
        log.exception('####################')
        trainer.save_best_model()

        return locals()



    
import sys
import pickle
if __name__ == '__main__':

    if sys.argv[1]:
        log.addFilter(CMDFilter(sys.argv[1]))

    flush = False
    if flush:
        log.info('flushing...')
        ids = tuple((Sample._fields.index('squad_id'),))
        dataset, vocabulary = load_squad_data('dataset/train-v1.1.json', ids)
        pickle.dump([dataset, dict(vocabulary)], open('train.squad', 'wb'))
    else:
        dataset, _vocabulary = pickle.load(open('train.squad', 'rb'))
        vocabulary = defaultdict(int)
        vocabulary.update(_vocabulary)
        
    log.info('dataset size: {}'.format(len(dataset)))
    log.info('dataset[:10]: {}'.format(pformat(dataset[0])))
    log.info('vocabulary: {}'.format(len(vocabulary)))
    
    VOCAB = Vocab(vocabulary, VOCAB)
    if 'train' in sys.argv:
        labelled_samples = [d for d in dataset if len(d.a_positions) < 2000] #[:100]
        pivot = int( Config().split_ratio * len(labelled_samples) )
        random.shuffle(labelled_samples)
        train_set, test_set = labelled_samples[:pivot], labelled_samples[pivot:]
        
        train_set = sorted(train_set, key=lambda x: len(x.context))
        test_set  = sorted(test_set, key=lambda x: len(x.context))
        exp_image = experiment(VOCAB, dataset, datapoints=[train_set, test_set])
        
    if 'predict' in sys.argv:
        model =  BiLSTMDecoderModel(Config(), len(VOCAB),  len(LABELS))
        if Config().cuda:  model = model.cuda()
        model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, '.pth')))
        start_time = time.time()
        strings = sys.argv[2]
        
        s = [WORD2INDEX[i] for i in word_tokenize(strings)] + [WORD2INDEX['PAD']]
        e1, e2 = [WORD2INDEX['ENTITY1']], [WORD2INDEX['ENTITY2']]
        output = model(s, e1, e2)
        output = output.data.max(dim=-1)[1].cpu().numpy()
        label = LABELS[output[0]]
        print(label)

        duration = time.time() - start_time
        print(duration)
