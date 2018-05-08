import os
import sys
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

from utilz import PtrGenSample as Sample
from utilz import PAD,  word_tokenize
from utilz import VOCAB
from anikattu.utilz import pad_seq

from anikattu.utilz import logger
from anikattu.vocab import Vocab
from anikattu.tokenstring import TokenString
from anikattu.utilz import LongVar, Var, init_hidden
import numpy as np

import glob

SELF_NAME = os.path.basename(__file__).replace('.py', '')

def load_squad_data(data_path, ids, max_para_len=600, max_ans_len=10, max_sample_size=None):
    samples = []
    qn, an = 0, 0
    skipped = 0

    vocabulary = defaultdict(int)
    
    try:
        for i, file_ in enumerate(glob.glob('dataset/en-10k/qa*_train.txt')):
            dataset = open(file_).readlines()
            prev_linenum = 1000000
            for line in dataset:
                questions, answers = [], []
                linenum, line = line.split(' ', 1)

                linenum = int(linenum)
                if prev_linenum > linenum:
                    story = ''

                if '?' in line:
                    q, a, _ = line.split('\t')

                    samples.append(
                        Sample('{}.{}'.format(i, linenum),
                               i, linenum,
                               TokenString(story, word_tokenize),
                               TokenString(q,     word_tokenize),
                               TokenString(a,     word_tokenize))
                        )

                else:
                    story += ' ' + line

                prev_linenum = linenum

    except:
        skipped += 1
        log.exception('{}'.format(i, linenum))

    print('skipped {} samples'.format(skipped))
    samples = sorted(samples, key=lambda x: -len(x.a + x.story))
    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building vocabulary...')
    for sample in samples:
        for token in sample.story + sample.q + sample.a:
            vocabulary[token] += 1
    return samples, vocabulary


# ## Loss and accuracy function
def process_output(decoding_index, output, batch,  *args, **kwargs):
    indices, (story, question), (answer, extvocab_story, target, extvocab_size) = batch
    pgen, vocab_dist, hidden, context, attn_dist, coverage = output

    vocab_dist, attn_dist = pgen * vocab_dist, (1-pgen) * attn_dist
    batch_size, vocab_size = vocab_dist.size()
    output  = vocab_dist
    if extvocab_size:
        zeros      = Var( torch.zeros(batch_size, extvocab_size) )
        vocab_dist = torch.cat( [vocab_dist, zeros], dim=-1 )
        output     = vocab_dist.scatter_add_(1, extvocab_story, attn_dist)

    return output

def process_predictor_output(decoding_index, output, batch, UNK):
    indices, (story, question), __  = batch
    pgen, vocab_dist, hidden, context,  attn_dist, coverage = output
    output = process_output(decoding_index, output, batch)
    output = F.log_softmax(output, dim=-1)
    output = output.max(1)[1]
    
    return output, (output.masked_fill_(output > vocab_dist.size(1), UNK), hidden, context, coverage + attn_dist)


def loss(decoding_index, output, batch, loss_function, UNK, *args, **kwargs):
    _, (story, _), (answer, ___ , target, ____ ) = batch
    pgen, vocab_dist, hidden, context, attn_dist, coverage = output
    cov_loss = torch.sum(torch.min(attn_dist, coverage) * (story > 0).float() , 1)
    
    output = process_output(decoding_index, output, batch)
    #output = F.log_softmax(output, -1)
    
    probs = output.gather(1, target[:, decoding_index].unsqueeze(1)).squeeze()
    loss_ = -torch.log(probs + 1e-9) + cov_loss * Config.cov_lr
    loss_ = loss_ * (target[:, decoding_index] > 0).float()

    """
    print("#######################################\n####################################################\n#######################################################################")
    [print(i) for i in zip(output.max(1)[1], target[:, decoding_index])]
    """

    #pprint(locals())
    
    return (
        (
            loss_
        ).mean(),
        
        (
            output.max(1)[1].masked_fill_(output.max(1)[1] > vocab_dist.size(1), UNK),
            hidden,
            context,
            coverage + attn_dist
        )
    )


def accuracy(decoding_index, output, batch, UNK, *args, **kwargs):
    _, __, (answer, ___ , target, ____ ) = batch
    pgen, vocab_dist, hidden, context, attn_dist, coverage = output
    output = process_output(decoding_index, output, batch)
    mask = target[:, decoding_index] > 0
    accu = (output.max(1)[1] == target[:, decoding_index]).float() * mask.float()
    return accu.sum()/output.size(0)


def f1score(output, batch, *args, **kwargs):
    p, r, f1 = 0.0, 0.0, 0.0

    _, __, (answer, ___ , target, ____ ) = batch
    output = torch.stack([o[0] for o in output]).transpose(0,1)
    batch_size = output.size(0)
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


def repr_function(output, batch, VOCAB, raw_samples):
    indices, (story, question), (answer, extvocab_story, target, extvocab_size) = batch
    
    results = []
    output = output.transpose(0,1).cpu().numpy()
    for idx, c, q, a, o in zip(indices, story, question, answer, output):

        c = ' '.join([VOCAB[i] for i in c])
        q = ' '.join([VOCAB[i] for i in q])
        a = ' '.join([ extvocab_story[i - len(VOCAB)]
               if i >= len(VOCAB)
               else VOCAB[i]
               for i in a])

        o = ' '.join([ extvocab_story[i - len(VOCAB)]
                           if i >= len(VOCAB)
                           else VOCAB[i]
                           for i in o])
        
        results.append([ c, q, a, o ])
        
    return results

def batchop(datapoints, WORD2INDEX, *args, **kwargs):
    indices = [d.id for d in datapoints]
    story = []
    question = []
    answer = []
    extvocab_story = []
    extvocab_answer = []
    
    def build_oov(d, WORD2INDEX):
        oov = [w for w in d.story + d.q + d.a if WORD2INDEX[w] == UNK]
        oov = list(set(oov))
        return oov
        
    UNK = WORD2INDEX['UNK']
    extvocab_size = 0
    for d in datapoints:
        story.append([WORD2INDEX[w] for w in d.story] + [WORD2INDEX['EOS']])
        question.append([WORD2INDEX[w] for w in d.q] + [WORD2INDEX['EOS']])
        
        answer.append([WORD2INDEX[w] for w in d.a] + [WORD2INDEX['EOS']])

        oov = build_oov(d, WORD2INDEX)
        extvocab_story.append(
            [ oov.index(w) + len(WORD2INDEX) if WORD2INDEX[w] == UNK else WORD2INDEX[w]
              for w in d.story] + [WORD2INDEX['EOS']]
        )
        
        extvocab_answer.append(
            [ oov.index(w) + len(WORD2INDEX) if WORD2INDEX[w] == UNK else WORD2INDEX[w]
              for w in d.a] + [WORD2INDEX['EOS']]
        )

        extvocab_size = max(extvocab_size, len(oov))
        
        
    story  = LongVar(pad_seq(story))
    question = LongVar(pad_seq(question))
    answer   = LongVar(pad_seq(answer))
    extvocab_answer   = LongVar(pad_seq(extvocab_answer))
    extvocab_story = LongVar(pad_seq(extvocab_story))
    
    batch = indices, (story, question), (answer, extvocab_story, extvocab_answer, extvocab_size)
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
        self.print_instance = 1
        
    def cpu(self):
        super(Base, self).cpu()
        return self
    
    def cuda(self):
        super(Base, self).cuda()
        return self
    
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n):
        return '{}.{}'.format(self._name, n)
    
class Encoder(Base):
    def __init__(self, Config, name, input_vocab_size):
        super(Encoder, self).__init__(Config, name)

        self.embed_size = Config.embed_size
        self.hidden_size = Config.hidden_size
        self.embed = nn.Embedding(input_vocab_size, self.embed_size)
        
        self.encode_story  = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True)
        self.encode_question = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        self.attn_story  = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.attn_question = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        
        
        if Config.cuda:
            self.cuda()
            
    def forward(self, input_):
        idxs, inputs, targets = input_
        story, question = inputs
        story = self.__( story, 'story')
        question = self.__(question, 'question')

        batch_size, story_size  = story.size()
        batch_size, question_size = question.size()
        
        story  = self.__( self.embed(story),  'story_emb')
        question = self.__( self.embed(question), 'question_emb')

        story  = story.transpose(1,0)
        story, _  = self.__(  self.encode_story(
            story,
            init_hidden(batch_size, self.encode_story)), 'C'
        )
        
        question  = question.transpose(1,0)
        question, _ = self.__(  self.encode_question(
            question,
            init_hidden(batch_size, self.encode_question)), 'Q'
        )

        story = self.__( story.transpose(0,1), 'story')
        question_state = self.__( question[-1], 'question')
        question_state = self.__( question_state.unsqueeze(1).expand_as(story), 'question')
        merged = self.__(   self.attn_story(story)
                          + self.attn_question(question_state),  'merged' )
        
        return merged.transpose(0,1)
                

class Attention(Base):
    def __init__(self, Config, name, size):
        super(Attention, self).__init__(Config, name)
        self.size = size
        self.attn_hidden =  nn.Linear(self.size, self.size)
        self.attn_story =  nn.Linear(self.size, self.size)
        self.attn_coverage =  nn.Linear(        1, self.size)
        self.squash_attn =  nn.Linear(self.size, 1)
        
    def forward(self, story, seq_mask,  hidden, coverage):
        seq_len, batch_size, hidden_size = story.size()
        
        story        = self.__( story.transpose(0, 1), 'story')
        attn_story   = self.__( self.attn_story(story.contiguous().view(-1, hidden_size)), 'attn_story')
        attn_hidden  = self.__( self.attn_hidden(hidden), 'attn_hidden')
        attn_hidden  = self.__( attn_hidden.unsqueeze(1).expand_as(story), 'attn_hidden')

        attn      = self.__( attn_story + attn_hidden.contiguous().view(-1, hidden_size), 'attn')

        attn_coverage  = self.__( self.attn_coverage(coverage.contiguous().view(-1, 1)), 'attn_coverage')

        attn      = self.__( attn + attn_coverage, 'attn')
        attn      = self.__( F.tanh(attn), 'attn')
        scores    = self.__( self.squash_attn(attn).contiguous().view(batch_size, seq_len), 'scores')

        attn_dist = self.__( F.softmax(scores, dim=-1) * seq_mask, 'attn_dist')
        attn_dist = self.__( attn_dist / attn_dist.sum(1, keepdim=True), 'attn_dist')

        context   = self.__( torch.bmm(attn_dist.unsqueeze(1),
                                       story.contiguous().view(batch_size, seq_len, hidden_size)), 'attn_story')
        context   = self.__( context.contiguous().view(-1, hidden_size), 'attn_story')

        coverage  = self.__( coverage + attn_dist, 'coverage')
 
        return context, attn_dist, coverage
        
class PtrDecoder(Base):
    def __init__(self, Config, name, embedding, initial_decoder_input, output_vocab_size):
        super(PtrDecoder, self).__init__(Config, name)

        self.hidden_size  = 2 * Config.hidden_size
        self.output_vocab_size = output_vocab_size
        
        self.embed = embedding
        self.initial_decoder_input = initial_decoder_input
        self.dropout = nn.Dropout(Config.dropout)

        self.input_context = nn.Linear(self.embed.embedding_dim + self.hidden_size, self.hidden_size)

        self.attn = Attention(Config, self.name('attn'), self.hidden_size)
        self.add_module('attn', self.attn)
        
        self.squash_pgen  = nn.Linear(self.hidden_size, 1)
        
        self.decode = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.project_output = nn.Linear(2 * self.hidden_size, self.output_vocab_size)

    def train(self, mode=True):
        self.attn.train(mode)
        super().train(mode)

    def eval(self):
        self.attn.eval()
        super().eval()
        
    def initial_input(self, input_, encoder_output):
        story_states = self.__( encoder_output, 'encoder_output')
        seq_len, batch_size, hidden_size = story_states.size()
        decoder_input = self.__( LongVar([self.initial_decoder_input]).expand(batch_size), 'decoder_input')
        hidden = self.__( story_states[-1], 'hidden')
        context, _ = self.__( story_states.max(0), 'context')
        coverage = self.__( Var(torch.zeros(batch_size, seq_len)), 'coverage')
            
        return decoder_input, hidden, context,  coverage
        
    def forward(self, input_, encoder_output, decoder_input):
        story_states = self.__( encoder_output, 'encoder_output')
        seq_len, batch_size, hidden_size = story_states.size()
        decoder_input, hidden, context, coverage = decoder_input
        
        _, (story, question), __ = input_
        seq_mask = (story > 0).float()        

        decoder_input  = self.__(  self.embed(decoder_input), 'decoder_input')        
        input_context  = self.__( torch.cat([decoder_input, context], dim=-1), 'input_context')

        decoder_input  = self.__( self.input_context(input_context), 'decoder_input')
        hidden = self.__( self.decode(decoder_input, hidden), 'decoder_output')

        context, attn_dist, coverage = self.attn(story_states, seq_mask, hidden, coverage)
                
        pgen_vector = self.__(   decoder_input + context + hidden, 'pgen_vector')        
        pgen        = self.__( F.sigmoid(self.squash_pgen(pgen_vector)), 'pgen')        

        vocab_dist = self.__( self.project_output(
            torch.cat([hidden, context], dim=-1)
        ), 'vocab_dist')
        
        return (pgen,
                F.softmax(vocab_dist, -1),
                hidden, context,
                F.softmax(attn_dist, -1),
                coverage)
            
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
        train_feed     = DataFeed(name, datapoints[0], batchop=_batchop, batch_size=100)
        test_feed      = DataFeed(name, datapoints[1], batchop=_batchop, batch_size=100)
        predictor_feed = DataFeed(name, datapoints[1], batchop=_batchop, batch_size=100)

        _loss = partial(loss, loss_function=nn.NLLLoss(), UNK=VOCAB['UNK'])
        _accuracy = partial(accuracy,  UNK=VOCAB['UNK'])
        trainer = Trainer(name=name,
                          model=(encoder, decoder),
                          loss_function=_loss, accuracy_function=_accuracy, f1score_function=f1score,
                          checkpoint=checkpoint, epochs=epochs,
                          feeder = Feeder(train_feed, test_feed))

        _repr_function=partial(repr_function, VOCAB=VOCAB, raw_samples=raw_samples)
        _process_predictor_output = partial(process_predictor_output, UNK=VOCAB['UNK'])
        predictor = Predictor(model = (encoder, decoder),
                              feed  = predictor_feed,
                              repr_function  = _repr_function,
                              process_output = _process_predictor_output)

        dump = open('results/experiment_attn.csv', 'w')        
        for e in range(eons):
            log.info('on {}th eon'.format(e))

            dump.write('#========================after eon: {}\n'.format(e))
            results = ListTable()
            for ri in tqdm(range(predictor_feed.num_batch//10)):
                output, _results = predictor.predict(predictor_feed.num_batch - ri, 3)
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

    if Config.flush:
        log.info('flushing...')
        ids = tuple((Sample._fields.index('id'),))
        dataset, vocabulary = load_squad_data('dataset/train-v1.1.json', ids)
        pickle.dump([dataset, dict(vocabulary)], open('train.squad', 'wb'))
    else:
        dataset, _vocabulary = pickle.load(open('train.squad', 'rb'))
        vocabulary = defaultdict(int)
        vocabulary.update(_vocabulary)
        
    log.info('dataset size: {}'.format(len(dataset)))
    log.info('dataset[:10]: {}'.format(pformat(dataset[0])))
    log.info('vocabulary: {}'.format(
        pformat(
            sorted(
                vocabulary.items(), key=lambda x: x[1], reverse=True)
        )))
    
    VOCAB = Vocab(vocabulary, VOCAB, freq_threshold=100)
    pprint(VOCAB.word2index)
    if 'train' in sys.argv:
        labelled_samples = [d for d in dataset if len(d.a) > 0] #[:100]
        pivot = int( Config().split_ratio * len(labelled_samples) )
        random.shuffle(labelled_samples)
        train_set, test_set = labelled_samples[:pivot], labelled_samples[pivot:]
        
        train_set = sorted(train_set, key=lambda x: -len(x.a + x.story))
        test_set  = sorted(test_set, key=lambda x: -len(x.a + x.story))
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
