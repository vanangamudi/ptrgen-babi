import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class Config(Base):
    split_ratio = 0.90
    hidden_size = 200
    embed_size = 200
    batch_size = 2
    pooling_size = 8
    max_iter = 4
    dropout = 0.1
    cuda = True
    tqdm = True
    flush = True
    vocab_limit= 1000
    cov_lr = 1
    max_grad_norm=2.0

    class Log(Base):
        class _default(Base):
            level=logging.CRITICAL
        class PREPROCESS(Base):
            level=logging.DEBUG
        class MODEL(Base):
            level=logging.INFO
        class TRAINER(Base):
            level=logging.INFO
        class DATAFEED(Base):
            level=logging.INFO
