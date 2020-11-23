"""
Created bt tz on 2020/10/28 
"""

__author__ = 'tz'

# 实现一个word2index index2word word2count 词典
import torch
import numpy
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Lang:
    """
    input lang对象
    output lang 对象
    """
    def __init__(self, name):
        super(Lang, self).__init__()
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2  # count SOS and EOS

    def add_sentence(self, sentence):
        """
        输入一条句子，对句子进行切分，
        进入添加单词函数逻辑处理
        :param sentence: 空格分开的句子
        :return:
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        添加每个字，将每个字添加到 word2index,index2word word2count中
        :param word: 单个单词
        :return:
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



"""
读取数据，进行简单的长度过滤处理，返回输入输出的lang对象
"""

def read_lang(lang1, lang2, reverse=False):
    """
    读取文件，创建input_lang,output_lang对象并返回
    创建一个 input output 的pairs，包含所有句子
    """
    print("Reading lines...")

    # 读取数据放入列表
    with open('chatdata_all.txt', encoding='utf-8') as f:
        # 这里的pairs列表是已经分词完成之后的
        # 如果是原始文本，则需要对文本的预处理，
        lines = f.read().strip().split('\n')
        pairs = [[s for s in l.split('@@')] for l in lines] # 简单的输入和输出的分割


    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filterPair(p):
    """
    过滤问答数据中句子长度小于最小长度的句子
    :param p: True / False
    :return:
    """
    # 过滤回答数小于MAX_LENGTH长度的句子
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_lang(lang1, lang2, reverse)
    # 文件中读取到的句子数量
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    # 打印过滤之后的句子数目
    print("filter Pairs after,Read %s sentence pairs" % len(pairs))
    # print("Counting work")
    # 处理过之后的数据加入到 问答lang 对象中，生成对应的index和word对应字典，和wordcount计数
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    # print("Counted words")
    # print(input_lang.name, input_lang.n_words)
    # print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# 返回一个input，output lang对象 和 输入输出句子后的组合
# input_lang, output_lang, pairs = prepare_data('human_lang', 'machine_lang')


"""
实现一个pair数据转tensor
word2idx
"""

# 句子转index
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

#句子转tensor
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes,dtype=torch.long).view(-1, 1)

#句子对转index
def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


"""
返回输入输出lang对象，对象属性包含 index2word word2index word2count
原始 pairs
打乱的tensor : train_pairs
tain_pairs shape：[n,1]  // 单词个数
"""
def train_data_main():
    """
    返回模型输入数据
    """
    input_lang, output_lang, pairs = prepare_data('human_lang','machine_lang')
    # train_pairs = (tensorsFromPair(random.choice(pairs)) for i in range(pairs))
    train_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(len(pairs))]
    return input_lang, output_lang, pairs, train_pairs


class Train_dataset(Dataset):

    def __init__(self):
        super(Train_dataset, self).__init__()
        self.train_pairs = self.get_data()

    def get_data(self):
        _,_,_,train_pairs = train_data_main()
        # print(train_pairs)
        return train_pairs

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        return self.train_pairs[idx][0], self.train_pairs[idx][1]

def pad_tensor(vec, pad, dim):
    """
    vec 要pad的tensor
    pad the size to pad to,最长的序列长度
    dim dimension to pad
    :return:
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    # 这里不知道怎么处理 longtensor的错误
    # 添加dtype 和 torch.LongTensor都不可以
    tmp = numpy.concatenate((vec.numpy(), torch.zeros(*pad_size)),axis=0)

    return torch.LongTensor(tmp)


class PadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_collate(self, batch):
        """
        batch list of (input, output)
        :param batch:
        :return:
        """
        # print(type(batch))
        # print(batch)
        # 寻找最长序列
        max_len_input = max(map(lambda x:x[0].shape[self.dim], batch))
        max_len_output = max(map(lambda x:x[1].shape[self.dim], batch))
        # 根据最长序列进行pad
        batch_input = map(lambda d:
                    (pad_tensor(d[0],pad=max_len_input,dim=self.dim),d[1]),batch)
        batch_output = map(lambda d:
                          (pad_tensor(d[1], pad=max_len_output, dim=self.dim), d[1]), batch)
        lbatch_input = list(batch_input)
        lbatch_output = list(batch_output)
        xs = torch.stack(tuple(map(lambda x:x[0],lbatch_input)), dim=0)
        ys = torch.stack(tuple(map(lambda x: x[0], lbatch_output)), dim=0)
        return xs,ys

    def __call__(self, batch):
        return self.pad_collate(batch)


train_loader = DataLoader(Train_dataset(), batch_size=128, shuffle=True,
                          collate_fn=PadCollate(dim=0))

