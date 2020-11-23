"""
Created bt tz on 2020/10/28 
"""

__author__ = 'tz'
# https://blog.csdn.net/qq_37236745/article/details/107085532
import random
import torch.optim as optim
import torch
import torch.nn as nn
from torch.functional import F

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src,hidden0=None):
        "src: [batch_size, src_len]"

        # embedded [batch_size, seq_len, embed_size]
        embedded = self.embedding(src)
        # [seq_len, batch_size, embed_size]
        embedded = embedded.transpose(0,1)
        # outputs: [seq_len, batch_size, enc_hid_dim*2]
        # s : [2, batch_size, enc_hid_dim]
        outputs, s = self.gru(embedded, hidden0)
        s = torch.tanh(self.linear(torch.cat((s[-1,:,:],s[-2,:,:]),dim=1)))
        return outputs, s

class Attention(nn.Module):
    ""
    def __init__(self,enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        # encoder_hidden_dim和decoder_hidden_dim
        self.attn = nn.Linear(enc_hid_dim*2+dec_hid_dim, dec_hid_dim,bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        """
        :param s: (batch_size, dec_hid_dim)
        :param enc_output: (seq_len, batch_size, enc_hidden_dim)
        :return:
        """
        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # 对于Linear层，我们第一个维度需要是batch_size
        # 将s维度变为 (batch_size, seq_len, dec_hid_dim)
        # enc_output维度 (batch_size, seq_len, enc_hidden_dim)
        s = s.unsqueeze(1).repeat(1,src_len,1)
        enc_output = enc_output.transpose(0,1)

        # (batch_size, seq_len, 1).squeeze(2)
        score = self.v(torch.tanh(self.attn(torch.cat((s,enc_output),dim=2)))).squeeze(2)

        return F.softmax(score)

class Decoder(nn.Module):
    "计算单个单词输入输出"
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim) # output dim输出的词典大小
        self.gru = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim*2+dec_hid_dim+emb_dim,output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        """
        :param dec_input: [dec_input:  batch_size]
        :param s:  [batch_size, dec_hid_dim]
        :param enc_output:  [src_len, batch_size,enc_hid_dim*2]
        :return:
        """
        dec_input = dec_input.unsqueeze(1) # [batch_size,1]
        embedded = self.dropout(self.embedding(dec_input)).transpose(0,1) # (1, batch_size,emb_dim)
        weights = self.attention(s, enc_output).unsqueeze(1) # [batch_size, 1, src_len]
        # print('weight',weights.shape)
        enc_output = enc_output.transpose(0,1)  # [batch_size, src_len, enc_hid_dim*2]

        c = torch.bmm(weights, enc_output) # [batch_size, 1, enc_hid_dim*2]
        c = c.transpose(0,1)               # [1, batch_size, enc_hid_dim*2]
        rnn_input = torch.cat((c, embedded),dim=2)
        # dec_output = [len=1, batch_size, dec_hid_dim]
        # dec_hidden = [1, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.gru(rnn_input, s.unsqueeze(0))

        embedded = embedded.squeeze(0)  # [batch_size, emb_dim]
        dec_output = dec_output.squeeze(0)  # [batch_size,dec_hid_dim]
        c = c.squeeze(0)  # [batch_size, enc_hid_dim*2]

        pred = self.fc_out(torch.cat((embedded, dec_output, c),dim=1))

        return pred, dec_hidden.squeeze(0)

class Seq2Seq(nn.Module):
    ""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src trg都是经过数据预处理的。
        :param src: [batch_size, src_len]
        :param trg: [batch_size, trg_len]
        :param teacher_forcing_ratio:  0.5
        :return:
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        enc_output, s = self.encoder(src)

        dec_input = trg[:,0]
        for t in range(1, trg_len):
            # s: [batch_size, dec_hid_dim]
            dec_output, s = self.decoder(dec_input, s, enc_output)

            outputs[t] = dec_output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = dec_output.argmax(1)
            dec_input = trg[:,t] if teacher_force else top1
        return outputs

if __name__ == '__main__':

    input_size = 2000
    output_size = 1500
    embed_size = 256
    hidden_size = 512
    dropout = 0.5
    # encoder 测试代码
    src = torch.LongTensor([1,3,4,1,2,0,0,0])
    src = src.unsqueeze(0) # batch_size, seq_len

    encoder = Encoder(input_size, embed_size, hidden_size)

    outputs, s = encoder(src)
    print(f"outputs shape:{outputs.shape}\n hidden shape: {s.shape}")

    # attention 测试
    attention = Attention(hidden_size, hidden_size)
    weights = attention(s, outputs)
    print(f"weight shape: {weights}")

    # decoder测试
    decoder = Decoder(output_size,embed_size,hidden_size,hidden_size,dropout,attention)
    # dec_input, s, enc_output
    pred, decode_hidden = decoder(torch.LongTensor([2]),s, outputs)
    print(f"pred shape: {pred.shape}, decoder hidden shape: {decode_hidden.shape}")

    # Seq2Seq测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq2seq = Seq2Seq(encoder, decoder, device).to(device)
    trg = torch.LongTensor([1,2,4,1,0,0]).unsqueeze(0)
    print(src.shape)
    print(trg.shape)
    out = seq2seq(src, trg)
    print(f"outputs shape: {out.shape}")

    loss = nn.CrossEntropyLoss()
    optim = optim.Adam(seq2seq.parameters(), lr=1e-3)
