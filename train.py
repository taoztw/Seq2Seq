"""
Created bt tz on 2020/10/28 
"""

__author__ = 'tz'

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from seq2seq_Attention.model import Attention, Encoder, Decoder, Seq2Seq
from seq2seq_Attention.data_pre import train_data_main
from seq2seq_Attention.data_dataset import train_loader

input_lang, output_lang, pairs, train_pairs = train_data_main()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = input_lang.n_words
OUTPUT_DIM = output_lang.n_words
ENC_EMB_DIM = 256  # encode embedding dimension
DEC_EMB_DIM = 256  # decode embedding dimension
ENC_HID_DIM = 512  # encode hidden dimension
DEC_HID_DIM = 512  # decode hidden dimension
ENC_DROPOUT = 0.5  # encode dropout
DEC_DROPOUT = 0.5  # decode dropout

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(encoder, decoder, device).to(device)


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[0]
        trg = batch[1]  # trg = [trg_len, batch_size]
        # pred = [trg_len, batch_size, pred_dim]
        src = src.squeeze().transpose(0, 1)
        trg = trg.squeeze().transpose(0, 1)
        pred = model(src, trg)

        pred_dim = pred.shape[-1]

        # trg = [(trg len - 1) * batch size]
        # pred = [(trg len - 1) * batch size, pred_dim]

        # trg = trg[1:].view(-1) # view只能在连续的tensor上起作用，这里需要.contiguous()方法返回连续
        trg = trg.contiguous().view(-1)
        pred = pred.view(-1, pred_dim)

        loss = criterion(pred, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]  # trg = [trg_len, batch_size]
            src = src.squeeze().transpose(0, 1)
            trg = trg.squeeze().transpose(0, 1)
            # output = [trg_len, batch_size, output_dim]
            output = model(src, trg, 0)  # turn off teacher forcing

            output_dim = output.shape[-1]

            # trg = [(trg_len - 1) * batch_size]
            # output = [(trg_len - 1) * batch_size, output_dim]
            output = output.contiguous().view(-1, output_dim)
            trg = trg.contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')

for epoch in range(10):
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss = evaluate(model, train_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


# 加载模型 并 test
# model.load_state_dict(torch.load('tut3-model.pt'))
#
# test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')