## 采用seq2seq Attention实现简单对话
> 语料采用小黄鸡中文对话预料  

`data_dataset.py`通过继承dataset类通过自定义collate_fn实现同意batch句子填充。

直接运行`train.py`即可。
