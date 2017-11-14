## 项目介绍
利用基本的Encoder-Decoder模型实现seq2seq进行发音预测。


## 代码框架
```
TensorFlow 1.3
Python 2.7(Linux, Mac)
Python 3.5(Windows)
```

## 发音训练数据集
本次采用的训练数据集与验证数据集已经分割好，位于根目录下`data_set`文件夹内。

## 模型介绍

本次项目将主要利用TensorFlow中[tf.contrib.seq2seq](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/seq2seq)中的模块进行模型搭建，采用了Encoder-Decoder模型，实现序列到序列的转化，进行发音预测，输入为单词，输出为模型预测的发音序列，采用的是ARPABET音标。下面将按步骤带领大家实现这一模型，需要大家实现的部分会在代码中用`TODO`标注出来。

### 1. 数据预处理
*此次项目统一使用提供的数据进行训练，验证，根目录demo数据仅仅作为演示*

首先要对发音训练数据集中的数据进行预处理，方便后续处理。

- 位于`Split_Dataset`中的`sp.py`功能是将数据集中的单词变为小写，去除标点符号的读法，将数据集分割为三部分，随机抽取数据，分别用于训练，验证，和测试训练好的模型，其中验证和测试部分数据各用10000个数据，剩下的用于测试。 
- 位于`tensor_seq`中的`converter.py`会将上一步生成的文件中的单词与发音分开并存至对应文件夹`/tensor_seq/dataset`。
- 位于`tensor_seq`中的`data.py`会将上一步产生的位于`/tensor_seq/dataset`文件夹中的数据进一步进行处理。
    - 增加四个特殊字符`<GO>`，`<UNK>`，`<PAD>`，`<EOS>`，分别表示解码开始，位置字符，占位符，和解码停止。
    - 分别将输入序列与输出序列中的符号（字母，读音）与整数一一对应，方便作为输入供神经网络处理。
    - 对读音与单词建立映射，方便后续统计模型预测正确率。
    - 将处理好的词典写入文件，方便后续调取。

完成上述函数后可以按顺序依次调用`sp.py`，`converter.py`，`data.py`，这会在`Split_Dataset`和`tensor_seq/data/`文件夹中生成对应文件，大家可以查看。

### 2. 模型搭建
本次发音预测的模型使用encoder-decoder结构实现，这个步骤中涉及到的代码位于`tensor_seq`中的`model.py`文件。

- 定义encoder层的RNN网络
    - 将输入的整数序列进行embedding，映射成浮点型向量 
- 定义decoder层的RNN网络
    - 对输出序列进行embedding
    - 定义训练时使用的连接方式
    - 定义验证时使用的连接方式
- 将定义好的encoder和decoder连接起来形成一个完整的模型

#### 这部分涉及到的函数、类：
[tf.contrib.rnn.MultiRNNCell](https://github.com/tensorflow/tensorflow/blob/b1ae917558b1ebb439a66374d12e16756a1a231e/tensorflow/python/ops/rnn_cell_impl.py#L1083)　将多个RNN Cell按顺序连接起来，方便构成多层RNN。

[tf.contrib.layers.embed_sequence](https://github.com/tensorflow/tensorflow/blob/b1ae917558b1ebb439a66374d12e16756a1a231e/tensorflow/contrib/layers/python/layers/encoders.py#L91)　输入符号序列，符号数量和embedding的维度，将符号序列映射成为embedding序列。

[tf.contrib.seq2seq.dynamic_decode](https://github.com/tensorflow/tensorflow/blob/b1ae917558b1ebb439a66374d12e16756a1a231e/tensorflow/contrib/seq2seq/python/ops/decoder.py#L150)　根据采用的decoder进行解码。

[tf.contrib.rnn.GRUCell](https://github.com/tensorflow/tensorflow/blob/408fd454d7d2a16269576ea12bcd516e25a6b0c5/tensorflow/python/ops/rnn_cell_impl.py#L262)RNN 中使用的GRU Cell。

[tf.contrib.rnn.LSTMCell](https://github.com/tensorflow/tensorflow/blob/408fd454d7d2a16269576ea12bcd516e25a6b0c5/tensorflow/python/ops/rnn_cell_impl.py#L417)
　RNN 中使用的LSTM Cell。
[tf.contrib.seq2seq.TrainingHelper](https://github.com/tensorflow/tensorflow/blob/408fd454d7d2a16269576ea12bcd516e25a6b0c5/tensorflow/contrib/seq2seq/python/ops/helper.py#L134)　训练时使用的helper类。

[tf.contrib.seq2seq.GreedyEmbeddingHelper](https://github.com/tensorflow/tensorflow/blob/408fd454d7d2a16269576ea12bcd516e25a6b0c5/tensorflow/contrib/seq2seq/python/ops/helper.py#L446)　验证时使用的helper类。

[tf.contrib.seq2seq.BasicDecoder](https://github.com/tensorflow/tensorflow/blob/408fd454d7d2a16269576ea12bcd516e25a6b0c5/tensorflow/contrib/seq2seq/python/ops/basic_decoder.py#L45)　基本解码方式，每一步选择概率最大的一个作为下一时刻的输入。

[tf.contrib.seq2seq.BeamSearchDecoder](https://github.com/tensorflow/tensorflow/blob/408fd454d7d2a16269576ea12bcd516e25a6b0c5/tensorflow/contrib/seq2seq/python/ops/beam_search_decoder.py#L131)　采用beam　search的方式进行结果查找，可以产生多个候选，具体可以参考[知乎介绍](https://www.zhihu.com/question/54356960)。

**常见用法**:
```python
cell = # instance of RNNCell

if mode == "train":
  helper = tf.contrib.seq2seq.TrainingHelper(
    input=input_vectors,
    sequence_length=input_lengths)
elif mode == "infer":
  helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      embedding=embedding,
      start_tokens=tf.tile([GO_SYMBOL], [batch_size]),
      end_token=END_SYMBOL)

decoder = tf.contrib.seq2seq.BasicDecoder(
    cell=cell,
    helper=helper,
    initial_state=cell.zero_state(batch_size, tf.float32))
outputs, _ = tf.contrib.seq2seq.dynamic_decode(
   decoder=decoder,
   output_time_major=False,
   impute_finished=True,
   maximum_iterations=20)
```

### ３. 训练（测试）模型
对于搭建好的模型进行训练（测试），这部分涉及到的代码位于`tensor_seq`文件夹中的`run.py`中。

- 将对应的数据切割成多个mini-batch用于训练
    - 每个mini-batch中的序列要通过补充占位符(`<PAD>`)保增长长度相同
- 定义计算图
    - 用`tf.placeholder`定义模型的输入部分
    - 将输入与输出变量和模型的输入输出对应起来
    - 利用模型的输出计算交叉熵
        * 训练模型
            + 根据计算得出的交叉熵进行梯度计算
            + 选取优化算法
            + 根据选定的优化方法进行反向传播
        * 测试模型
            + 返回模型的输出结果
    
- 定义session运行计算图
    * 训练
        + 根据定义的epoch和mini-batch的大小对模型进行训练
        + 每隔一段时间输出统计信息(正确率，交叉熵...)
        + 保存模型
    * 测试模型
        + 读取保存的模型
        + 将测试数据集作为输入对模型进行评估


#### 这部分涉及到的函数、类：
[tf.summary.scalar](https://github.com/tensorflow/tensorflow/blob/231ca9dd4e258b898cc76a283a90050fd17ee69a/tensorflow/python/summary/summary.py#L76)　用于利用summary记录一个标量

[tf.summary.FileWriter](https://github.com/tensorflow/tensorflow/blob/231ca9dd4e258b898cc76a283a90050fd17ee69a/tensorflow/python/summary/writer/writer.py#L278)　将summary记录下来的值等写入文件

[tf.train.GradientDescentOptimizer](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/training/gradient_descent.py)　SGD优化算法

[tf.train.AdamOptimizer](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/training/adam.py)　Adam算法

[tf.train.RMSPropOptimizer](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/training/rmsprop.py)　RMSProp算法

[tf.clip_by_norm](https://github.com/tensorflow/tensorflow/blob/408fd454d7d2a16269576ea12bcd516e25a6b0c5/tensorflow/python/ops/clip_ops.py#L73)　根据L2范数进行梯度剪裁

[tf.train.Saver](https://github.com/tensorflow/tensorflow/blob/408fd454d7d2a16269576ea12bcd516e25a6b0c5/tensorflow/python/training/saver.py#L948)　用于保存或者恢复模型的变量

### 4. 参数说明
与模型相关的各种参数的定义位于`tensor_seq`中的`params.py`中，可以自己进行调整。
```python
# 学习率
learning_rate = 0.001
# 模型使用的优化算法， 0 对应 SGD, 1 对应 Adam, 2 对应 RMSProp
optimizer_type = 1
# mini-batch的大小
batch_size = 512
# RNN结构 0 对应 LSTM, 1 对应 GRU
Cell_type = 0
# 激活函数的种类， 0 对应 tanh, 1 对应 relu, 2 对应 sigmoid
activation_type = 0
# 每层rnn中神经元的个数
rnn_size = 128
# 层数
num_layers = 2
# embedding的大小
encoding_embedding_size = 64
decoding_embedding_size = 128
# Decoder使用的种类，0　使用basic decoder, 1使用beam search
Decoder_type = 0
#　选择beam search decoder　时的　beam width，影响最终结果的个数 
beam_width = 3
# 最大模型训练次数
epochs = 60
# 1是训练模型，2是测试模型
isTrain = 1
# 每隔多少mini-batch输出一次
display_step = 50
# 保存最近几个模型
max_model_number = 5
```

### 5. 运行方法
将步骤1中生成`Split_Dataset`文件夹中的`training`和`validation`文件用本次项目提供的训练集和测试集替换。并且再次运行`converter.py`和`data.py`。

再运行`tensor_seq`中的`run.py`文件即可。
```shell
python run.py
```


## 参考资料
---
1. [tensorflow 官网介绍](https://www.tensorflow.org/get_started/)
2. [tensorflow　github页面](https://github.com/tensorflow/tensorflow/tree/r1.3)
3. [斯坦福CS 20SI: Tensorflow for Deep Learning Research](https://web.stanford.edu/class/cs20si/)
