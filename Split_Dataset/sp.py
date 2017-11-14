# -*- coding: utf-8 -*-
import re
import numpy as np

f = open('../demo', 'r')

t_source_list = []
t_target_list = []

# separete the words with the pronunciations, and
# change all the characters in words to lowercase
for line in f.readlines():
    t = line.split()[0].lower()
    if re.match('^[a-z]+[\'\.]?[a-z.]+$', t):
        t_source_list.append(t)
        t_target_list.append(' '.join(line.split()[1:]))
    elif re.match('^([a-z]+[\'\.]?[a-z.]+)\(2\)$', t):
        tt = re.match('^([a-z]+[\'\.]?[a-z.]+)(\(2\))$', t).group(1)
        t_source_list.append(tt)
        # print line.split()
        t_target_list.append(' '.join(line.split()[1:]))

training = open('./training', 'w')
testing = open('./testing', 'w')
validation = open('./validation', 'w')
whole = open('./whole', 'w')

# TODO
# 经过上面代码的处理以后，t_source_list与t_target_list中分别单词与单词
# 对应的音标序列，你需要将全部数据写入whole中，然后将数据打乱，抽取10000个
# 写入testing，抽取10000个写入validation，剩下的写入training中。
# 单词，音标之间用空格分离，每个单词一行。
# 如： enabled EH0 N EY1 B AH0 L D

# 设置参数n_0来表示序列长度，逐行向whole写入单词和音标，用if控制最后一行不输入换行符
n_0 = len(t_source_list)
for i in range(n_0):
    if i < n_0-1:
        whole.write(t_source_list[i]+' '+t_target_list[i]+'\n')
    else:
        whole.write(t_source_list[i]+' '+t_target_list[i])
# 设置表示testing和validation两个文件内容长度的参数，以他们的和为size从n_0中选取不重复随机数
# 并将其分为0~9999/10000~19999两个部分，逐行写入单词和音标，用if控制两个文件最后一行不输入换行符
i_testing = 10000
i_validation = 10000
i_not_training = i_testing+i_validation
rand_0 = np.random.choice(n_0, i_not_training, replace=False)
k_index = 0
for k in rand_0:
    if k_index < i_testing-1:
        testing.write(t_source_list[k]+' '+t_target_list[k]+'\n')
    elif k_index == i_testing-1:
        testing.write(t_source_list[k]+' '+t_target_list[k])
    elif k_index == i_not_training-1:
        validation.write(t_source_list[k]+' '+t_target_list[k])
    else:
        validation.write(t_source_list[k]+' '+t_target_list[k]+'\n')
    k_index += 1
# 生成由剩余行的序号组成的列表rand_1，并用和whole相同的方法写入training文件
rand_1 = list(set(range(n_0)) ^ set(rand_0))
t_index=0
for training_num in rand_1:
    if t_index < len(rand_1)-1:
        training.write(t_source_list[training_num]+' '+t_target_list[training_num]+'\n')
    else:
        training.write(t_source_list[training_num]+' '+t_target_list[training_num])
    t_index += 1
# print(k_index, '\n', len(rand_0), len(rand_1))
validation.close()
testing.close()
training.close()
whole.close()
