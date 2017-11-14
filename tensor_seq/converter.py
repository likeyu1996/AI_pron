# -*- coding: utf-8 -*-
'''
提取单词
'''
import os
import re

# TODO
# 完成以下函数
def extract_words(dict_path, source_path, target_path, file_name):
    """读取数据，并将数据分为单词与读音，存至对应的目录内。

       Args:
           dict_path: 读取数据路径
           source_path: 单词存放路径
           target_path: 发音序列存放路径
           file_name: 文件名前缀

       其中单词文件为每一行为一个单词：
       cheered
       benshoof
       achieve
       
       发音文件为每行一个单词的发音：
       CH IH1 R D
       B EH1 N SH UH0 F
       AH0 CH IY1 V

    """
    file_r = open(dict_path+file_name, 'r')
    t_source_list = []
    t_target_list = []
    for line in file_r.readlines():
        t = line.split()[0].lower()
        if re.match('^[a-z]+[\'\.]?[a-z.]+$', t):
            t_source_list.append(t)
            t_target_list.append(' '.join(line.split()[1:]))
        elif re.match('^([a-z]+[\'\.]?[a-z.]+)\(2\)$', t):
            tt = re.match('^([a-z]+[\'\.]?[a-z.]+)(\(2\))$', t).group(1)
            t_source_list.append(tt)
            # print line.split()
            t_target_list.append(' '.join(line.split()[1:]))
    n_0 = len(t_source_list)
    file_s = open(source_path+file_name, 'w')
    file_t = open(target_path+file_name, 'w')
    for i in range(n_0):
        if i < n_0-1:
            file_s.write(t_source_list[i]+'\n')
            file_t.write(t_target_list[i]+'\n')
        else:
            file_s.write(t_source_list[i])
            file_t.write(t_target_list[i])
    file_r.close()
    file_s.close()
    file_t.close()
data_set_path = './dataset/'
if not os.path.exists(data_set_path):
    os.makedirs(data_set_path)

dict_path_pre = '../Split_Dataset/'
source_path_pre = data_set_path + 'source_list_'
target_path_pre = data_set_path + 'target_list_'
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'training')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'testing')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'validation')
extract_words(dict_path_pre, source_path_pre, target_path_pre, 'whole')
