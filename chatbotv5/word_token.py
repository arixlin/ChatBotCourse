# editor: linming
# mail: linmingzxx@gmail.com

# coding:utf-8
import sys
import jieba
import numpy as np
import re

zhPattern = re.compile(u'.*[\u4e00-\u9fa5].*')


def re_content(content):
    """
    judge content
    """
    if zhPattern.search(content):
        return True
    else:
        return False


class WordToken(object):
    def __init__(self):
        # 最小起始id号, 保留的用于表示特殊标记
        self.START_ID = 4
        self.word2id_dict = {}
        self.id2word_dict = {}

    def load_file_list(self, file_list):
        """
        加载样本文件列表，全部切词后统计词频，按词频由高到低排序后顺次编号
        并存到self.word2id_dict和self.id2word_dict中
        """
        words_count = {}
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as file_object:
                text = file_object.read()
                seg_list = set(text)
                for str in seg_list:
                    if str in words_count:
                        words_count[str] = words_count[str] + 1
                    else:
                        words_count[str] = 1

        sorted_list = [[v[1], v[0]] for v in words_count.items()]
        for index, item in enumerate(sorted_list):
            word = item[1]
            self.word2id_dict[word] = self.START_ID + index
            self.id2word_dict[self.START_ID + index] = word
        return index


    def load_dict(self):
        """

        :param file_list:
        :return:
        """
        with open('./conf/word2id_dict.txt', 'r', encoding='utf-8') as f:
            word2id = f.read()
            self.word2id_dict = eval(word2id)
        with open('./conf/id2word_dict.txt', 'r', encoding='utf-8') as f:
            id2word = f.read()
            self.id2word_dict = eval(id2word)


    def word2id(self, word):
        # if not isinstance(word, unicode):
        if not isinstance(word, str):
            print ("Exception: error word not unicode")
            sys.exit(1)
        # print (self.word2id_dict)
        if word in self.word2id_dict:
            return self.word2id_dict[word]
        else:
            return None


    def id2word(self, id):
        id = int(id)
        if id in self.id2word_dict:
            return self.id2word_dict[id]
        else:
            return '。'

