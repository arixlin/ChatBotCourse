# coding:utf-8
# author: lichuang
# mail: shareditor.com@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import word_token
import jieba
import random

# 输入序列长度
input_seq_len = 10
# 输出序列长度
output_seq_len = 10
# 空值填充0
PAD_ID = 0
# 输出序列起始标记
GO_ID = 1
# 结尾标记
EOS_ID = 2
# LSTM神经元size
size = 8
# 最大输入符号数
num_encoder_symbols = 32
# 最大输出符号数
num_decoder_symbols = 32
# 初始学习率
init_learning_rate = 1

wordToken = word_token.WordToken()

# 放在全局的位置，为了动态算出num_encoder_symbols和num_decoder_symbols
# max_token_id = wordToken.load_file_list(['./samples/question', './samples/answer'])
global max_token_id
max_token_id = 0
# max_token_id = 10000
samplepath = './samples/chat'

#to one word
def get_id_list_from(sentence):
    sentence_id_list = []
    for str in sentence:
        id = wordToken.word2id(str)
        if id:
            sentence_id_list.append(wordToken.word2id(str))
    return sentence_id_list


def get_train_set():
    """

    :return:
    """
    global num_encoder_symbols, num_decoder_symbols
    train_set = []
    with open(samplepath, 'r', encoding='utf-8') as talk_file:
        while True:
            talk = talk_file.readline().split('|')
            if len(talk) >= 2:
                question = talk[0].strip()
                answer = talk[1].strip()
            else:
                break
            if question and answer:
                question = question.strip()
                answer = answer.strip()

                question_id_list = get_id_list_from(question)
                answer_id_list = get_id_list_from(answer)
                answer_id_list.append(EOS_ID)
                train_set.append([question_id_list, answer_id_list])
            else:
                break
    return train_set



def get_samples(train_set, batch_num):
    """构造样本数据
    :return:
        encoder_inputs: [array([0, 0], dtype=int32), array([0, 0], dtype=int32), array([5, 5], dtype=int32),
                        array([7, 7], dtype=int32), array([9, 9], dtype=int32)]
        decoder_inputs: [array([1, 1], dtype=int32), array([11, 11], dtype=int32), array([13, 13], dtype=int32),
                        array([15, 15], dtype=int32), array([2, 2], dtype=int32)]
    """
    # train_set = [[[5, 7, 9], [11, 13, 15, EOS_ID]], [[7, 9, 11], [13, 15, 17, EOS_ID]], [[15, 17, 19], [21, 23, 25, EOS_ID]]]
    for n in range(0, len(train_set), batch_num):
        raw_encoder_input = []
        raw_decoder_input = []
        if batch_num >= len(train_set):
            batch_train_set = train_set
        else:
            # random_start = random.randint(0, len(train_set)-batch_num)
            batch_train_set = train_set[n:n+batch_num]

        for sample in batch_train_set:
            raw_encoder_input.append([PAD_ID] * (input_seq_len - len(sample[0])) + sample[0])
            raw_decoder_input.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))

        encoder_inputs = []
        decoder_inputs = []
        target_weights = []

        for length_idx in range(input_seq_len):
            encoder_inputs.append(np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input], dtype=np.int32))
        for length_idx in range(output_seq_len):
            decoder_inputs.append(np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
            target_weights.append(np.array([
                0.0 if length_idx == output_seq_len - 1 or decoder_input[length_idx] == PAD_ID else 1.0 for decoder_input in raw_decoder_input
            ], dtype=np.float32))
        yield encoder_inputs, decoder_inputs, target_weights, n


def seq_to_encoder(input_seq):
    """从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
    """

    input_seq_array = [int(v) for v in input_seq.split()]
    encoder_input = [PAD_ID] * (input_seq_len - len(input_seq_array)) + input_seq_array
    decoder_input = [GO_ID] + [PAD_ID] * (output_seq_len - 1)
    encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
    decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
    target_weights = [np.array([1.0], dtype=np.float32)] * output_seq_len
    return encoder_inputs, decoder_inputs, target_weights

class Model():
    def __init__(self, feed_previous=False):
        self.feed_previous = feed_previous
        self._build_model()

    def _build_model(self):
        """构造模型
        """
        learning_rate = 0.01
        # learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
        # learning_rate_decay_op = learning_rate.assign(learning_rate * 0.999)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(input_seq_len):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in range(output_seq_len + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
        for i in range(output_seq_len):
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))


        # decoder_inputs左移一个时序作为targets
        targets = [self.decoder_inputs[i + 1] for i in range(output_seq_len)]

        cell = tf.contrib.rnn.BasicLSTMCell(size)

        # 这里输出的状态我们不需要
        self.outputs, _ = seq2seq.embedding_attention_seq2seq(
                            self.encoder_inputs,
                            self.decoder_inputs[:output_seq_len],
                            cell,
                            num_encoder_symbols=num_encoder_symbols,
                            num_decoder_symbols=num_decoder_symbols,
                            embedding_size=size,
                            output_projection=None,
                            feed_previous=self.feed_previous,
                            dtype=tf.float32)

        # 计算加权交叉熵损失
        self.loss = seq2seq.sequence_loss(self.outputs, targets, self.target_weights)
        # 梯度下降优化器
        # opt = tf.train.GradientDescentOptimizer(learning_rate)
        # RMSprop适合处理非平稳目标，更适合RNN
        opt = tf.train.RMSPropOptimizer(learning_rate)
        # 优化目标：让loss最小化
        self.update = opt.apply_gradients(opt.compute_gradients(self.loss))


def train():
    """
    训练过程
    """
    # train_set = [[[5, 7, 9], [11, 13, 15, EOS_ID]], [[7, 9, 11], [13, 15, 17, EOS_ID]],
    #              [[15, 17, 19], [21, 23, 25, EOS_ID]]]
    train_set = get_train_set()
    model = Model()
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(100):
            for sample_encoder_inputs, sample_decoder_inputs, sample_target_weights, n in get_samples(train_set, 100):
                # print (n)
                input_feed = {}
                for l in range(input_seq_len):
                    input_feed[model.encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(output_seq_len):
                    input_feed[model.decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[model.target_weights[l].name] = sample_target_weights[l]
                input_feed[model.decoder_inputs[output_seq_len].name] = np.zeros([len(sample_decoder_inputs[0])], dtype=np.int32)

                [loss_ret, _] = sess.run([model.loss, model.update], input_feed)
            print('step=', step, 'loss=', loss_ret)
            saver.save(sess, './model/demo')


def predict():
    """
    预测过程
    """
    model = Model(feed_previous=True)
    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        saver.restore(sess, './model/demo')
        sys.stdout.write("> ")
        sys.stdout.flush()
        input_seq = sys.stdin.readline()
        while input_seq:
            input_seq = input_seq.strip()
            input_id_list = get_id_list_from(input_seq)
            if (len(input_id_list)):
                sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = seq_to_encoder(' '.join([str(v) for v in input_id_list]))

                input_feed = {}
                for l in range(input_seq_len):
                    input_feed[model.encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(output_seq_len):
                    input_feed[model.decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[model.target_weights[l].name] = sample_target_weights[l]
                input_feed[model.decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

                # 预测输出
                outputs_seq = sess.run(model.outputs, input_feed)
                # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
                outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
                # 如果是结尾符，那么后面的语句就不输出了
                if EOS_ID in outputs_seq:
                    outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
                outputs_seq = [wordToken.id2word(v) for v in outputs_seq]
                print("".join(outputs_seq))
            else:
                print("WARN：词汇不在服务区")

            sys.stdout.write("> ")
            sys.stdout.flush()
            input_seq = sys.stdin.readline()


if __name__ == "__main__":
    # if sys.argv[1] == 'train':
    #     train()
    # else:
    # predict()

    # max_token_id = wordToken.load_file_list(['./samples/answer', './samples/question'])
    max_token_id = wordToken.load_file_list([samplepath])
    num_encoder_symbols = max_token_id + 5
    num_decoder_symbols = max_token_id + 5

    #eval()
    # wordToken.load_dict()
    # predict()

    #train()
    # with open('./conf/word2id_dict.txt', 'w', encoding='utf-8') as f:
    #     f.write(str(wordToken.word2id_dict))
    # with open('./conf/id2word_dict.txt', 'w', encoding='utf-8') as f:
    #     f.write(str(wordToken.id2word_dict))
    train()