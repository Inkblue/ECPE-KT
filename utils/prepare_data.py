# encoding: utf-8


import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time
from bert_serving.client import BertClient
from fnmatch import fnmatch  #通配符匹配

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if not test: np.random.shuffle(index)
    for i in range(int( (length + batch_size -1) / batch_size ) ):
        ret = index[i * batch_size : (i + 1) * batch_size]
        if not test and len(ret) < batch_size : break
        yield ret
##############ECPE-2D里面的load_w2v###############################################
def load_w2v(embedding_dim, embedding_dim_pos, data_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile = open(data_file_path, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        d_len = int(line.strip().split()[1])
        inputFile.readline()
        for i in range(d_len):
            words.extend(inputFile.readline().strip().split(',')[-1].split())
    words = set(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))

    w2v = {}
    inputFile = open(embedding_path, 'r', encoding='utf-8')
    inputFile.readline()
    for line in inputFile.readlines():
        line = line.strip().split()
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

##############ECPE-KA里面的load_w2v###############################################
def load_w2v_KA(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')
    # 把每个词列出索引并排序
    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():  # 读取文件中的每一行
        line = line.strip().split(',')  # 移除每一行中的空格，并以逗号划分
        emotion, clause = line[2], line[-1]
        ##words.extend( [emotion] + clause.split())  # extend在列表末尾一次性追加另一个序列中的多个值
        words.extend(clause.split())  ## extend在列表末尾一次性追加另一个序列中的多个值
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 每个词及词的位置，k表示词的索引(从0开始)，c表示词本身
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))
    # enumerate将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

    # 给每个词创建向量
    w2v = {}
    inputFile2 = open(embedding_path, 'r', encoding='utf-8')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]  # w保存词，ebd保存其向量表示
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0  # 记录在向量表中存在的单词个数
    for item in words:  # 取词
        if item in w2v:  # 取该词的向量表示
            vec = list(map(float, w2v[item]))  # map(float, w2v[item])将向量值转为float型  vec为list列表类型
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])
    ##embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(768)])   ##
    '''
    np.random.normal正态分布
    loc均值为0，以Y轴为对称轴
    scale标准差为0.1，越小曲线越高瘦
    size(int 或者整数元组)输出的值赋在embedding_dim_pos里
    '''

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)  ## embedding_pos维度是201*50

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

def load_bert_embedding(embedding_dim, embedding_dim_pos, train_file_path):
    print('\nload bert embedding...')
    # 把每个词列出索引并排序
    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():  # 读取文件中的每一行
        line = line.strip().split(',')  # 移除每一行中的空格，并以逗号划分
        emotion, clause = line[2], line[-1]
        ##words.extend( [emotion] + clause.split())  # extend在列表末尾一次性追加另一个序列中的多个值
        words.extend(clause.split())  ## extend在列表末尾一次性追加另一个序列中的多个值
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 每个词及词的位置，k表示词的索引(从0开始)，c表示词本身
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))
    # enumerate将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

    # 给每个词创建向量
    w = list(words)
    bc = BertClient()
    ebd = bc.encode(w)  # 获取词的向量表示
    ebd = list(ebd)

    bert_ebd = {}
    bert_ebd = dict(zip(w, ebd))# w保存词，ebd保存其向量表示

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0  # 记录在向量表中存在的单词个数
    for item in words:  # 取词
        if item in bert_ebd:  # 取该词的向量表示
            vec = list(map(float, bert_ebd[item]))  # map(float, w2v[item])将向量值转为float型  vec为list列表类型
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('all_words: {} hit_words: {}'.format(len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])
    '''
    np.random.normal正态分布
    loc均值为0，以Y轴为对称轴
    scale标准差为0.1，越小曲线越高瘦
    size(int 或者整数元组)输出的值赋在embedding_dim_pos里
    '''
    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

def load_data(input_file, word_idx, max_doc_len, max_sen_len):
    print('load data_file: {}'.format(input_file))
    doc_id, y_emotion, y_cause, y_pairs, x, sen_len, doc_len = [[] for i in range(7)]

    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        emo, cause = zip(*pairs)
        y_em, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        for i in range(d_len):
            y_em[i][int(i+1 in emo)]=1
            y_ca[i][int(i+1 in cause)]=1
            words = inputFile.readline().strip().split(',')[-1]
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])
        y_emotion.append(y_em)
        y_cause.append(y_ca)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
    print('n_cut {}'.format(n_cut))
    return doc_id, y_emotion, y_cause, y_pairs, x, sen_len, doc_len

def get_y_pair_CR(doc_len, max_doc_len, y_pairs):
    y_pair = []
    for i in range(len(doc_len)):
        y_tmp = np.zeros((max_doc_len*max_doc_len, 2))
        for j in range(doc_len[i]):
            for k in range(doc_len[i]):
                if (j+1,k+1) in y_pairs[i]:
                    y_tmp[j*max_doc_len+k][1] = 1
                else :
                    y_tmp[j*max_doc_len+k][0] = 1
        y_pair.append(y_tmp)
    return y_pair

def get_y_pair_WC(doc_len, max_doc_len, window_size, y_pairs):
    y_pair, pair_cnt, pair_left_cnt = [], 0, 0
    for i in range(len(doc_len)): #依次读取每个文本
        y_tmp = np.zeros((max_doc_len*(window_size*2+1), 2))
        for j in range(doc_len[i]): #依次读取每个文本中的每句话
            for k in range(-window_size,window_size+1):
                if (j+k) in range(doc_len[i]): #判断当前句子加上window_size是否超过了文本句子界限
                    if (j+1,j+k+1) in y_pairs[i]:
                        y_tmp[j*(window_size*2+1)+k+window_size][1] = 1 #该子句对打标为1
                    else :
                        y_tmp[j*(window_size*2+1)+k+window_size][0] = 1 #该子句对打标为0
        y_pair.append(y_tmp)
        for j, k in y_pairs[i]:
            pair_cnt += 1 #统计情感原因对总个数
            if k-j not in range(-window_size,window_size+1):
                pair_left_cnt += 1 #统计不在window_size范围内的情感原因对个数
    print('pair_cnt {}, pair_left_cnt {}'.format(pair_cnt, pair_left_cnt))
    return y_pair, pair_left_cnt

def load_data_CR(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    doc_id, y_emotion, y_cause, y_pairs, x, sen_len, doc_len = load_data(input_file, word_idx, max_doc_len, max_sen_len)
    y_pair = get_y_pair_CR(doc_len, max_doc_len, y_pairs)
    
    y_emotion, y_cause, y_pair, x, sen_len, doc_len = map(np.array, [y_emotion, y_cause, y_pair, x, sen_len, doc_len])
    for var in ['y_emotion', 'y_cause', 'y_pair', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair, y_pairs, x, sen_len, doc_len

def load_data_WC(input_file, word_idx, max_doc_len = 75, max_sen_len = 45, window_size = 3):
    doc_id, y_emotion, y_cause, y_pairs, x, sen_len, doc_len = load_data(input_file, word_idx, max_doc_len, max_sen_len)
    y_pair, pair_left_cnt = get_y_pair_WC(doc_len, max_doc_len, window_size, y_pairs)

    y_emotion, y_cause, y_pair, x, sen_len, doc_len = map(np.array, [y_emotion, y_cause, y_pair, x, sen_len, doc_len])
    for var in ['y_emotion', 'y_cause', 'y_pair', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair, y_pairs, x, sen_len, doc_len, pair_left_cnt

def cal_prf(pred_y, true_y, doc_len, average='binary'): 
    pred_num, acc_num, true_num = 0, 0, 0
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            if pred_y[i][j]:
                pred_num += 1
            if true_y[i][j]:
                true_num += 1
            if pred_y[i][j] and true_y[i][j]:
                acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def pair_prf_CR(pred_y, true_y, doc_len, threshold = 0.5):
    pred_num, acc_num, true_num = 0, 0, 0
    max_doc_len = int(np.sqrt(pred_y.shape[1]))
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            for k in range(doc_len[i]):
                idx = j*max_doc_len+k
                if pred_y[i][idx][1] > threshold:
                    pred_num += 1
                if true_y[i][idx][1]>0.5:
                    true_num += 1
                if true_y[i][idx][1]>0.5 and pred_y[i][idx][1] > threshold:
                    acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f


def pair_prf_WC(pred_y, true_y, doc_len, pair_left_cnt = 0, threshold = 0.5, window_size = 3):
    pred_num, acc_num, true_num = 0, 0, pair_left_cnt
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]*(window_size*2+1)):
            if max(true_y[i][j]) > 1e-8:
                if pred_y[i][j][1] > threshold:
                    pred_num += 1
                if true_y[i][j][1]>0.5:
                    true_num += 1
                if true_y[i][j][1]>0.5 and pred_y[i][j][1] > threshold:
                    acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def bert_word2id(words, max_sen_len_bert, tokenizer, i, x_tmp, sen_len_tmp):
    # 首先转换成unicode
    tokens_a, ret = tokenizer.tokenize(words), 0
    if len(tokens_a) > max_sen_len_bert - 2:
        ret += 1
        tokens_a = tokens_a[0:(max_sen_len_bert - 2)]
    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    sen_len_tmp[i] = len(input_ids)
    for j in range(len(input_ids)):
        x_tmp[i][j] = input_ids[j]
    return ret

def load_data_bert(input_file, tokenizer, word_idx, max_doc_len, max_sen_len_bert, max_sen_len):
    print('load data_file: {}'.format(input_file))
    doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len = [[] for i in range(9)]
    
    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        emo, cause = zip(*pairs)
        y_emotion_tmp, y_cause_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2))
        x_bert_tmp, sen_len_bert_tmp = np.zeros((max_doc_len, max_sen_len_bert),dtype=np.int32), np.zeros(max_doc_len,dtype=np.int32)
        x_tmp, sen_len_tmp = np.zeros((max_doc_len, max_sen_len),dtype=np.int32), np.zeros(max_doc_len,dtype=np.int32)
        for i in range(d_len):
            y_emotion_tmp[i][int(i+1 in emo)]=1
            y_cause_tmp[i][int(i+1 in cause)]=1
            words = inputFile.readline().strip().split(',')[-1]
            n_cut += bert_word2id(words, max_sen_len_bert, tokenizer, i, x_bert_tmp, sen_len_bert_tmp)
            # pdb.set_trace()
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    break
                x_tmp[i][j] = int(word_idx[word])
        
        y_emotion.append(y_emotion_tmp)
        y_cause.append(y_cause_tmp)
        x_bert.append(x_bert_tmp)
        sen_len_bert.append(sen_len_bert_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
    print('n_cut {}'.format(n_cut))
    return doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len

def load_data_CR_Bert(input_file, tokenizer, word_idx, max_doc_len = 75, max_sen_len_bert = 60, max_sen_len = 30):
    doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len = load_data_bert(input_file, tokenizer, word_idx, max_doc_len, max_sen_len_bert, max_sen_len)
    y_pair = get_y_pair_CR(doc_len, max_doc_len, y_pairs)
    
    y_emotion, y_cause, y_pair, x_bert, sen_len_bert, x, sen_len, doc_len = map(np.array, [y_emotion, y_cause, y_pair, x_bert, sen_len_bert, x, sen_len, doc_len])
    for var in ['y_emotion', 'y_cause', 'y_pair', 'x_bert', 'sen_len_bert', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len

def load_data_WC_Bert(input_file, tokenizer, word_idx, max_doc_len = 75, max_sen_len_bert = 60, max_sen_len = 30, window_size = 3):
    doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len = load_data_bert(input_file, tokenizer, word_idx, max_doc_len, max_sen_len_bert, max_sen_len)
    y_pair, pair_left_cnt = get_y_pair_WC(doc_len, max_doc_len, window_size, y_pairs)
    
    y_emotion, y_cause, y_pair, x_bert, sen_len_bert, x, sen_len, doc_len = map(np.array, [y_emotion, y_cause, y_pair, x_bert, sen_len_bert, x, sen_len, doc_len])
    for var in ['y_emotion', 'y_cause', 'y_pair', 'x_bert', 'sen_len_bert', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('load data done!\n')
    return doc_id, y_emotion, y_cause, y_pair, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len, pair_left_cnt

def load_LIWC(LIWC_filename, train_file_path, embedding_dim_LIWC):
    inputFile = open(LIWC_filename, 'r', encoding='utf-8')
    inputFile.readline()  # 读取首行的%
    lines = inputFile.readlines()  #读取文件所有行
    type2num = dict()
    word2type = dict()  # 存储词典内一般的词
    word2type_wlid = dict()  # 存储词典内带*的词

    # 读取LIWC词典的种类
    for i, line in enumerate(lines):
        if '%' in line:
            loc_num = i
            break
        tmp = line.strip().split()
        type2num[int(tmp[0])] = i  #{种类编号：embedding索引}，如{1：0}

    # 将LIWC词典中所有词所属类别存为字典
    for line in lines[loc_num + 1:]:
        tmp = line.strip().split()
        #print(type(tmp[0]))  # 输出<class 'str'>
        if '*' in tmp[0]:
            word2type_wlid[tmp[0]] = list(map(int, tmp[1:]))
        else:
            word2type[tmp[0]] = list(map(int, tmp[1:]))

    # 读取情感原因数据集中所有的单词，去重
    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():  # 读取文件中的每一行
        line = line.strip().split(',')  # 移除每一行中的空格，并以逗号划分
        emotion, clause = line[2], line[-1]
        words.extend(clause.split())  ## extend在列表末尾一次性追加另一个序列中的多个值
    words = set(words)  # 所有不重复词的集合

    # 对数据集中每个词进行LIWC的embedding
    LIWC_embedding = [list(np.zeros(embedding_dim_LIWC))]
    hit = 0  # 记录在向量表中存在的单词个数
    keys = list(word2type_wlid.keys())  # 存储*号词典的所有词
    for item in words:  # 取词
        vec = list(np.zeros(embedding_dim_LIWC))  # 初始化

        if item in word2type:  # 取非*号词的种类
            kinds = list(map(int, word2type[item]))  # map(float, word2type[item])将向量值转为int型  kinds为list列表类型
            for i, value in enumerate(kinds):
                vec[type2num.get(value)] = 1
            hit += 1
        else:  #*号词或者不存在的词
            for key in keys:
                if fnmatch(item,key):  # 判断是否为*号词
                    kinds = list(map(int, word2type_wlid[key]))  # map(float, word2type_wild[key])将向量值转为int型  kinds为list列表类型
                    for i, value in enumerate(kinds):
                        vec[type2num.get(value)] = 1
                    hit += 1
        LIWC_embedding.append(vec)
    LIWC_embedding = np.array(LIWC_embedding)
    print('all_words: {} LIWC_hit_words: {}'.format(len(words), hit))

    return LIWC_embedding


def load_data_weibo(input_file, word_idx, max_doc_len=75, max_sen_len=45):
    print('load data_file: {}'.format(input_file))
    y_position, y_cause, y_pairs, x, sen_len, doc_len = [], [], [], [], [], []
    doc_id = []

    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])  # 文本的索引
        d_len = int(line[1])  # 该文本句子个数
        pairs = eval('[' + inputFile.readline().strip() + ']')  # 返回  情感-原因对，句子的序号 [(1, 4), (2, 5), (3, 6)]
        doc_len.append(d_len)
        y_pairs.append(pairs)
        pos, cause = zip(*pairs)  # 解压pairs，将情感句序号存储在pos，原因句序号存储在cause中[(1, 2, 3), (4, 5, 6)]
        y_po, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros(max_doc_len, dtype=np.int32), np.zeros((max_doc_len, max_sen_len), dtype=np.int32)
        for i in range(d_len):
            y_po[i][
                int(i + 1 in pos)] = 1  # 该文本中各句子是否为情感句的标签。标注当前句子是否为情感句。[int(i+1 in pos)]  i+1=pos，则表示该句为情感句，返回1，否则返回0
            y_ca[i][int(i + 1 in cause)] = 1  # 该文本中各句子是否为原因句的标签。
            words = inputFile.readline().strip().split(',')[-1]  # 存储该句的内容
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)  # 存储当前句子的单词数
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1  # 当前句超出最大句子长度的单词个数
                    break
                x_tmp[i][j] = int(word_idx[word])  # 将当前句的每个词存储在x_tmp中

        y_position.append(y_po)
        y_cause.append(y_ca)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)

    y_position, y_cause, x, sen_len, doc_len = map(np.array, [y_position, y_cause, x, sen_len, doc_len])  # 将所有变量变成数组
    for var in ['y_position', 'y_cause', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return doc_id, y_position, y_cause, y_pairs, x, sen_len, doc_len
