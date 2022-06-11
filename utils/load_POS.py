from fnmatch import fnmatch
import numpy as np

# 读取词语与词性对应的POS_result文件，只保留需要的词性类别，其余归为“其他类”，并加载数据集中词语的POS embedding
def load_POS(POS_result, train_file_path):
    inputFile = open(POS_result, 'r', encoding='utf-8')
    inputFile.readline()  # 读取首行
    lines = inputFile.readlines()  # 读取文件所有行
    word2POS = dict()
    POS2num = {}

    # 构建词性词典，汇总后的词性共11类
    POS_dic = {'nr*': 'nr', 'ns*': 'ns', 'n[!rs]*': 'n', 'n': 'n', 'a*': 'a', 'd*': 'd', 'v*': 'v', 'rr': 'rr','r[!r]*': 'r', 'r': 'r'}
    POS_keys = list(POS_dic.keys())

    for i, line in enumerate(lines):
        tmp = line.strip().split()
        # 将每个词性归类
        initPOS = 'other'
        for key in POS_keys:
            if fnmatch(tmp[1], key):  # 判断该词性属于哪一类别
                initPOS = POS_dic[key]
        # 将相同词语的词性进行合并，将词与词性对应
        if tmp[0] in word2POS:  # 存在该词的字典，则添加它的词性
            POS_list = []
            POS_list.extend(word2POS[tmp[0]])  # 初始化已有的词性
            POS_list.append(initPOS)  # 添加该词的新词性
            word2POS[tmp[0]] = POS_list  # 重新写入字典
        else:  # 不存在该词
            POS_list = []
            POS_list.append(initPOS)  # 添加该词词性
            word2POS[tmp[0]] = POS_list
    #print(word2POS)

    # 将词性与维度对应
    POS = list(POS_dic.values())
    POS.append('other')
    POS = set(POS)  # 词性的种类
    #print(len(POS))  # 词性的维度大小
    #print('POS', POS)  # 输出POS {'r', 'v', 'nr', 'd', 'rr', 'n', 'a', 'other', 'ns'}
    for i, item in enumerate(POS):
        POS2num[item] = i
    #print(POS2num)  # 输出{'r': 0, 'v': 1, 'nr': 2, 'd': 3, 'rr': 4, 'n': 5, 'a': 6, 'other': 7, 'ns': 8}

    # 读取情感原因数据集中所有的单词，去重
    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():  # 读取文件中的每一行
        line = line.strip().split(',')  # 移除每一行中的空格，并以逗号划分
        emotion, clause = line[2], line[-1]
        words.extend(clause.split())  # extend在列表末尾一次性追加另一个序列中的多个值
    words = set(words)  # 所有不重复词的集合

    # 对数据集中每个词进行词性的embedding
    embedding_dim_wordPOS = len(POS)
    wordPOS_embedding = [list(np.zeros(embedding_dim_wordPOS))]
    # hit = 0  # 记录在向量表中存在的单词个数
    for item in words:  # 取词
        vec = list(np.zeros(embedding_dim_wordPOS))  # 初始化
        kinds = word2POS[item]
        for i, kind in enumerate(kinds):
            vec[POS2num.get(kind)] = 1
        # hit += 1
        wordPOS_embedding.append(vec)
    wordPOS_embedding = np.array(wordPOS_embedding)
    print('load part-of-speech done')
    return wordPOS_embedding