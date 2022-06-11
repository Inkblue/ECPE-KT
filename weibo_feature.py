# encoding: utf-8


import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb
import pickle

sys.path.append('./utils')
##from tf_funcs import *
##from prepare_data import *
from utils.tf_funcs import *  ##
from utils.prepare_data import *  ##
from utils.load_POS import *  ##

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_string('SC_LIWC_file', 'data/sc_liwc.dic', 'Simplified Chinese LIWC file')  ##
tf.app.flags.DEFINE_string('POS_result', 'data/Word_seg_all.txt', 'Word segmentation and POS results')  ##
##tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
tf.app.flags.DEFINE_integer('embedding_dim_LIWC', 71, 'dimension of LIWC embedding')  ##
##=======bert端口使用的参数
tf.app.flags.DEFINE_integer('embedding_dim_word', 768, 'dimension of word embedding')  ##
tf.app.flags.DEFINE_integer('embedding_dim_all', 848, 'dimension of word embedding')  ##768+71+9(bert词向量+LIWC+词性)
##=======bert端口使用的参数

##=======w2v使用的参数
#tf.app.flags.DEFINE_integer('embedding_dim_word', 200, 'dimension of word embedding')  ## w2v的词向量
#tf.app.flags.DEFINE_integer('embedding_dim_all', 280, 'dimension of word embedding')  ##200+71+9(w2v词向量+LIWC+词性)
##=======w2v使用的参数
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of sentences per document')
## model struct ##
tf.app.flags.DEFINE_string('model_type', 'Inter-EC', 'model type: Indep, Inter-CE, Inter-EC')
tf.app.flags.DEFINE_string('trans_type', 'cross_road', 'transformer type: cross_road, window_constrained')
tf.app.flags.DEFINE_integer('window_size', 3, 'window_size')
tf.app.flags.DEFINE_integer('trans_iter', 2, 'number of cross-road 2D transformer layers')
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
tf.app.flags.DEFINE_string('scope', 'TEMP', 'scope')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.002, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.7, 'keep prob for word embedding')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'keep prob for softmax layer')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('emo', 1., 'loss weight of emotion ext.')
tf.app.flags.DEFINE_float('cause', 1., 'loss weight of cause ext.')
tf.app.flags.DEFINE_float('pair', 1., 'loss weight of pair ext.')
tf.app.flags.DEFINE_float('threshold', 0.5, 'threshold for pair ext.')
tf.app.flags.DEFINE_integer('feature_num', 30, 'feature vector length of pairs')
tf.app.flags.DEFINE_integer('training_iter', 10, 'number of training iter')


def build_subtasks(x, sen_len, doc_len, is_training):
    def get_s(inputs, sen_len, name):
        with tf.name_scope('word_encode'):
            inputs = biLSTM(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'word_layer' + name)
        # inputs shape:        [-1, FLAGS.max_sen_len, 2 * FLAGS.n_hidden]
        with tf.name_scope('word_attention'):
            sh2 = 2 * FLAGS.n_hidden
            w1 = get_weight_varible('word_att_w1' + name, [sh2, sh2])
            b1 = get_weight_varible('word_att_b1' + name, [sh2])
            w2 = get_weight_varible('word_att_w2' + name, [sh2, 1])
            s = att_var(inputs, sen_len, w1, b1, w2)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
        return s

    def emo_cause_prediction(s_ec, is_training, name):
        s1 = tf.nn.dropout(s_ec, keep_prob=is_training * FLAGS.keep_prob2 + (1. - is_training))
        s1 = tf.reshape(s1, [-1, 2 * FLAGS.n_hidden])
        w_ec = get_weight_varible('softmax_w_' + name, [2 * FLAGS.n_hidden, FLAGS.n_class])
        b_ec = get_weight_varible('softmax_b_' + name, [FLAGS.n_class])
        pred_ec = tf.nn.softmax(tf.matmul(s1, w_ec) + b_ec)
        pred_ec = tf.reshape(pred_ec, [-1, FLAGS.max_doc_len, FLAGS.n_class])
        return pred_ec, w_ec, b_ec

    with tf.name_scope('emotion_prediction'):
        s1 = get_s(x, sen_len, name='word_encode_emo')
        s_emo = biLSTM(s1, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_encode_emo')
        pred_emo, w_emo, b_emo = emo_cause_prediction(s_emo, is_training, name='emotion')

    # with tf.name_scope('cause_prediction'):
    #     s1 = get_s(x, sen_len, name='word_encode_cause')
    #     feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])
    #     if FLAGS.model_type in ['Inter-CE', 'Inter-EC']:
    #         s1 = tf.concat([s1, pred_emo], 2) * feature_mask
    #     s_cause = biLSTM(s1, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'sentence_encode_cause')
    #     pred_cause, w_cause, b_cause = emo_cause_prediction(s_cause, is_training, name='cause')

    # reg = tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)
    # reg += tf.nn.l2_loss(w_emo) + tf.nn.l2_loss(b_emo)
    reg = tf.nn.l2_loss(w_emo) + tf.nn.l2_loss(b_emo)
    # if FLAGS.model_type in ['Inter-CE']:
    #     return pred_cause, pred_emo, s_cause, s_emo, reg
    return pred_emo, s_emo, reg


def build_model(wordPOS_embedding, LIWC_embedding, word_embedding, pos_embedding, x, sen_len, doc_len, is_training):##
    embedding_tmp = tf.concat([word_embedding, LIWC_embedding], 1)  ##
    embedding_all = tf.concat([embedding_tmp, wordPOS_embedding], 1)  ##
    x = tf.nn.embedding_lookup(embedding_all, x)  ##
    x = tf.reshape(x, [-1, FLAGS.max_sen_len,FLAGS.embedding_dim_all])  ## 将x变成行数不确定，但列数为max_sen_len，深度为embedding_dim_all的张量
    x = tf.nn.dropout(x, keep_prob=is_training * FLAGS.keep_prob1 + (1. - is_training))
    sen_len = tf.reshape(sen_len, [-1])
    # x shape:        [-1, FLAGS.max_sen_len, FLAGS.embedding_dim]

    ########################################## emotion & cause extraction  ############
    print('building subtasks')
    pred_emo, s_emo, reg = build_subtasks(x, sen_len, doc_len, is_training)
    print('build subtasks Done!')
    # feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])
    # pred_emo_feature = tf.stop_gradient(pred_emo * feature_mask + 1e-8)
    # pred_cause_feature = tf.stop_gradient(pred_cause * feature_mask + 1e-8)

    return pred_emo, s_emo, reg


def print_info():
    # print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    # print('model_type {} \ntrans_type {} \ntrans_iter {} \nwindow_size {}'.format(
    #     FLAGS.model_type, FLAGS.trans_type, FLAGS.trans_iter, FLAGS.window_size))

    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print('batch {} \nlr {} \nkb1 {} \nkb2 {} \nl2_reg {}'.format(
        FLAGS.batch_size, FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    # print('FLAGS.emo {} \nFLAGS.cause {} \nFLAGS.pair {} \nthreshold {} \ntraining_iter {}\n\n'.format(
    #     FLAGS.emo, FLAGS.cause, FLAGS.pair, FLAGS.threshold, FLAGS.training_iter))


def get_batch_data(x, sen_len, doc_len, is_training, y_emotion, y_cause, batch_size, test=False):
    for index in batch_index(len(y_cause), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], is_training, y_emotion[index], y_cause[index]]
        yield feed_list, len(index)


def run():
    if FLAGS.log_file_name:
        if not os.path.exists('log'):
            os.makedirs('log')
        sys.stdout = open(FLAGS.log_file_name, 'w')
    print_time()
    tf.reset_default_graph()
    # Model Code Block
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_bert_embedding(FLAGS.embedding_dim_word, FLAGS.embedding_dim_pos, 'data/clause_keywords_all.csv') ## bert编码
    ##word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v_KA(FLAGS.embedding_dim_word,FLAGS.embedding_dim_pos, 'data/clause_keywords_all.csv', FLAGS.w2v_file) ## w2v编码
    ##word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, 'data/all_data_pair.txt', FLAGS.w2v_file)
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    LIWC_embedding = load_LIWC(FLAGS.SC_LIWC_file, 'data/clause_keywords_all.csv', FLAGS.embedding_dim_LIWC)  ##
    LIWC_embedding = tf.constant(LIWC_embedding, dtype=tf.float32, name='LIWC_embedding')  ##

    wordPOS_embedding = load_POS(FLAGS.POS_result, 'data/clause_keywords_all.csv')  ##
    wordPOS_embedding = tf.constant(wordPOS_embedding, dtype=tf.float32, name='wordPOS_embedding')  ##

    print('build model...')
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.float32)  # for Bert
    y_emotion = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_cause = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    s_emo = tf.placeholder(tf.float32,[None, FLAGS.max_doc_len, FLAGS.max_sen_len] )
    # if FLAGS.trans_type == 'cross_road':
    #     y_pair = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len * FLAGS.max_doc_len, FLAGS.n_class])
    # else:
    #     y_pair = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len * (FLAGS.window_size * 2 + 1), FLAGS.n_class])
    placeholders = [x, sen_len, doc_len, is_training, y_emotion, y_cause, s_emo]

    pred_emo, s_emo, reg = build_model(wordPOS_embedding, LIWC_embedding, word_embedding, pos_embedding, x, sen_len, doc_len, is_training) ##
    ##pred_emo, pred_cause, pred_pair, reg = build_model(word_embedding, pos_embedding, x, sen_len, doc_len, is_training)
    print('build model done!\n')

    loss_emo = - tf.reduce_sum(y_emotion * tf.log(pred_emo)) / tf.cast(tf.reduce_sum(y_emotion), dtype=tf.float32)
    # loss_cause = - tf.reduce_sum(y_cause * tf.log(pred_cause)) / tf.cast(tf.reduce_sum(y_cause), dtype=tf.float32)
    # loss_pair = - tf.reduce_sum(y_pair * tf.log(pred_pair)) / tf.cast(tf.reduce_sum(y_pair), dtype=tf.float32)
    # loss_op = loss_cause * FLAGS.cause + loss_emo * FLAGS.emo + loss_pair * FLAGS.pair + reg * FLAGS.l2_reg
    loss_op = loss_emo * FLAGS.emo + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)

    true_y_emo_op = tf.argmax(y_emotion, 2)
    pred_y_emo_op = tf.argmax(pred_emo, 2)
    # true_y_cause_op = tf.argmax(y_cause, 2)
    # pred_y_cause_op = tf.argmax(pred_cause, 2)
    # true_y_pair_op = y_pair
    # pred_y_pair_op = pred_pair

    # Training Code Block
    print_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        emo_list, cause_list, pair_list = [], [], []

        for fold in range(1, 2):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            # print('############# fold {} begin ###############'.format(fold))

            # Data Code Block
            # train_file_name = 'fold{}_train.txt'.format(fold)
            # test_file_name = 'fold{}_test.txt'.format(fold)
            train_file_name = 'all_data_pair_weibo.txt'
            test_file_name = 'all_data_pair.txt'
            tr_doc_id, tr_y_emotion, tr_y_cause, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len = load_data_weibo('data/' + train_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            te_doc_id, te_y_emotion, te_y_cause, te_y_pairs, te_x, te_sen_len, te_doc_len = load_data_weibo('data/' + test_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            # if FLAGS.trans_type == 'cross_road':
            #     tr_doc_id, tr_y_emotion, tr_y_cause, tr_y_pair, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len = load_data_CR('data/' + train_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            #     te_doc_id, te_y_emotion, te_y_cause, te_y_pair, te_y_pairs, te_x, te_sen_len, te_doc_len = load_data_CR('data/' + test_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            # else:
            #     tr_doc_id, tr_y_emotion, tr_y_cause, tr_y_pair, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len, tr_pair_left_cnt = load_data_WC('data/' + train_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len, window_size=FLAGS.window_size)
            #     te_doc_id, te_y_emotion, te_y_cause, te_y_pair, te_y_pairs, te_x, te_sen_len, te_doc_len, te_pair_left_cnt = load_data_WC('data/' + test_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len, window_size=FLAGS.window_size)

            max_f1_emo, max_f1_cause, max_f1_pair = [-1.] * 3
            print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))
            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, 1., tr_y_emotion, tr_y_cause, FLAGS.batch_size):
                    _, loss, pred_y_emo, true_y_emo, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_emo_op, true_y_emo_op, doc_len], feed_dict=dict(zip(placeholders, train)))
                    if step % 10 == 0:
                        print('step {}: train loss {:.4f} '.format(step, loss))
                        p, r, f1 = cal_prf(pred_y_emo, true_y_emo, doc_len_batch)
                        print('emotion_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                        # p, r, f1 = cal_prf(pred_y_cause, true_y_cause, doc_len_batch)
                        # print('cause_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                        # if FLAGS.trans_type == 'cross_road':
                        #     p, r, f1 = pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, threshold=FLAGS.threshold)
                        # else:
                        #     p, r, f1 = pair_prf_WC(pred_y_pair, true_y_pair, doc_len_batch, threshold=FLAGS.threshold, window_size=FLAGS.window_size)
                        # print('pair_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                    step = step + 1
                # test
                test = [te_x, te_sen_len, te_doc_len, 0., te_y_emotion, te_y_cause]
                loss, pred_y_emo, true_y_emo, doc_len_batch,s_emo = sess.run(
                    [loss_op, pred_y_emo_op, true_y_emo_op, doc_len, s_emo], feed_dict=dict(zip(placeholders, test)))
                print('\nepoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(i, loss, time.time() - start_time))

                p, r, f1 = cal_prf(pred_y_emo, true_y_emo, doc_len_batch)
                if f1 > max_f1_emo:
                    max_p_emo, max_r_emo, max_f1_emo = p, r, f1
                print('emotion_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_emo, max_r_emo, max_f1_emo))

                # p, r, f1 = cal_prf(pred_y_cause, true_y_cause, doc_len_batch)
                # if f1 > max_f1_cause:
                #     max_p_cause, max_r_cause, max_f1_cause = p, r, f1
                # print('cause_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                # print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_cause, max_r_cause, max_f1_cause))

                # if FLAGS.trans_type == 'cross_road':
                #     p, r, f1 = pair_prf_CR(pred_y_pair, true_y_pair, doc_len_batch, threshold=FLAGS.threshold)
                # else:
                #     p, r, f1 = pair_prf_WC(pred_y_pair, true_y_pair, doc_len_batch, te_pair_left_cnt, threshold=FLAGS.threshold, window_size=FLAGS.window_size)
                # if f1 > max_f1_pair:
                #     max_p_pair, max_r_pair, max_f1_pair = p, r, f1
                # print('pair_prediction: test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                # print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p_pair, max_r_pair, max_f1_pair))

                ##将句子embedding保存到pkl文件中
                embedding_file_name = 'Embedding/ECPE_all_data_embedding{}.pkl'.format(i)
                g = open(embedding_file_name, 'wb')
                pickle.dump(s_emo, g)

            print('Optimization Finished!\n')

            ##====================输出原文本和对应句子编码-开始===========================================================
            # def get_pair_data(file_name, doc_id, doc_len, y_pairs, pred_y_emo, x, sen_len, word_idx_rev, s_emo):
            #     g = open(file_name, 'w', encoding='utf-8')
            #     for i in range(len(doc_id)):
            #         g.write(doc_id[i] + ' ' + str(doc_len[i]) + '\n')
            #         g.write(str(y_pairs[i]) + '\n')
            #         for j in range(doc_len[i]):
            #             clause = ''
            #             for k in range(sen_len[i][j]):
            #                 clause = clause + word_idx_rev[x[i][j][k]] + ' '
            #             g.write(str(j + 1) + ', ' + str(pred_y_emo[i][j]) + ', ' + clause + ', ' + str(s_emo[i][j]) + '\n') ##这里的s_emo再次读取出来为字符串
            #     print('write {} done'.format(file_name))
            # get_pair_data('ECPE_all_data_embedding.txt', te_doc_id, te_doc_len, te_y_pairs, pred_y_emo, te_x, te_sen_len, word_idx_rev, s_emo)
            ##====================输出原文本和对应句子编码-结束===========================================================
            # ##将句子embedding保存到pkl文件中
            # g = open('data/ECPE_all_data_embedding.pkl', 'wb')
            # pickle.dump(s_emo, g)

            emo_list.append([max_p_emo, max_r_emo, max_f1_emo])
            # cause_list.append([max_p_cause, max_r_cause, max_f1_cause])

        emo_list, cause_list, pair_list = map(lambda x: np.array(x), [emo_list, cause_list, pair_list])

        # print('\nemotion_prediction: test f1 in 10 fold: {}'.format(emo_list[:, 2:]))
        p, r, f1 = emo_list.mean(axis=0)
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p, r, f1))

        # print('\ncause_prediction: test f1 in 10 fold: {}'.format(cause_list[:, 2:]))
        # p, r, f1 = cause_list.mean(axis=0)
        # print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p, r, f1))

        print_time()


def main(_):
    for FLAGS.model_type in ['Inter-EC']: ##
        FLAGS.trans_type, FLAGS.trans_iter = 'window_constrained', 1
        run()


if __name__ == '__main__':
    tf.app.run()