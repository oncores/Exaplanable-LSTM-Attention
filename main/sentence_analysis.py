#!/usr/bin/env python
# coding: utf-8

def main(prob_pkl,num2label,data,visualize):

    import numpy as np
    import pandas as pd
    import sys
    sys.path.append('..')
    from preprocessing.input_data_index_embedding import load_dataset,split_dataset,text_preprocess,join_date,bow_vocab,load_bow_vocab,bow_label,max_length,x_data_set,y_data_set,int_to_label,labels_to_vecs,make_index_embed,evaluate,split_newdataset_sw,split_ptl_inference,document_label_dataset_training,document_label_dataset_infer,tagging_row_index,row_embed,vecs2labels
    from preprocessing.preprocessing_code_190418 import title_catcher, preprocess, date_process
    import pickle
    import re

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import OneHotEncoder

    from collections import Counter
    from konlpy.tag import Komoran

    import keras
    import matplotlib.pyplot as plt
    from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout, Concatenate, Flatten, Conv1D, Conv2D, GlobalMaxPooling1D, TimeDistributed, SpatialDropout1D, GRU, multiply, Lambda, Reshape, CuDNNGRU, CuDNNLSTM, Permute, RepeatVector, Multiply
    from keras.layers import MaxPool1D
    from keras.models import Model, Sequential
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam
    from keras.engine.topology import Layer
    from keras.preprocessing.sequence import pad_sequences
    from keras import regularizers
    from keras.layers.normalization import BatchNormalization
    from keras.layers import Activation
    import seaborn as sn

    import tensorflow as tf
    import warnings
    import os
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("Analyzing Sentence")

    test_prob = prob_pkl
    data_dir = data
    vocab_to_int_dir = './preprocessing/insu_vocab_to_int.pkl'
    int_to_vocab_dir = './preprocessing/insu_int_to_vocab.pkl'

    origin_data = load_dataset(data_dir)
    origin_data = join_date(origin_data)
    origin_data.head(3)

    ptl_df = document_label_dataset_infer(origin_data)
    ptl_df.head(3)

    def split_newdataset_sw_(data, standard):
        contract_names = np.unique(data[standard])

        x_all = []

        for name in contract_names:
            temp = data[data[standard] == name]
            temp_contract = []

            for c in temp['doc'].values:
                temp_contract.append(c)
            x_all.append(temp_contract)
        x_all = [x_all[con][0] for con in range(len(x_all))]

        return x_all

    x_all =  split_ptl_inference(ptl_df, 'index')

    valid_class = np.array([
    '-', 'PR-02-01', 'PR-02-02', 'PR-02-03', 'PR-02-04', 'PR-02-05',
        'PR-02-06', 'PR-02-07', 'PR-02-08', 'PR-02-09', 'PR-02-10',
        'PR-02-11', 'PR-02-12', 'PR-02-13', 'PR-02-14', 'PR-02-15',
        'PR-02-16', 'PR-02-17', 'PR-02-18', 'PR-02-19', 'PR-02-20',
        'PR-02-21', 'PR-02-22', 'PR-02-24', 'PR-02-25', 'PR-02-26',
        'PR-02-27', 'PR-02-28', 'PR-02-29', 'PR-02-30', 'PR-02-31',
        'PR-02-32', 'PR-03-01', 'PR-03-02', 'PR-03-03', 'PR-03-04',
        'PR-04-01', 'PR-04-02', 'PR-04-03', 'PR-04-04', 'PR-04-05',
        'PR-04-06', 'PR-04-07', 'PR-04-08', 'PR-04-09', 'PR-04-10',
        'PR-04-11', 'PR-04-12', 'PR-04-13', 'PR-04-14', 'PR-04-15',
        'PR-04-16', 'PR-04-17', 'PR-04-18', 'PR-04-19', 'PR-04-20',
        'PR-04-21', 'PR-04-22', 'PR-04-23', 'PR-04-24', 'PR-04-25',
        'PR-04-26', 'PR-04-27', 'PR-04-28', 'PR-04-29', 'PR-04-30',
        'PR-04-31', 'PR-04-32', 'PR-04-33', 'PR-04-34', 'PR-04-35',
        'PR-04-36', 'PR-04-37', 'PR-04-38', 'PR-04-39', 'PR-04-40',
        'PR-04-41', 'PR-04-42', 'PR-04-43', 'PR-04-44', 'PR-04-45',
        'PR-04-46', 'PR-04-47', 'PR-04-48', 'PR-04-49', 'PR-04-50',
        'PR-04-51', 'PR-04-52', 'PR-05-01', 'PR-05-02', 'PR-05-03',
        'PR-05-04', 'PR-05-05', 'PR-05-06', 'PR-05-07', 'PR-05-08',
        'PR-05-09', 'PR-05-10', 'PR-05-11', 'PR-05-12', 'PR-05-13',
        'PR-05-14', 'PR-05-15', 'PR-05-16', 'PR-05-17', 'PR-05-18',
        'PR-05-19', 'PR-05-20', 'PR-05-21', 'PR-06-01', 'PR-06-02',
        'PR-06-03', 'PR-06-04', 'PR-06-05', 'PR-06-06', 'PR-06-07',
        'PR-06-08', 'PR-06-09', 'PR-06-10', 'PR-06-11', 'PR-06-12',
        'PR-06-13', 'PR-06-14', 'PR-07-01', 'PR-07-02', 'PR-07-03',
        'PR-07-04', 'PR-07-05', 'PR-07-06', 'PR-07-07', 'PR-08-01',
        'PR-08-02', 'PR-08-03', 'PR-08-04', 'PR-08-05', 'PR-08-06',
        'PR-08-07', 'PR-08-08', 'PR-08-09', 'PR-08-10', 'PR-08-11',
        'PR-08-12', 'PR-08-13', 'PR-08-14', 'PR-08-15', 'PR-08-16',
        'PR-08-17', 'PR-08-18', 'PR-08-19', 'PR-08-20', 'PR-08-21',
        'PR-08-22', 'PR-08-23', 'PR-08-24', 'PR-08-25', 'PR-08-26',
        'PR-08-27', 'PR-08-28', 'PR-08-29', 'PR-09-01', 'PR-09-02',
        'PR-09-03', 'PR-09-04', 'PR-09-05', 'PR-09-06', 'PR-09-07',
        'PR-09-08', 'PR-09-09', 'PR-09-10', 'PR-09-11', 'PR-09-12',
        'PR-09-13', 'PR-09-14', 'PR-09-15', 'PR-10-01', 'PR-10-02',
        'PR-10-03', 'PR-10-04', 'PR-10-05', 'PR-10-06', 'PR-11-01',
        'PR-11-02', 'PR-11-03', 'PR-11-04', 'PR-11-05', 'PR-11-07',
        'PR-11-08', 'PR-12-01', 'PR-13-01', 'PR-13-02', 'PR-13-03',
        'PR-14-01', 'PR-14-02', 'PR-14-03', 'PR-14-04', 'PR-14-05',
        'PR-14-06', 'PR-15-01', 'PR-16-01', 'PR-17-01', 'PR-18-01',
        'PR-19-01', 'PR-19-02', 'PR-20-01', 'PR-20-02', 'PR-20-03',
        'PR-20-04', 'PR-20-05', 'PR-20-06', 'PR-20-07', 'PR-20-08',
        'PR-20-09', 'PR-20-10', 'PR-20-11', 'PR-20-12', 'PR-20-13',
        'PR-20-14', 'PR-21-01', 'PR-21-02', 'PR-22-01', 'PR-23-01',
        'PR-23-02', 'PR-23-03', 'PR-23-04', 'PR-23-05', 'PR-23-06',
        'PR-24-01', 'PR-24-02', 'PR-24-03', 'PR-24-04', 'PR-24-05',
        'PR-24-06', 'PR-24-07', 'PR-24-08', 'PR-24-09', 'PR-24-10',
        'PR-25-01', 'PR-25-02', 'PR-25-03', 'PR-25-04', 'PR-26-01',
        'PR-27-01', 'PR-28-01', 'PR-29-01', 'PR-29-02', 'PR-29-03',
        'PR-29-04', 'PR-29-05', 'PR-29-06', 'PR-30-01'])

    class_size = len(valid_class)

    vocab_to_int, int_to_vocab = load_bow_vocab(vocab_to_int_dir, int_to_vocab_dir)

    max_len = 350
    max_row = 121

    unique_par_class = ['정의', '인수_및_모집', '본_사채의_발행조건', '수요예측', '모집관계사항', '불성실_수요예측_참여자의_관리', '발행회사의_보장', '기업실사',
                        '확약_또는_선약', '인수시기', '제서식의_작성_및_공고', '수수료', '비용', '원리금_상환사무의_대행', '인수_및_모집_일정의_변경', '사채권의_발행여부',
                        '채권상장신처_및_채권등록_발행', '특약사항', '사채금_사용용도', '원리금_지급의무', '책임부담', '해지_또는_해제', '통보_및_요청', '자료제출',
                        '평과결과_공시_등', '관할법원', '계약의_해석원칙_등', '공모금리_결정_및_배정', '개별책임']

    label_to_int = bow_label(unique_par_class)[0]

    test_proba = np.concatenate([[test_prob[num] for cnt in x_all[num]] for num in range(len(x_all))])

    x_row = tagging_row_index(x_all)
    x_row_ = row_embed(x_row, max_row)
    x_all_ = x_data_set(x_all, max_len, vocab_to_int)

    n_words = len(int_to_vocab) + 2
    embed_size = 100

    batch_size = 8
    learning_rate = 0.0001
    epochs = 500

    sentence_wise_lstm_size = 128

    dense_dropout = 0.5
    l2_reg = regularizers.l2(0.0001)

    dense_size = 128
    attention_dim = 100
    rnn_dim = 256

    class AttentionLayer(Layer):
        def __init__(self, attention_dim, **kwargs):
            self.attention_dim = attention_dim
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='Attention_Weight',
                                    shape=(input_shape[-1], self.attention_dim),
                                    initializer='random_normal',
                                    trainable=True)
            self.b = self.add_weight(name='Attention_Bias',
                                    shape=(self.attention_dim, ),
                                    initializer='random_normal',
                                    trainable=True)
            self.u = self.add_weight(name='Attention_Context_Vector',
                                    shape=(self.attention_dim, 1),
                                    initializer='random_normal',
                                    trainable=True)
            super(AttentionLayer, self).build(input_shape)

        def call(self, x):
            u_it = K.tanh(K.dot(x, self.W) + self.b)
            a_it = K.dot(u_it, self.u)
            a_it = K.squeeze(a_it, -1)
            a_it = K.softmax(a_it)
            return a_it

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[1])

    def WeightedSum(attentions, representations):
        repeated_attentions = RepeatVector(K.int_shape(representations)[-1])(attentions)

        repeated_attentions = Permute([2, 1])(repeated_attentions)

        aggregated_representation = Multiply()([representations, repeated_attentions])
        aggregated_representation = Lambda(lambda x: K.sum(x, axis=1))(aggregated_representation)

        return aggregated_representation

    def SenWeightedSum(attentions, representations):
        repeated_attentions = RepeatVector(K.int_shape(representations)[-1])(attentions)
        repeated_attentions = Permute([2, 1])(repeated_attentions)
        aggregated_representation = Multiply()([representations, repeated_attentions])
        return aggregated_representation

    def TabSen():

        K.clear_session()
        np.random.seed(1201)

        row_embed = Input(shape = (max_row, ), name = 'row_input')
        col_embed = Input(shape = (len(unique_par_class), ), name = 'col_input')

        row_layer = Dense(128)(row_embed)
        col_layer = Dense(128)(col_embed)

        word_inp_embed = Input(shape = (None, ), name = 'word_input')
        word_embed = Embedding(n_words, embed_size, trainable = True)(word_inp_embed)

        lstm = Bidirectional(CuDNNLSTM(sentence_wise_lstm_size, return_sequences=True))(word_embed)
        lstm_bn = BatchNormalization()(lstm)

        attn_score = AttentionLayer(attention_dim)(lstm_bn)
        attn_out = WeightedSum(attn_score, lstm_bn)

        concat = Concatenate()([attn_out, row_layer, col_layer])

        fc_layer = Dense(dense_size,
                    activation='relu',
                    kernel_regularizer = keras.regularizers.l2(1e-5),
                    bias_regularizer = keras.regularizers.l1(1e-3))(concat)
        dropout = Dropout(dense_dropout)(fc_layer)
        output = Dense(class_size, activation = 'softmax')(dropout)

        model = Model(inputs = [word_inp_embed, row_embed, col_embed], outputs = output)

        word_attention_extractor = Model(inputs=[word_inp_embed],
                                        outputs=[attn_score])

        model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate), metrics = ['accuracy'])
        return model, word_attention_extractor

    print("Load Sentence Model")
    tabsen, word_attention_extractor = TabSen()
    tabsen.load_weights('./model/sentence/33-0.1379.hdf5')

    def int_to_label(y_vectors, valid_class):
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(valid_class.reshape(-1,1))
        labels = enc.inverse_transform(y_vectors)
        return labels

    int_to_vocab[0] = 'pad'
    int_to_vocab[1] = 'UNK'

    import matplotlib
    import matplotlib.font_manager as fm
    fm.get_fontconfig_fonts()
    font_location = 'font/malgun.ttf' # For Windows
    font_name = fm.FontProperties(fname=font_location).get_name()

    init_x_dict = {}
    init_x_dict['sequence'] = x_all_

    split = 'sequence'
    threshold=0.5

    pred_attention = word_attention_extractor.predict(init_x_dict[split])
    pred=tabsen.predict([x_all_, x_row_, test_proba])
    labels = [i for i in np.concatenate(int_to_label(pred, valid_class))]

    if visualize=='on' :
        print('sentence visual file extracting...')
        plt.rcParams.update({'figure.max_open_warning': 0})
        words_list = []
        for sent_idx, sentence in enumerate(init_x_dict[split]):
            if sentence[0] == 0:
                continue

            for word_idx in range(max_len):
                if sentence[word_idx] == 0:
                    words = [int_to_vocab[word_id] for word_id in sentence[0:word_idx]]
                    pred_att = pred_attention[sent_idx][0:len(words)]
                    pred_att = np.expand_dims(pred_att, axis=0)
                    break

            fig, ax = plt.subplots(figsize=(len(words), 1))
            plt.rc('font', family=font_name)
            plt.rc('xtick', labelsize=12)
            midpoint = (max(pred_att[:, 0]) - min(pred_att[:, 0])) / 2
            heatmap = sn.heatmap(pred_att, xticklabels=words, yticklabels=False, square=True, linewidths=0.1, cmap='coolwarm', center=midpoint, vmin=0, vmax=1)
            words_list.append([np.array(pred_att[0]), words, labels[sent_idx]])
            plt.xticks(rotation=45)
            plt.title(labels[sent_idx],)
            fig = plt.gcf()
            fig.savefig('./output/sentence/output_sen_vis/sentence_attention_{}_{}'.format(split,sent_idx+1), bbox_inches = "tight")
            plt.close(fig)

        scores = [i[0] for i in words_list]
        tokens = [i[1] for i in words_list]
        for sen_idx, (score,token) in enumerate(zip(scores, tokens)) :
            df = pd.DataFrame(score,token).T
            df.to_excel('./output/sentence/output_sen_dis/sentence_distribution_{}_{}.xlsx'.format(split, sen_idx+1), index=False)
        print('Check the /output/sentence/output_sen_dis, output_sen_vis', end='\n\n')
    elif visualize=='off':
        print('skip visualze', end='\n\n')
    else :
        print('visualize param check!')

    line_index = pd.read_excel('./data/index_line_label.xlsx', header=None)

    line_dict = {}
    for i in line_index.values:
        line_dict[i[1]] = i[0]
    line_dict['-'] = '-'

    para_dict = num2label
    output_filename = './output/contract_tagging.xlsx'
    def output_prediction(infer_pred):
        pred_label = [line_dict[i] for i in np.concatenate(int_to_label(infer_pred, valid_class))]

        par_pred = [np.where(i>=threshold)[0] for i in test_prob]
        par_pred_label = [', '.join([para_dict[j] for j in i]) for i in par_pred]
        par_pred_length = [len(i) for i in ptl_df.doc.values]
        par_pred_label_ = np.concatenate([length*[pars] for length,pars in zip(par_pred_length, par_pred_label)])
        
        origin_data['line_label'] = pred_label
        
        origin_data['par_label'] = par_pred_label_

        output = origin_data.iloc[:,:-1]
        output.columns = ['문서번호','문단번호','조항번호','라인번호','내용','문단클래스','라인클래스']
        output.to_excel(output_filename, index=False)

        return output

    output = output_prediction(pred)
    print("Finish Contract Analysis")
    print("Check the Output file ", output_filename)