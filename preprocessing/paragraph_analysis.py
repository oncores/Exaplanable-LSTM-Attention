def main(data, visulize):

    import tensorflow as tf
    import warnings
    import os
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    import time
    import re
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from preprocessing.preprocessing_code_190418 import preprocess, title_catcher, date_process, phone_process, time_process, title_process
    from konlpy.tag import Komoran

    import keras
    from keras import backend as K
    from keras.layers import Input, Embedding, Bidirectional, CuDNNLSTM, BatchNormalization
    from keras.layers import RepeatVector, Permute, Multiply, Lambda, TimeDistributed
    from keras.layers import Dense, Flatten
    from keras.models import Model, Sequential, model_from_json
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam
    from keras.engine.topology import Layer
    from keras.preprocessing.sequence import pad_sequences


    print("Analyzing Paragraph")

    def load_dataset(data):
        origin_data = pd.read_excel(data)
        if len(origin_data.columns) == 9:
            origin_data.columns = ['doc_id', 'par_id', 'art_id', 'line_id', 'text', 'par_label', 'line_label', 'none1', 'none2']
            origin_data['split_id'] = origin_data['doc_id'].map(str) + '_' + origin_data['par_label']
        elif len(origin_data.columns) == 7:
            origin_data.columns = ['doc_id', 'par_id', 'art_id', 'line_id', 'text', 'par_label', 'line_label']
            origin_data['split_id'] = origin_data['doc_id'].map(str) + '_' + origin_data['par_label']
        else:
            raise ValueError("Columns is not 7 or 9!")
        return origin_data
    
    def join_date(original_data):
        p = re.compile('[0-9]{4}[ .년]{0,3}[0-9]{1,2}[ .월]{0,3}[0-9]{1,2}[ .일]{0,3}')
        split_date_idx = [idx for idx, [lines, labels] in enumerate(original_data[['text', 'line_label']].values) if len(p.findall(str(lines))) == 1 and labels == 'PR-04-13' and len(lines) <= 15]
        date_diff = [i for i in range(len(split_date_idx) - 1) if split_date_idx[i + 1] - split_date_idx[i] >= 3]

        try:
            seq_date_idx = []
            seq_date_idx.append(split_date_idx[0:date_diff[0] + 1])
            for i in range(len(date_diff) - 1):
                seq_date_idx.append(split_date_idx[date_diff[i] + 1:date_diff[i + 1] + 1])
            for j in seq_date_idx:
                original_data.iloc[j[0], 4] = ' '.join(original_data.iloc[j]['text'].values)
            processed_data = original_data.drop(np.concatenate([i[1:] for i in seq_date_idx]))
        except:
            processed_data = original_data
        return processed_data

    def document_label_dataset(processed_data):
        processed_data = processed_data.reset_index()
        contents = processed_data.iloc[:, 5].tolist()

        temp = []
        for text in processed_data['text']:
            try:
                result = title_catcher(text)
                temp.append(result)
            except BaseException:
                temp.append(False)
        processed_data['title'] = temp

        start_idx = processed_data[processed_data['title'] == True].index.tolist()
        end_idx = start_idx[1:]
        end_idx.append(processed_data.index[-1] + 1)

        contract = []
        for start, end in zip(start_idx, end_idx):
            temp = processed_data['text'][start:end]
            contract.append(list(temp.values))

        new_df = pd.DataFrame({"doc": contract}).reset_index()
        return new_df

    def split_newdataset(data, standard, seed):
        contract_names = np.unique(data[standard])
        x_all = []

        for name in contract_names:
            temp = data[data[standard] == name]
            temp_contract = []

            for c in temp['doc'].values:
                temp_contract.append(c)
            x_all.append(temp_contract)

        return x_all

    def make_paragraph_x_dataset(x):
        return [x[paragraph][0] for paragraph in range(len(x))]

    def make_paragraph_y_dataset(y):
        return [y[paragraph][0].split(',') for paragraph in range(len(y))]

    def text_preprocess(text):
        text = preprocess(text)
        text = title_process(text)
        text = time_process(text)
        text = date_process(text)
        text = phone_process(text)
        text = re.sub('[^가-힣".,()~%_ ]+', '', text)
        try:
            text = ' '.join(np.array(komoran.pos(text))[:, 0])
        except BaseException:
            text = '_빈칸_'
        return text

    def word2idx(text):
        try:
            re_text = re.sub('[^가-힣".,()~%_ ]+', '', text)
            re_text = re.sub('[^가-힣_]+','PUNC', re_text)
            return vocab_to_int[re_text]
        except BaseException:
            return 1

    def sentence2idx(sentence):
        p = re.compile('([ㄱ-ㅎㅏ-ㅣ]+)')
        return [word2idx(word) for word in sentence.split() if len(p.findall(word)) == 0]

    def contract2idx(contract, max_len):
        temp = [sentence2idx(text_preprocess(line)) for line in contract]
        return pad_sequences(temp, maxlen=max_len)

    def x_dataset(contracts, max_row, max_len):
        contracts = [contract2idx(contract, max_len) for contract in contracts]
        return pad_sequences(contracts, maxlen=max_row, padding='post')

    def y_dataset(labels):
        output = np.zeros(class_size)
        for label in labels:
            if label in label2num.keys():
                output += np.eye(class_size)[label2num[label]]
        return output

    class AttentionLayer(Layer):
        def __init__(self, attention_dim=100, **kwargs):
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
        aggregated_representation = Lambda(lambda x: K.sum(x, axis=1))(aggregated_representation)
        return aggregated_representation

    def Hie_Attention():
        embedding_layer = Embedding(input_dim=max_nb_words,
                                    output_dim=embedding_dim,
                                    input_length=max_len,
                                    trainable=True,
                                    mask_zero=False)

        # Sentence Encoder
        sentence_input = Input(shape=(max_len, ), name='sentence_input')
        embedded_sentence = embedding_layer(sentence_input)
        contextualized_sentence = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True), name="WORD_BiLSTM")(embedded_sentence)
        word_attention = AttentionLayer(attention_dim)(contextualized_sentence)
        sentence_representation = WeightedSum(word_attention, contextualized_sentence)
        sentence_encoder = Model(inputs=[sentence_input], outputs=[sentence_representation])

        # Document Encoder
        document_input = Input(shape=(max_row, max_len,), name='document_input')
        embedded_document = TimeDistributed(sentence_encoder)(document_input)
        contextualized_document = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True), name="SENTENCE_BiLSTM")(embedded_document)

        sentence_attention = AttentionLayer(attention_dim)(contextualized_document)
        document_representation = SenWeightedSum(sentence_attention, contextualized_document)
        layer = Dense(dense_size, activation='relu')(document_representation)
        output = Dense(class_size, activation='sigmoid')(layer)
        model = Model(inputs=[document_input], outputs=[output])

        # Attention Extractor
        word_attention_extractor = Model(inputs=[sentence_input], outputs=[word_attention])
        word_attentions = TimeDistributed(word_attention_extractor)(document_input)
        attention_extractor = Model(inputs=[document_input], outputs=[sentence_attention])
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
        return model, attention_extractor

    def vecs2labels(vecs):
        output = []
        for i, vec in enumerate(vecs):
            if vec == 1:
                output.append(num2label[i])
        return output

    def model_output(output_):
        return [[vecs2labels((output > threshold) * 1)] for output in output_]

    def model_pred(model, input_):
        out = []
        for contract in input_:
            out.extend(model.predict(np.array([contract])))
        return [[vecs2labels((output > threshold) * 1)] for output in out]

    def model_probability(model, input_):
        out = []
        for contract in input_:
            out.extend(model.predict(np.array([contract])))
        return [output for output in out]

    def multilabel_evaluate(class_pred, output, original_x):
        accuracy = []
        class_pred = [class_pred[label][0] for label in range(len(class_pred))]
        output = [output[label][0] for label in range(len(output))]
        for i, contract in enumerate(original_x):
            contract_class_pred = class_pred[i]
            contract_output = output[i]

            ans = set(contract_class_pred)
            pred = set(contract_output)

            if (pred <= ans and len(pred) > 0) or (len(pred) == 0 and len(ans) == 0):
                score = 1
            else:
                score = 0
            accuracy.append(score)
        return np.mean(accuracy)

    fm.get_fontconfig_fonts()
    font_location = './font/H2GTRE.TTF'
    font_name = fm.FontProperties(fname=font_location).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    print("Load Data directory:", data)
    original_data = load_dataset(data)
    processed_data = join_date(original_data)
    new_df = document_label_dataset(processed_data)
    x_all = split_newdataset(new_df, 'index', 1103)
    x_all = make_paragraph_x_dataset(x_all)

    para_index = pd.read_excel('./data/index_par_label.xlsx')
    para_index = para_index.iloc[:, 1:]
    para_dict = {}
    for i in para_index.values:
        para_dict[i[0]] = i[1]
    label2num = para_dict
    num2label = {word: i for i, word in label2num.items()}

    class_size = len(num2label)

    max_row = 100
    max_len = 200

    komoran = Komoran(userdic='preprocessing/userdict_190411.txt')

    with open('./preprocessing/para_int_to_vocab.pickle','rb') as f:
        int_to_vocab = pickle.load(f)
        
    with open('./preprocessing/para_vocab_to_int.pickle','rb') as f:
        vocab_to_int = pickle.load(f)

    x_all_ = x_dataset(x_all, max_row, max_len)

    max_nb_words = len(int_to_vocab) + 1
    embedding_dim = 200
    attention_dim = 100
    lstm_dim = 100
    learning_rate = 0.0001
    dense_size = 256
    
    print("Load Pragraph Model")
    model, attention_extractor = Hie_Attention()
    model.load_weights("model/para/model_30.h5")
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate), metrics=['accuracy'])

    attention_distribution = attention_extractor.predict(x_all_)
    df_result = pd.DataFrame()
   
    sentence_value, attention_value = [], []
    for sentence, attention in zip(x_all,attention_distribution):
        if len(sentence) <= 100:
            sentence_value.append(sentence[:len(sentence)])
            attention_value.append(attention[:len(sentence)])
        else:
            sentence_value.append(sentence[:100])
            attention_value.append(attention[:100])

    if visulize=='on' :
        print('paragraph visual file extracting...')
        for idx_1, sentence_attention in enumerate(zip(sentence_value, attention_value)):
            sentence = sentence_attention[0]
            attention = sentence_attention[1]
            tmp1 = np.array(attention).reshape(-1, 1)
            tmp2 = np.array(sentence).reshape(-1, 1)

            fig, ax = plt.subplots(figsize = (10, len(tmp1)))
            ax.tick_params(labelsize = 20)
            ax = sns.heatmap(data = tmp1,  yticklabels = tmp2, annot = True, cmap = "Reds", annot_kws = {"size" : 30}, cbar = False)
            plt.savefig("./output/paragraph/output_par_vis/" + "paragraph_attention_sequence_" + str(idx_1 + 1) + ".png", bbox_inches = "tight")
            plt.close(fig)
        print('Check the /output/paragraph/output_par_dis, output_par_vis', end='\n\n')
    else :
        print('skip visualize', end='\n\n')

    df_sample = pd.DataFrame({"Sentence" : sentence_value, "Attention": attention_value})    
    df_result = pd.concat([df_result, df_sample], axis = 0)
    
    df_result.to_excel('output/paragraph/output_par_dis/paragraph_distribution.xlsx', encoding = "utf-8") # Data to Excel
    paragraph_prob = model_probability(model, x_all_)
    return paragraph_prob, num2label