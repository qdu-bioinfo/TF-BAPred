from preprocessing.data_loader import get_data
# from preprocessing.data_augmentation import reverse
from model.three_channel_model import network
from model.train import train, performance, all_prob
from feature_extraction.first_channel import feature_loader1
from feature_extraction.second_channel import feature_loader2
from feature_extraction.third_channel import feature_loader3
from tensorflow.keras.layers import Input
from util.visua_result import visualization
import numpy as np

peptide_dict = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
batch_size = 64
epochs = 30
embedding_size = 128
results = []
if __name__ == '__main__':
    peptide_seq = {}
    label = {}
    peptide_seq, label = get_data()
    # peptide_seq_dict_reverse, label = reverse(peptide_seq_dict, label)
    # 第一通道：
    feature1, maxlen = feature_loader1(peptide_seq, peptide_dict)
    # 第二通道：
    feature2 = feature_loader2(peptide_seq, peptide_dict)
    # 第三通道：
    feature3 = feature_loader3(peptide_seq)
    cross_val_num = 5
    for count in range(cross_val_num):  # 五折交叉验证
        for fold in range(cross_val_num):
            train1 = np.array([x for i, x in enumerate(feature1) if i % cross_val_num != fold])
            test1 = np.array([x for i, x in enumerate(feature1) if i % cross_val_num == fold])
            train_label = np.array([x for i, x in enumerate(label) if i % cross_val_num != fold])
            test_label = np.array([x for i, x in enumerate(label) if i % cross_val_num == fold])

        for fold in range(cross_val_num):
            train2 = np.array([x for i, x in enumerate(feature2) if i % cross_val_num != fold])
            test2 = np.array([x for i, x in enumerate(feature2) if i % cross_val_num == fold])

        for fold in range(cross_val_num):
            train3 = np.array([x for i, x in enumerate(feature3) if i % cross_val_num != fold])
            test3 = np.array([x for i, x in enumerate(feature3) if i % cross_val_num == fold])

        fealen2 = train2.shape[-1]
        fealen3 = train3.shape[-1]

        in_x1 = Input(shape=(maxlen,))
        in_x2 = Input(shape=fealen2, )
        in_x3 = Input(shape=fealen3)
        model = network(in_x1, in_x2, in_x3)  # 模型设计
        train(train1, train2, train3, test1, test2, test3, train_label, batch_size, epochs,
              test_label, model)
    print('mean performance of TCN-BAPred')
    print(np.mean(np.array(performance), axis=0))
    prob_result = all_prob[0]

