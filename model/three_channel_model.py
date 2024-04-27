from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model

from tcn import TCN


def network(in_x1, in_x2, in_x3):
    print("***********搭建网络架构***********")
    # 一通道特征
    x1 = Embedding(21, 128)(in_x1)
    x1 = TCN(nb_filters=128, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32, 64, 128])(x1)
    x1 = Dropout(0.2)(x1)
    # 二通道特征
    x2 = Dense(512, 'relu')(in_x2)
    # 三通道特征
    x3 = Dense(512, 'relu')(in_x3)
    # 三通道特征结合
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)  # 添加 Dropout 层进行正则化
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)  # 添加 BatchNormalization 层进行正则化
    x = Dense(256, activation='relu')(x)
    out_x=Dense(1,activation='sigmoid')(x)
    model=Model(inputs=[in_x1,in_x2,in_x3],outputs=[out_x])
    model.summary()
    out_x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[in_x1, in_x2, in_x3], outputs=[out_x])
    model.summary()
    return model

