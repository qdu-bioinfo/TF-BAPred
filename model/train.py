from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from model import three_channel_model
from util.metrics import calculate_performace

performance = []
all_prob = {}
all_prob[0] = []
all_average = []


def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 10
    if epoch % decay_step == 0 and epoch > 0:
        return lr * decay_rate
    return lr


def train(train1, train2, train3, test1, test2, test3, train_label, batch_size, epochs, real_labels, model):  # 模型训练
    print("***********训练模型***********")
    optimizer = Adam(learning_rate=0.001)  # 设置初始学习率
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
    model.fit([train1, train2, train3], train_label, batch_size=batch_size, epochs=epochs,
              callbacks=[lr_scheduler_callback])
    probability = model.predict([test1, test2, test3])
    all_prob[0] = all_prob[0] + [val for val in probability]
    y_pred = transfer_label_from_prob(probability)
    acc, precision, sensitivity, specificity, f1_score, MCC = calculate_performace(len(real_labels), y_pred,
                                                                                   real_labels)
    performance.append([acc, precision, sensitivity, specificity, f1_score, MCC])
