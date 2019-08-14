from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPool1D
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data
from need_data_helper import  load_need_data
import numpy as np
import csv


print('Loading data')
dataset = 'sandy'
best_weight = 'weights.002-0.1179'

x, y, vocabulary, vocabulary_inv, label_voc, label_voc_inv = load_need_data(dataset=dataset)

# x.shape -> (10662, 56)
# y.shape -> (10662, 2)
# len(vocabulary) -> 18765
# len(vocabulary_inv) -> 18765

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# X_train.shape -> (8529, 56)
# y_train.shape -> (8529, 2)
# X_test.shape -> (2133, 56)
# y_test.shape -> (2133, 2)

label_size = len(label_voc_inv) # 71
sequence_length = x.shape[1] # 8
vocabulary_size = len(vocabulary_inv) # 640
embedding_dim = 32
filter_sizes = [2,3,4]
num_filters = 128
drop = 0.1

epochs = 100
batch_size = 30

# this returns a tensor
print("Creating Model...")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length, embedding_dim,1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)

#### Start simple CNN
# dropout = Dropout(drop)(embedding)
# cov1 = Conv1D(3, 3, padding='valid', activation='relu', strides=1)(dropout)
# max_pool = GlobalMaxPool1D()(cov1)
# output = Dense(units=label_size, activation='softmax')(max_pool)
### END simple CNN ####


output = Dense(units=label_size, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)

weights_path = 'output/' + dataset + '/categorical_crossentropy/' + best_weight + '.hdf5'
model.load_weights(weights_path)

y_hat = model.predict(x)
y_dim = y_hat.shape[1]

max_label_per_day = 25
average_label_per_day = 3

avg_prob = 1.0 / 1

y_labels = []

for item_y in y_hat:
    tmp = np.zeros(y_dim)

    tmp_label = []
    max_index = 0
    max_prob = 0
    for i in range(0, y_dim):
        if item_y[i] >= avg_prob:
            item_y[i] = 1
            tmp_label = tmp_label + [label_voc_inv[i]]

        if item_y[i] > max_prob:
            max_prob = item_y[i]
            max_index = i

    if len(tmp_label) < 1:
        top_three_indices = sorted(range(len(item_y)), key=lambda i: item_y[i], reverse=True)[:24]
        for top_idx in top_three_indices:
            tmp_label.append(label_voc_inv[top_idx])

    y_labels.append(tmp_label)

with open('need_data/' + dataset + '_predicted_need.csv', 'w',  newline='') as writer:
    # csv_writer = csv.writer(writer, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for index, item_x in enumerate(x):
        current_label = y_labels[index]
        y_text = ' '.join(current_label)
        x_text = ''
        for idx, x_i in enumerate(item_x):
            if idx > 0:
                x_text = x_text + ',' + vocabulary_inv[x_i]
            else:
                x_text = vocabulary_inv[x_i]

        writer.write(x_text + ', ' + y_text + '\n')
        # csv_writer.writerow(item_x + [y_text])