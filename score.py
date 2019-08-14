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
from sklearn.metrics import jaccard_similarity_score

print('Loading data')
dataset = 'sandy'

# total_need = 71 #sharvey

total_need = 72 #sandy
x, y, vocabulary, vocabulary_inv, label_voc, label_voc_inv = load_need_data(dataset=dataset)

label_size = len(label_voc_inv) # 71
sequence_length = x.shape[1] # 8
vocabulary_size = len(vocabulary_inv) # 640
embedding_dim = 32
filter_sizes = [2,3,4]
num_filters = 128
drop = 0.1

epochs = 100
batch_size = 30


with open('need_data/' + dataset + '_predicted_need.csv') as file_pointer:
    reader = csv.reader(file_pointer, delimiter=',')

    with open('need_data/' + dataset + '_score.csv', 'w', newline='') as writer:
        csv_writer = csv.writer(writer,  delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for idx, row in enumerate(reader):
            predicted_needs = row[8]
            predicted_needs = set(predicted_needs.strip().split())

            predicted_needs_vector = np.zeros(shape=(total_need, ))
            for need in predicted_needs:
                need_idx = label_voc[need]
                predicted_needs_vector[need_idx] = 1

            actual_need = y[idx]
            jc_score = jaccard_similarity_score(actual_need, predicted_needs_vector)

            x_text = ', '.join(row[0:8])
            writer.write(x_text + ', ' + str(jc_score) + '\n')
            print('index:', idx, '; score:', jc_score)