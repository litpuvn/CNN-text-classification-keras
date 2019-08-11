from data_helpers import pad_sentences
from data_helpers import build_vocab
from data_helpers import build_input_data
from data_helpers import clean_str
import numpy as np
import csv


def load_need_and_labels():
    """
       Loads polarity data from files, splits the data into words and generates labels.
       Returns split sentences and labels.
       """
    my_x = []
    my_y = []
    with open('need_data/fixed_need_dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(reader):
            if idx < 1:
                continue

            labels = row[8]

            tmp_x = row[0:8]
            my_x = my_x + [tmp_x]

            my_y = my_y + [labels]
            # for l in labels.split():
            #     tmp_x = row[0:8]
            #     my_x = my_x + [tmp_x]
            #     my_y = my_y + [l]

    print("sample size:", len(my_x))

    return [my_x, my_y]
    # # Load data from files
    # positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # x_text = [s.split(" ") for s in x_text]
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    # return [x_text, y]


def build_need_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])

    # index to word
    tmp_labels = []
    for l in labels:
        tmp_labels = list(set(tmp_labels + l.split()))

    label_voc_inv = sorted(tmp_labels)
    # word to index
    label_voc = {x: i for i, x in enumerate(label_voc_inv)}

    label_size = len(label_voc_inv)
    print("Label size:", label_size)
    my_y = []
    for l in labels:
        tmp = np.zeros(shape=(label_size, ))
        for sub_l in l.split():
            if sub_l in label_voc:
                idx = label_voc[sub_l]
                tmp[idx] = 1

        my_y = my_y + [tmp]
    y = np.array(my_y)
    return [x, y], label_voc, label_voc_inv

def load_need_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_need_and_labels()
    sentences_padded = sentences # don't have to pad sentences
    # sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)

    [x, y], label_voc, label_voc_inv = build_need_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, label_voc, label_voc_inv]


load_need_data()