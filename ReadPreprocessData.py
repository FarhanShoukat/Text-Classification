from os import listdir
from random import shuffle
import re

HAM = 0
SPAM = 1


def read_data1(data, folder, x):
    files = listdir(folder)
    prefix = folder + "/"

    to_remove = "[0123456789@#$%^&=?!,;:_.(){}`'/+*<>\"¤—\x05„\x14\x00®¯™¡¡\x10»€«·‘\x0e\x03´\x1b§”\x16\x07¬\x15¦…" \
                "\x12\x0f÷\x06\x01~\x11¨©\xad\x02\x08\x13±¥£¶\x17\x19–°•˜’“|]"

    for file in files:
        handle = open(prefix + file)
        handle.__next__()
        a = handle.read().replace("\n", " ").lower()
        a = re.sub(to_remove, " ", a)
        a = a.replace('-', ' ')
        a = a.replace('\\', ' ')
        a = a.replace('[', ' ')
        a = a.replace(']', ' ')
        data.append((" ".join(re.sub(r'\b\w{1,3}\b', ' ', a).split()), x))
        handle.close()
    return data


def read_data(data_type):
    data = []
    # reading legitimate data
    data = read_data1(data, "ham/" + data_type, HAM)
    # reading spam data
    data = read_data1(data, "spam/" + data_type, SPAM)

    # shuffling/mixing spam and legitimate data
    shuffle(data)

    return separate_features_labels(data)


def separate_features_labels(data):
    features = []
    labels = []
    for v in data:
        features.append(v[0])
        labels.append(v[1])

    return features, labels


def read_preprocess():
    train_emails, train_labels = read_data("train")
    test_emails, test_labels = read_data("test")

    return train_emails, train_labels, test_emails, test_labels
