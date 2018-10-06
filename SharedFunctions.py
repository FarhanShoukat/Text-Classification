from ReadPreprocessData import HAM, SPAM

import time
fmt = '%H:%M:%S'


def get_current_time():
    time.ctime()
    return time.strftime(fmt)


def find_accuracy(predicted_labels, actual_labels):
    accuracy = 0
    wrong_ham = 0
    wrong_spam = 0

    i = 0
    l = len(actual_labels)
    while i < l:
        if actual_labels[i] == predicted_labels[i]:
            accuracy += 1
        elif actual_labels[i] == HAM:
            wrong_ham += 1
        else:
            wrong_spam += 1
        i += 1

    print("Accurately Identified:", accuracy, "Percentage:", accuracy * 100 / l)
    print("Wrongly Identified Legitimate as Spam:", wrong_ham, "Percentage:", wrong_ham * 100 / l)
    print("Wrongly Identified Spam as Legitimate:", wrong_spam, "Percentage:", wrong_spam * 100 / l)
