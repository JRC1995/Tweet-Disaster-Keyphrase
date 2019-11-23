import pickle
import random
import math
import numpy as np
from nltk.stem import WordNetLemmatizer
import string

random.seed(a=101)
wordnet_lemmatizer = WordNetLemmatizer()

window_size = 3
with open('Processed_Data/vocab_and_embd.pkl', 'rb') as fp:
    data = pickle.load(fp)

vocab2idx = data[0]


def vectorize(tweets):
    vec_tweets = []
    for tweet in tweets:
        vec_tweet = [vocab2idx.get(word, vocab2idx['<unk>']) for word in tweet]
        vec_tweets.append(vec_tweet)
    return vec_tweets


def lemmatize(word):
    word = word.strip("").strip(" ").strip("\t").strip("\n").strip("\b").strip("\n\n")
    if word in disaster_vocab:
        return word
    else:
        word = word.split("'s")[0]
        if word == "sos" or word == "stuck":
            return word
        elif word in string.punctuation:
            return "<NOT_IN_LIST>"
        else:
            l1 = wordnet_lemmatizer.lemmatize(word, 'v')
            l2 = wordnet_lemmatizer.lemmatize(l1, 'a')
            l3 = wordnet_lemmatizer.lemmatize(l2, 'r')
            l4 = wordnet_lemmatizer.lemmatize(l3, 'n')
            return l4


max_char_len = 20

with open('Processed_Data/word_to_ipa_vec.pkl', 'rb') as fp:
    data = pickle.load(fp)

word2ipa_vec = data[0]
word2phono_vec = data[1]
phono_dim = data[2]


tweet_pos_vocab = ['N', 'O', 'S', '^', 'Z', 'V', 'L', 'M', 'A', 'R', '!',
                   'D', 'P', '&', 'T', 'X', 'Y', '~', 'U', 'E', '$', ',', 'G']
pos2idx = {}

for pos in tweet_pos_vocab:
    pos2idx[pos] = len(pos2idx)


def ipafy(word):
    pad = np.zeros((max_char_len), np.float32)
    return word2ipa_vec.get(word, pad)


def phonofy(word):
    pad = np.zeros((max_char_len, phono_dim), np.float32)
    return word2phono_vec.get(word, pad)


def vectorize_pos(pos_tweets):
    vec_pos_tweets = []
    for pos_tweet in pos_tweets:
        vec_pos_tweet = [pos2idx.get(pos, pos2idx[","]) for pos in pos_tweet]
        vec_pos_tweets.append(vec_pos_tweet)
    return vec_pos_tweets


with open('Processed_Data/Intermediate_Data.pkl', 'rb') as fp:
    data = pickle.load(fp)

label2idx2label1idx = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}

train_tweets = data[0]
train_labels_2 = data[1]
train_labels_1 = []
for tweet in train_labels_2:
    label_1 = [label2idx2label1idx[label] for label in tweet]
    train_labels_1.append(label_1)
train_pos = data[2]

val_tweets = data[3]
val_labels_2 = data[4]
val_labels_1 = []
for tweet in val_labels_2:
    val_labels_1.append([label2idx2label1idx[label] for label in tweet])
val_pos = data[5]
test_tweets = data[6]
test_labels_2 = data[7]
test_labels_1 = {}

for disaster_type in test_labels_2:
    test_labels_1[disaster_type] = []
    for tweet in test_labels_2[disaster_type]:
        test_labels_1[disaster_type].append([label2idx2label1idx[label] for label in tweet])
test_pos = data[8]

x = [i for i in range(0, len(train_tweets))]
random.shuffle(x)

train_tweets = [train_tweets[x[i]]
                for i in range(0, len(train_tweets))]
train_labels_1 = [train_labels_1[x[i]]
                  for i in range(0, len(train_tweets))]
train_labels_2 = [train_labels_2[x[i]]
                  for i in range(0, len(train_tweets))]
train_pos = [train_pos[x[i]]
             for i in range(0, len(train_tweets))]

print("\nSome sample training data\n")
for i in range(0, 5):
    print("Sentence: {}".format(train_tweets[i]))
    print("Label_1: {}".format(train_labels_1[i]))
    print("Label_2: {}".format(train_labels_2[i]))
    print("POS tags: {}".format(train_pos[i]))
    print("\n\n")

print("\nSome sample testing data:\n")

# for disaster in test_tweets:
# print(len(test_tweets[disaster]))
for disaster_type in test_tweets:
    print("FROM {}\n".format(disaster_type))
    for i in range(0, 5):
        print("Sentence: {}".format(test_tweets[disaster_type][i]))
        print("Label_1: {}".format(test_labels_1[disaster_type][i]))
        print("Label_2: {}".format(test_labels_2[disaster_type][i]))
        print("POS tags: {}".format(test_pos[disaster_type][i]))
        print("\n\n")


train_ipa = [list(map(ipafy, tweet)) for tweet in train_tweets]
test_ipa = {}
for disaster_type in test_tweets:
    test_ipa[disaster_type] = [list(map(ipafy, tweet)) for tweet in test_tweets[disaster_type]]
val_ipa = [list(map(ipafy, tweet)) for tweet in val_tweets]

train_phono = [list(map(phonofy, tweet)) for tweet in train_tweets]
val_phono = [list(map(phonofy, tweet)) for tweet in val_tweets]
test_phono = {}
for disaster_type in test_tweets:
    test_phono[disaster_type] = [list(map(phonofy, tweet)) for tweet in test_tweets[disaster_type]]


train_pos = vectorize_pos(train_pos)
val_pos = vectorize_pos(val_pos)
test_pos_vec = {}
for disaster_type in test_tweets:
    test_pos_vec[disaster_type] = vectorize_pos(test_pos[disaster_type])
test_pos = test_pos_vec

train_tweets_vec = vectorize(train_tweets)
val_tweets_vec = vectorize(val_tweets)
test_tweets_vec = {}
for disaster_type in test_tweets:
    test_tweets_vec[disaster_type] = vectorize(test_tweets[disaster_type])

print("AFTER VECTORIZATION:\n\n")

print("\nSome sample training data\n")
for i in range(0, 5):
    print("Sentence: {}".format(train_tweets[i]))
    print("Sentence: {}".format(train_tweets_vec[i]))
    print("Label_1: {}".format(train_labels_1[i]))
    print("Label_2: {}".format(train_labels_2[i]))
    print("POS: {}".format(train_pos[i]))
    print("IPA: {}".format(train_ipa[i]))
    print("Phono: {}".format(train_phono[i][0]))
    print("\n\n")

print("\nSome sample testing data:\n")
for disaster_type in test_tweets:
    print("FROM {}\n".format(disaster_type))
    for i in range(0, 5):
        print("Sentence: {}".format(test_tweets[disaster_type][i]))
        print("Sentence: {}".format(test_tweets_vec[disaster_type][i]))
        print("Label_1: {}".format(test_labels_1[disaster_type][i]))
        print("Label_2: {}".format(test_labels_2[disaster_type][i]))
        print("POS tags: {}".format(test_pos[disaster_type][i]))
        print("IPA: {}".format(test_ipa[disaster_type][i]))
        print("Phono: {}".format(test_phono[disaster_type][i][0]))
        print("\n\n")


def set_window(tweets):

    windowed_tweets = []

    for tweet in tweets:
        windowed_tweet = []
        for i in range(0, len(tweet)):
            window_word = []
            j = 0
            d = window_size // 2
            while j < window_size:
                if j <= window_size//2-1:
                    if i-d == -1:
                        window_word.append(vocab2idx['<sos>'])
                    elif i-d < -1:
                        window_word.append(vocab2idx['<pad>'])
                    else:
                        window_word.append(tweet[i-d])
                else:
                    # d should become non-positive here
                    if i-d == len(tweet):
                        window_word.append(vocab2idx['<eos>'])
                    elif i-d > len(tweet):
                        window_word.append(vocab2idx['<pad>'])
                    else:
                        window_word.append(tweet[i-d])

                d -= 1
                j += 1
            windowed_tweet.append(window_word)

        windowed_tweets.append(windowed_tweet)

    return windowed_tweets


train_tweets_window = set_window(train_tweets_vec)
val_tweets_window = set_window(val_tweets_vec)
test_tweets_window = {}
for disaster_type in test_tweets:
    test_tweets_window[disaster_type] = set_window(test_tweets_vec[disaster_type])

print("\nAFTER WINDOWING: \n")

print("\nSome sample training data\n")
for i in range(0, 5):
    print("Sentence: {}".format(train_tweets[i]))
    print("Sentence: {}".format(train_tweets_vec[i]))
    print("Windowed Sentence: {}".format(train_tweets_window[i]))
    print("Label_1: {}".format(train_labels_1[i]))
    print("Label_2: {}".format(train_labels_2[i]))
    print("POS: {}".format(train_pos[i]))
    print("IPA: {}".format(train_ipa[i]))
    print("Phono: {}".format(train_phono[i][0]))
    print("\n\n")

print("\nSome sample testing data:\n")
for disaster_type in test_tweets:
    print("FROM {}\n".format(disaster_type))
    for i in range(0, 5):
        print("Sentence: {}".format(test_tweets[disaster_type][i]))
        print("Sentence: {}".format(test_tweets_vec[disaster_type][i]))
        print("Sentence: {}".format(test_tweets_window[disaster_type][i]))
        print("Label_1: {}".format(test_labels_1[disaster_type][i]))
        print("Label_2: {}".format(test_labels_2[disaster_type][i]))
        print("POS tags: {}".format(test_pos[disaster_type][i]))
        print("IPA: {}".format(test_ipa[disaster_type][i]))
        print("Phono: {}".format(test_phono[disaster_type][i][0]))
        print("\n\n")


pickle_list = [train_tweets,
               train_tweets_vec,
               train_tweets_window,
               train_labels_1,
               train_labels_2,
               train_pos,
               train_ipa,
               train_phono,
               [],
               val_tweets,
               val_tweets_vec,
               val_tweets_window,
               val_labels_1,
               val_labels_2,
               val_pos,
               val_ipa,
               val_phono,
               [],
               test_tweets,
               test_tweets_vec,
               test_tweets_window,
               test_labels_1,
               test_labels_2,
               test_pos,
               test_ipa,
               test_phono,
               []]

with open('Processed_Data/Processed_Data.pkl', 'wb') as fp:
    pickle.dump(pickle_list, fp)
