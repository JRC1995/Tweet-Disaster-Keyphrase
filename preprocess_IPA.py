import pickle
import epitran
import panphon
import csv
import numpy as np

# %%
epi = epitran.Epitran('eng-Latn')
ft = panphon.FeatureTable()
max_char_len = 20

# %%

phonological_features = ['syl', 'son', 'cons', 'cont', 'delrel',
                         'lat', 'nas', 'strid', 'voi', 'sg', 'cg',
                         'ant', 'cor', 'distr', 'lab', 'hi', 'lo',
                         'back', 'round', 'velaric', 'tense', 'long']

# %%
ipa_vocab2idx = {}
with open('Data/ipa.csv', newline='') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        new_ipa = row['IPA']
        if new_ipa not in ipa_vocab2idx:
            ipa_vocab2idx[new_ipa] = len(ipa_vocab2idx)+1


def ipa2vec(ipa):
    ipa_idx = ipa_vocab2idx.get(ipa, 0)
    # ipa_vocab_len = len(ipa_vocab2idx)+1  # plus one for the default 0 not in dict
    #one_hot = np.zeros((ipa_vocab_len,), np.float32)
    #one_hot[ipa_idx] = 1
    # return one_hot
    return ipa_idx


def ipa_word2vec(ipa_word):

    pad = 0  # np.zeros((len(ipa_vocab2idx)+1,), np.float32)

    list_vec = [ipa2vec(ipa) for ipa in ipa_word]

    while len(list_vec) < max_char_len:
        list_vec.append(pad)

    if len(list_vec) > max_char_len:
        list_vec = list_vec[0:max_char_len]

    vec = np.asarray(list_vec, np.int32)
    return vec


def word2ipa_phono_vec(word):

    ipa_word = epi.trans_list(word)

    ipa_vec = ipa_word2vec(ipa_word)
    phono_vec = ft.word_array(
        phonological_features, ''.join(ipa_word)).tolist()

    pad = np.zeros((len(phonological_features),), np.float32)
    while len(phono_vec) < max_char_len:
        phono_vec.append(pad)
    if len(phono_vec) > max_char_len:
        phono_vec = phono_vec[0:max_char_len]
    phono_vec = np.asarray(phono_vec, np.float32)

    if phono_vec.shape[0] != ipa_vec.shape[0]:
        print(phono_vec.shape)
        print(ipa_vec.shape)
        print(ipa_word)
        print(word)
        print(len(word))
    return ipa_vec, phono_vec  # np.concatenate((ipa_vec, phono_vec), axis=-1)


# %%


with open('Processed_Data/vocab.pkl', 'rb') as fp:
    counter = pickle.load(fp)

word2ipa_vec = {}
word2phono_vec = {}

i = 0
for word in counter:
    word2ipa_vec[word], word2phono_vec[word] = word2ipa_phono_vec(word)
    i += 1
    if i % 1000 == 0:
        print("{} words processed....".format(i))


print("SOME SAMPLES")

i = 0
for word in word2ipa_vec:
    i += 1
    print(word)
    print(word2ipa_vec[word])
    print(word2phono_vec[word])
    print(word2phono_vec[word].shape)
    if i == 10:
        break

# print(len(phonological_features)+1+len(ipa_vocab2idx))
list = [word2ipa_vec, word2phono_vec, len(phonological_features), 1+len(ipa_vocab2idx)]
with open('Processed_Data/word_to_ipa_vec.pkl', 'wb') as fp:
    pickle.dump(list, fp)
