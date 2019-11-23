import pickle
import numpy as np
import math
from flair.embeddings import WordEmbeddings
from flair.data import Sentence

glove_embeddings = WordEmbeddings('twitter')

with open('Processed_Data/vocab.pkl', 'rb') as fp:
    counter = pickle.load(fp)

print("Initial vocab len: {}".format(len(counter)))

vocab2idx = {}
special_tags = ['<sos>', '<eos>', '<unk>', '<pad>', 'num']

for word in special_tags:
    vocab2idx[word] = len(vocab2idx)

min_count = 1
for word in counter:
    if counter[word] >= min_count:
        if word not in vocab2idx:
            vocab2idx[word] = len(vocab2idx)


print("Trimmed vocab len: {}".format(len(vocab2idx)))


def embed(word):
    word_vec_dim = 100
    if word == "<pad>":
        return np.zeros((word_vec_dim,), dtype=np.float32)
    sentenced_word = Sentence(word)
    glove_embeddings.embed(sentenced_word)
    for token in sentenced_word:
        embd = np.asarray(token.embedding.cpu().numpy(), np.float32)
        break
    pad = np.zeros((word_vec_dim,), dtype=np.float32)
    if np.all(np.equal(embd, pad)):
        return np.random.uniform(-math.sqrt(3/word_vec_dim), +math.sqrt(3/word_vec_dim),
                                 (word_vec_dim,))
    else:
        return embd


embd = []
crisis_embd_1 = []
crisis_embd_2 = []
i = 0
for word in vocab2idx:
    embd.append(embed(word))
    i += 1
    # print(i)
    if i % 1000 == 0:
        print("{} words embeded...".format(i))

print("Sample Embeddings:")
for i in range(5, 10):
    print(embd[i])

list = [vocab2idx, embd]
with open('Processed_Data/vocab_and_embd.pkl', 'wb') as fp:
    pickle.dump(list, fp)
