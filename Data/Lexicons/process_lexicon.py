from nltk.stem import WordNetLemmatizer
import pickle

wordnet_lemmatizer = WordNetLemmatizer()

disaster_vocab = []

filename = "CrisisLexRec.txt"
with open(filename) as infile:
    for line in infile:
        phrase = line.strip('\n').strip(' ').strip('').strip('\t').lower()
        if phrase != "":
            disaster_vocab.append(phrase)


temp_disaster_vocab = disaster_vocab

filename = "Lexicon.txt"
with open(filename) as infile:
    for line in infile:
        phrase = line.strip('\n').strip(' ').strip('').strip('\t').lower()
        if phrase != "":
            disaster_vocab.append(phrase)

disaster_vocab = set(disaster_vocab)
print(len(disaster_vocab))
first_word_disaster = []
second_word_disaster = []


def lemmatize(word):
    word = word.strip("").strip(" ").strip("\t").strip("\n").strip("\b")
    word = word.split("'s")[0]
    if word == "sos":
        return word
    elif word == "goods":
        return word
    elif word == "lives":
        return word
    elif word == "stuck":
        return word
    l1 = wordnet_lemmatizer.lemmatize(word, 'v')
    l2 = wordnet_lemmatizer.lemmatize(l1, 'a')
    l3 = wordnet_lemmatizer.lemmatize(l2, 'r')
    l4 = wordnet_lemmatizer.lemmatize(l3, 'n')
    if l4 == 'm':
        return "<NULL>"
    else:
        return l4


disaster_word = {}
for bigram in disaster_vocab:
    bigram = bigram.strip('').strip(' ').strip('\t').strip("\n").strip("\b").split(' ')
    if lemmatize(bigram[0]) not in disaster_word and lemmatize(bigram[0]) != '':
        disaster_word[lemmatize(bigram[0])] = {}
    if len(bigram) > 1:
        if lemmatize(bigram[1]) not in disaster_word[lemmatize(bigram[0])]:
            disaster_word[lemmatize(bigram[0])][lemmatize(bigram[1])] = lemmatize(bigram[1])
    elif '<NULL>' not in disaster_word[lemmatize(bigram[0])]:
        disaster_word[lemmatize(bigram[0])]["<NULL>"] = "<NULL>"


with open('Lexicon.pkl', 'wb') as fp:
    pickle.dump(disaster_word, fp)
