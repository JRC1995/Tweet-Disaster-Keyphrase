import numpy as np
import string
import pickle
import math
from wordsegment import load, segment
import emoji
import ark_tweet.CMUTweetTagger as ct
from nltk.stem import WordNetLemmatizer
from langdetect import detect
import random
import re

# %%
load()  # loads word segment
wordnet_lemmatizer = WordNetLemmatizer()
max_char_len = 20

special_tags = ['<unk>', '<pad>', 'num']

labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
idx2labels = {v: k for k, v in labels2idx.items()}

# %%
with open('Data/Lexicons/Lexicon.pkl', 'rb') as fp:
    disaster_word = pickle.load(fp)

# %%


def lemmatize(word):
    word = word.strip("").strip(" ").strip("\t").strip("\n").strip("\b").strip("\n\n")
    if word in disaster_word:
        return [word]
    else:
        word = word.split("'s")[0]
        words = re.split('-|/', word)
        l4 = []
        for word in words:
            if word == "sos":
                l4.append(word)
            elif word == "goods":
                l4.append(word)
            elif word == "lives":
                l4.append(word)
            elif word == "stuck":
                l4.append(word)
            elif word in string.punctuation:
                l4.append("<NOT_IN_LIST>")
            else:
                l1 = wordnet_lemmatizer.lemmatize(word, 'v')
                l2 = wordnet_lemmatizer.lemmatize(l1, 'a')
                l3 = wordnet_lemmatizer.lemmatize(l2, 'r')
                l4.append(wordnet_lemmatizer.lemmatize(l3, 'n'))

        return l4

# %%


def quality_check(word):
    # add conditions here to filter words
    translator = str.maketrans('', '', string.punctuation.replace(
        '#', '').replace('&', '').replace("'", '').replace("-", ''))
    regnumber = re.compile(r'\d+[,\.]*\d*')
    if 'https://' in word or 'http://' in word or 'www.' in word:
        return False  # "<url>"
    elif '@' in word and word != '@':
        return False  # "<user>"
    elif word in emoji.UNICODE_EMOJI or word in ['"', '', " ", ' ', '#', 'rt', "-", "'"]:
        return False
    elif regnumber.match(word):
        return "num"
    elif word[-1] == '#' or word[-1] == "'":
        word = ''.join(word[0:-1].lower().split(' ')).translate(translator)
        if word in ['', " ", ' ']:
            return False
        else:
            return word
    else:
        word = ''.join(word.lower().split(' ')).translate(translator)
        if word in ['', " ", ' ']:
            return False
        else:
            return word


# %%

def segment_adv(word):
    if word in ["hagupit", "awaran", "rubyph",
                "bhopa", "marinph", "bopha",
                "rescueph", "pabloph", "reliefph",
                "kkf", "floodph", "yolandaph",
                "damayan", "yovanielrd", "bantanyan",
                "midufinga", "jordsklv"]:
        segmented_words = [word]
    elif word == "typhoonhagupit":
        segmented_words = ["typhoon", "hagupit"]
    else:
        segmented_words = segment(word)
    return segmented_words

# %%


def preprocess(tweets_tagged, counter={}):
    # input: list of tweets (in string format) (Example: ["x y z", "a bc",......]])

    lex_phrases = 0
    lex_tweets = 0
    hashtags = 0
    hashtag_tweets = 0
    hashtag_counter = {}
    keyphrase_counter = {}
    total_tweets_per_data = {}
    skipped_count = 0
    keyphrases = 0
    keyphrase_tweets = 0
    tweet2idx = {}
    i = 0

    tweets = []
    info_labels = []
    processed_tweets = []
    hashtags_this_tweets = []
    lex_phrases_this_tweets = []
    keyphrases_this_tweets = []
    labels = []
    pos_tags = []

    for tweet_tagged in tweets_tagged:

        hashtags_this_tweet = 0
        lex_phrases_this_tweet = 0
        keyphrases_this_tweet = 0

        tokenized_tweet = [word[0] for word in tweet_tagged]

        filter_tweet = list(map(quality_check, tokenized_tweet))
        filter_tweet = [word for word in filter_tweet if word]

        filter_tweet_ = []

        for word in filter_tweet:

            if len(word) > 1:

                if '&' in word and len(word) > 1:

                    words_ = word.split('&')
                    for c, word_ in enumerate(words_):
                        if word_ not in ['', " "]:
                            # print(word_)
                            filter_tweet_.append(word_)
                        if c != len(words_)-1:
                            filter_tweet_.append("&")

                    # print(filter_tweet_)
                elif word not in ['', " "]:
                    filter_tweet_.append(word)

            elif word not in ['', " "]:
                filter_tweet_.append(word)

        filter_tweet = filter_tweet_

        if len(filter_tweet) >= 5 and len(filter_tweet) <= 200:

            tweet = ' '.join(filter_tweet)
            flag = 0

            try:
                if detect(tweet) == 'en':
                    flag = 1
            except:
                flag = 0

            if flag == 1 and tweet not in tweet2idx:

                ignore_flag = 0

                tweet2idx[tweet] = len(tweet2idx)

                processed_tweet = []
                label = []

                c = 0
                for word in filter_tweet:

                    if word[0] == '#' and ignore_flag == 0:

                        if word not in hashtag_counter:
                            hashtag_counter[word] = 1
                        else:
                            hashtag_counter[word] += 1

                        hashtags_this_tweet += 1

                        word_without_hash = word[1:]

                        segmented_words = segment_adv(word_without_hash)

                        if len(segmented_words) == 0:
                            ignore_flag = 1

                        if ignore_flag == 0:

                            p = 0

                            if c >= 1:
                                if label[c-1] == labels2idx['B'] or label[c-1] == labels2idx['I']:
                                    p = 1

                            for p2, word in enumerate(segmented_words):

                                processed_tweet.append(word)

                                if word not in counter:
                                    counter[word] = 1
                                else:
                                    counter[word] += 1

                                if len(segmented_words) == 1 and p == 0:

                                    label_flag = 0

                                    words_c = lemmatize(word)
                                    for word_c in words_c:
                                        if word_c in disaster_word and label_flag == 0:
                                            disaster_vocab_second_list = disaster_word[word_c]

                                            if c+1 >= len(filter_tweet):
                                                if '<NULL>' in disaster_vocab_second_list:
                                                    label.append(labels2idx['S'])
                                                    lex_phrases_this_tweet += 1
                                                    label_flag = 1

                                            elif c+1 < len(filter_tweet):

                                                if '#' in filter_tweet[c+1]:
                                                    word_c1 = filter_tweet[c+1][1:]
                                                    segmented_words_c1 = segment_adv(
                                                        word_c1)
                                                    if len(segmented_words_c1) == 0:
                                                        words_c1 = [""]
                                                        ignore_flag = 1
                                                    else:
                                                        words_c1 = lemmatize(
                                                            segmented_words_c1[0])
                                                else:
                                                    words_c1 = lemmatize(filter_tweet[c+1])

                                                for word_c1 in words_c1:

                                                    if label_flag == 0:

                                                        if word_c1 in disaster_vocab_second_list:
                                                            label.append(labels2idx['B'])
                                                            label_flag = 1
                                                            lex_phrases_this_tweet += 1

                                                        elif '<NULL>' in disaster_vocab_second_list:
                                                            # print("unigram match decided")
                                                            label.append(labels2idx['S'])
                                                            label_flag = 1
                                                            lex_phrases_this_tweet += 1

                                    if label_flag == 0:
                                        label.append(labels2idx['S'])
                                    # label.append(labels2idx['S'])
                                    keyphrases_this_tweet += 1
                                else:
                                    if p == 0 and p2 == 0:
                                        label.append(labels2idx['B'])
                                        keyphrases_this_tweet += 1

                                    elif p2 == len(segmented_words)-1:

                                        label_flag = 0
                                        words_c = lemmatize(word)

                                        for word_c in words_c:

                                            if word_c in disaster_word and label_flag == 0:
                                                disaster_vocab_second_list = disaster_word[word_c]
                                                if c+1 >= len(filter_tweet):
                                                    if '<NULL>' in disaster_vocab_second_list:
                                                        label.append(labels2idx['E'])
                                                        label_flag = 1

                                                elif c+1 < len(filter_tweet):

                                                    if '#' in filter_tweet[c+1]:
                                                        word_c1 = filter_tweet[c+1][1:]
                                                        segmented_words_c1 = segment_adv(
                                                            word_c1)
                                                        if len(segmented_words_c1) == 0:
                                                            words_c1 = [""]
                                                            ignore_flag = 1
                                                        else:
                                                            words_c1 = lemmatize(
                                                                segmented_words_c1[0])
                                                    else:
                                                        words_c1 = lemmatize(filter_tweet[c+1])

                                                    for word_c1 in words_c1:

                                                        if label_flag == 0:

                                                            if word_c1 in disaster_vocab_second_list:
                                                                label.append(labels2idx['I'])
                                                                label_flag = 1

                                                            elif '<NULL>' in disaster_vocab_second_list:
                                                                label.append(labels2idx['E'])
                                                                label_flag = 1

                                        if label_flag == 0:
                                            label.append(labels2idx['E'])

                                    else:
                                        label.append(labels2idx['I'])

                                c += 1
                                p += 1

                    elif ignore_flag == 0:

                        processed_tweet.append(word)

                        if word not in counter:
                            counter[word] = 1
                        else:
                            counter[word] += 1

                        label_flag = 0

                        if c >= 1:
                            if label[c-1] == labels2idx['B'] or label[c-1] == labels2idx['I']:

                                words_c = lemmatize(word)

                                for word_c in words_c:

                                    if word_c in disaster_word and label_flag == 0:

                                        disaster_vocab_second_list = disaster_word[word_c]

                                        if c+1 >= len(filter_tweet):
                                            if '<NULL>' in disaster_vocab_second_list:
                                                label.append(labels2idx['E'])
                                                label_flag = 1

                                        elif c+1 < len(filter_tweet):
                                            if '#' in filter_tweet[c+1]:
                                                word_c1 = filter_tweet[c+1][1:]
                                                segmented_words_c1 = segment_adv(word_c1)
                                                if len(segmented_words_c1) == 0:
                                                    words_c1 = [""]
                                                    ignore_flag = 1
                                                else:
                                                    words_c1 = lemmatize(segmented_words_c1[0])
                                            else:
                                                words_c1 = lemmatize(filter_tweet[c+1])

                                            for word_c1 in words_c1:

                                                if label_flag == 0:
                                                    if word_c1 in disaster_vocab_second_list:
                                                        label.append(labels2idx['I'])
                                                        label_flag = 1

                                                    elif '<NULL>' in disaster_vocab_second_list:
                                                        label.append(labels2idx['E'])
                                                        label_flag = 1

                                if label_flag == 0:
                                    label.append(labels2idx['E'])
                                    label_flag = 1

                        if label_flag == 0:

                            words_c = lemmatize(word)

                            for word_c in words_c:

                                if word_c in disaster_word and label_flag == 0:

                                    disaster_vocab_second_list = disaster_word[word_c]

                                    if c+1 >= len(filter_tweet):

                                        if '<NULL>' in disaster_vocab_second_list:
                                            label.append(labels2idx['S'])
                                            label_flag = 1
                                            lex_phrases_this_tweet += 1
                                            keyphrases_this_tweet += 1

                                    elif c+1 < len(filter_tweet):

                                        if '#' in filter_tweet[c+1]:
                                            word_c1 = filter_tweet[c+1][1:]
                                            segmented_words_c1 = segment_adv(word_c1)
                                            if len(segmented_words_c1) == 0:
                                                words_c1 = [""]
                                                ignore_flag = 1
                                            else:
                                                words_c1 = lemmatize(segmented_words_c1[0])
                                        else:
                                            words_c1 = lemmatize(filter_tweet[c+1])

                                        for word_c1 in words_c1:

                                            if label_flag == 0:
                                                if word_c1 in disaster_vocab_second_list:
                                                    label.append(labels2idx['B'])
                                                    label_flag = 1
                                                    lex_phrases_this_tweet += 1
                                                    keyphrases_this_tweet += 1

                                                elif '<NULL>' in disaster_vocab_second_list:
                                                    label.append(labels2idx['S'])
                                                    label_flag = 1
                                                    lex_phrases_this_tweet += 1
                                                    keyphrases_this_tweet += 1

                        if label_flag == 0:
                            label.append(labels2idx['O'])

                        c += 1

                if ignore_flag == 0:

                    if i % 1000 == 0:

                        print("{}th Tweet Example: ".format(i))
                        print("Original tweet: {}".format(tweet))
                        print("Processed tweet: ", end="")

                        phrase = ""
                        for word, tag in zip(processed_tweet, label):
                            if tag == labels2idx['S']:
                                print("["+word+"] ", end="")
                                phrase = word
                                if phrase in keyphrase_counter:
                                    keyphrase_counter[phrase] += 1
                                else:
                                    keyphrase_counter[phrase] = 1
                            elif tag in [labels2idx['O'], labels2idx['I']]:
                                print(word+" ", end="")
                                phrase += word + " "
                            elif tag == labels2idx['B']:
                                print("["+word+" ", end="")
                                phrase += word+" "
                            elif tag == labels2idx['E']:
                                print(word+"] ", end="")
                                phrase += word
                                if phrase in keyphrase_counter:
                                    keyphrase_counter[phrase] += 1
                                else:
                                    keyphrase_counter[phrase] = 1
                        print("\n\n")

                    i += 1

                    if hashtags_this_tweet != 0:
                        hashtag_tweets += 1
                    if lex_phrases_this_tweet != 0:
                        lex_tweets += 1
                    if keyphrases_this_tweet != 0:
                        keyphrase_tweets += 1

                    lex_phrases += lex_phrases_this_tweet
                    hashtags += hashtags_this_tweet
                    keyphrases += keyphrases_this_tweet

                    processed_tweets.append(processed_tweet)
                    labels.append(label)
                    hashtags_this_tweets.append(hashtags_this_tweet)
                    lex_phrases_this_tweets.append(lex_phrases_this_tweet)
                    keyphrases_this_tweets.append(keyphrases_this_tweet)

    return processed_tweets, labels, counter

    string_tweets = [" ".join(processed_tweet) for processed_tweet in processed_tweets]
    tweets_tagged = ct.runtagger_parse(string_tweets)


def pos_tag(processed_tweets, labels, tweets_tagged):

    new_processed_tweets = []
    new_labels = []
    new_POS_tags = []

    for processed_tweet, label, tweet_tagged in zip(processed_tweets, labels,
                                                    tweets_tagged):
        POS_tag = [word[1] for word in tweet_tagged]

        if len(POS_tag) != len(processed_tweet):

            print("\n\n POS ANOMALY DETECTED. SKIPPING SAMPLE. \n\n")
            print(processed_tweet)
            print(POS_tag)
            print("\n\n")

            """

            total_tweets_per_data[filename] -= 1

            hashtags -= hashtags_this_tweet
            lex_phrases -= lex_phrases_this_tweet
            keyphrases -= keyphrases_this_tweet

            if hashtags_this_tweet != 0:
                hashtag_tweets -= 1
            if lex_phrases_this_tweet != 0:
                lex_tweets -= 1
            if keyphrases_this_tweet != 0:
                keyphrase_tweets -= 1

            skipped_count += 1
            """

        else:
            new_processed_tweets.append(processed_tweet)
            new_labels.append(label)
            new_POS_tags.append(POS_tag)

    return new_processed_tweets, new_labels, new_POS_tags
