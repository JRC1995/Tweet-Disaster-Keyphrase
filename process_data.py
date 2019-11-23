import json
from Utils.preprocess import preprocess as utils_preprocess
from Utils.preprocess import pos_tag as utils_pos_tag
import ark_tweet.CMUTweetTagger as ct
import pickle


def extract(filename):
    tweets = []
    sources = []
    with open(filename) as file:
        for json_data in file:
            obj = json.loads(json_data)
            tweets.append(obj["tweet"])
            sources.append(obj["source"])
    return tweets, sources


def preprocess(tweets, counter):

    tweets_tagged = ct.runtagger_parse(tweets)

    processed_tweets, labels, counter = utils_preprocess(tweets_tagged, counter)

    string_tweets = [" ".join(processed_tweet) for processed_tweet in processed_tweets]
    tweets_tagged = ct.runtagger_parse(string_tweets)

    tweets, labels, pos_tags = utils_pos_tag(processed_tweets, labels, tweets_tagged)

    return tweets, labels, pos_tags, counter


source_map = {"(Ours) California Fire": "California Fire",
              "(Ours) Maria Hurricane": "Maria Hurricane",
              "2012_Philipinnes_floods-tweets_labeled.csv (CrisisLexT26-v1.0)": "Philipinnes Flood"}


train_tweets, train_sources = extract("Data/disaster_tweet_train.json")
val_tweets, val_sources = extract("Data/disaster_tweet_dev.json")
test_tweets, test_sources = extract("Data/disaster_tweet_test.json")

test_tweets_dict = {"General": [], "California Fire": [],
                    "Maria Hurricane": [], "Philipinnes Flood": []}

for test_tweet, test_source in zip(test_tweets, test_sources):
    source = source_map.get(test_source, "General")
    test_tweets_dict[source].append(test_tweet)

counter = {}

# train_tweets = train_tweets[0:1000]

train_tweets, train_labels, train_pos, counter = preprocess(train_tweets, counter)
val_tweets, val_labels, val_pos, counter = preprocess(val_tweets, counter)

test_tweets_ = {}
test_labels = {}
test_pos = {}

for disaster in test_tweets_dict:
    tweets, labels, pos, counter = preprocess(test_tweets_dict[disaster], counter)
    test_tweets_[disaster] = tweets
    test_labels[disaster] = labels
    test_pos[disaster] = pos

test_tweets = test_tweets_

with open('Processed_Data/vocab.pkl', 'wb') as fp:
    pickle.dump(counter, fp)

pickle_list = [train_tweets, train_labels, train_pos,
               val_tweets, val_labels, val_pos,
               test_tweets, test_labels, test_pos]


with open('Processed_Data/Intermediate_Data.pkl', 'wb') as fp:
    pickle.dump(pickle_list, fp)

print("Training Size: ", len(train_tweets))
print("Testing Size: ", sum([len(test_tweets[disaster]) for disaster in test_tweets]))
print("Validation Size: ", len(val_tweets))
