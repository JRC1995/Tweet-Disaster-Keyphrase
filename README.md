# Tweet-Disaster-Keyphrase

Official repository of "On Identifying Hashtags in Disaster Twitter Data" (AAAI 2020)

### Credits:

[ark_tweet](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/tree/master/ark_tweet) contains the codes for Tweet tokenization and POS-taging. It comes from here: 
http://www.cs.cmu.edu/~ark/TweetNLP/
https://code.google.com/archive/p/ark-tweet-nlp/downloads

[CMUTweetTagger.py](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/ark_tweet/CMUTweetTagger.py) is a modified version of a python wrapper for ark-tweet-nlp from:  
https://github.com/ianozsvald/ark-tweet-nlp-python

### Requirements:
---

* Tensorflow v1.12 
  >should work in higher versions, but may require changing a few lines for compatibility with Tensorflow 2.0
* [Flair](https://github.com/zalandoresearch/flair) 
  >We only used it for loading GloVe Twitter Embeddings in preprocess_vocab.py. It is possible to remove this dependency by changing this file to load the word embeddings from other sources.
* [Epitran](https://github.com/dmort27/epitran) for IPA features
  >https://pypi.org/project/epitran/
* [Panphon](https://github.com/dmort27/panphon)  for phonological features
  >https://pypi.org/project/panphon/
* [Tensorflow-Hub v0.1.1](https://www.tensorflow.org/hub/)
  >We used the [Hub module](https://tfhub.dev/google/elmo/3) for [ELMo Embeddings](https://allennlp.org/elmo).
* [NLTK](https://www.nltk.org/) for lemmatization
 
### Disaster Tweets:

Following the Twitter licenses we can publicly only share the Tweet ids. The train, test, and validation data are provided as [disaster_tweet_id_train.json](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/Data/disaster_tweet_id_train.json), [disaster_tweet_id_test.json](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/Data/disaster_tweet_id_test.json), and [disaster_tweet_id_dev.json](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/Data/disaster_tweet_id_dev.json) respectively. The data is in the form of new-line separated json where each json object is of the format:

`{'tweet_id' 134333535, 'source': 'ABCD Disaster (from XYZ dataset)'}`

The source value contains enough information about which dataset it originally belonged to and which disaster it is about.  
  
Two notable exceptions are Joplin Tornado and Sandy Hurricane datasets from:  

*[Muhammad Imran, Shady Elbassuoni, Carlos Castillo, Fernando Diaz, and Patrick Meier. Extracting Information Nuggets from Disaster-Related Messages in Social Media.In Proceedings of the 10th International Conference on Information Systems for Crisis Response and Management (ISCRAM), May 2013, Baden-Baden, Germany](https://mimran.me/papers/imran_shady_carlos_fernando_patrick_iscram2013.pdf)*  

*[Muhammad Imran, Shady Elbassuoni, Carlos Castillo, Fernando Diaz, and Patrick Meier. Practical Extraction of Disaster-Relevant Information from Social Media. In Proceedings of the 22nd international conference on World Wide Web companion, May 2013, Rio de Janeiro, Brazil](https://mimran.me/papers/imran_shady_carlos_fernando_patrick_practical_2013.pdf)*. 

I did not find the tweet ids in those datasets so I instead used their unit_id as the value for the tweet_id in the JSON files. However, these datasets are publicly available (see [resource # 2 and resource # 3](https://crisisnlp.qcri.org/)) and it is possible to retrieve the original tweets by the unit_id. In the corresponding source values I provided enough information about which file to retrieve from. 

One can also just use the whole dataset from [resource # 2 and resource # 3](https://crisisnlp.qcri.org/) and ignore any tweet_id values with length < 11 from our JSON datasets; the pre-processing code will filter them appropriately either way.

Checking the length is also the way to identify these anomalous entries in general for any need. 

### Disaster Lexicon:

The disaster lexicon used in the paper is provided [here](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/Data/Lexicons/Lexicon.txt).

[Lexicon.pkl](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/Data/Lexicons/Lexicon.pkl) is the processed version of the Lexicon (generated using [process_lexicon.py](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/Data/Lexicons/process_lexicon.py)), and it is used by other components of the software. 

### Pre-processing:

Pre-processing scripts should be done in the following order:

1. [process_data.py](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/process_data.py) for initial pre-processing (pos-tagging, annotating based on lexicon, preparing labels etc.)
2. [preprocess_vocab.py](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/preprocess_vocab.py) to create vocabularity and prepare embeddings after step 1 is done.
3. [preprocess_IPA.py](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/preprocess_IPA.py) to create word to IPA and phonological feature maps after step 1 is done.
4. [vectorize_data.py](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/vectorize_data.py) to prepare the final vectorized data for training/testing/validating after steps 1-3 are done. 

Note, however, process_data.py expects JSON files with full tweets of the following format:

`{'tweet_id' 134333535, 'tweet': 'this is a tweet', 'source': 'ABCD Disaster (from XYZ dataset)'}`

Nevertheless, the bulk of the work in [process_data.py](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/process_data.py) is done in the imported (from Utils) preprocess function which only uses a list of tweets as its main argument. So any format of data can be used if one is able to prepare a list of raw tweets to feed into the preprocess function within [process_data.py](https://github.com/JRC1995/Tweet-Disaster-Keyphrase/blob/master/process_data.py).

### Training and Testing:

Inside Models folder, go into any folder corresponding to any of the LSTM-based model you want. Use train_disaster.py to train that corresponding model, and test_disaster.py to evaluate the same. 


### Cite

(Will be updated)

