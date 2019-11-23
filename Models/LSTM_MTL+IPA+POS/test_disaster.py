# %%

from models.JRNN import Joint_Encoders
import pickle
import numpy as np
import configparser
import tensorflow as tf
from bucket_and_batch import bucket_and_batch
from eval import eval_NER_exact_entity_match

# %%
cp = configparser.ConfigParser()
bucket_and_batch = bucket_and_batch()

# %%


def str2bool(text):
    if text in ['True']:
        return True
    else:
        return False


# %%
cp.read('configs.ini')

window_size = int(cp['HYPERPARAMETERS']['window_size'])
batch_size = int(cp['HYPERPARAMETERS']['batch_size'])
hidden_size = int(cp['HYPERPARAMETERS']['hidden_size'])
learning_rate = float(cp['HYPERPARAMETERS']['learning_rate'])
momentum = float(cp['HYPERPARAMETERS']['momentum'])
decay = float(cp['HYPERPARAMETERS']['decay'])
fine_tune_lr = float(cp['HYPERPARAMETERS']['fine_tune_lr'])
fine_tune = str2bool(cp['HYPERPARAMETERS']['fine_tune'])
gross_tune = str2bool(cp['HYPERPARAMETERS']['gross_tune'])
epochs = int(cp['HYPERPARAMETERS']['epochs'])
l2 = float(cp['HYPERPARAMETERS']['l2'])
dropout = float(cp['HYPERPARAMETERS']['dropout'])
freeze_epochs = int(cp['HYPERPARAMETERS']['freeze_epochs'])
optimizer_to_use = str(cp['HYPERPARAMETERS']['optimizer_to_use'])
variational_dropout = str2bool(cp['HYPERPARAMETERS']['variational_dropout'])
embedding_dropout = str2bool(cp['HYPERPARAMETERS']['embedding_dropout'])
alpha = float(cp['HYPERPARAMETERS']['alpha'])
grad_max_norm = float(cp['HYPERPARAMETERS']['grad_max_norm'])
model_name = str(cp['HYPERPARAMETERS']['model'])


with open('../../Processed_Data/vocab_and_embd.pkl', 'rb') as fp:
    data = pickle.load(fp)

vocab2idx = data[0]
embd = data[1]


labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
idx2labels = {v: k for k, v in labels2idx.items()}

labels_set = list(labels2idx.keys())
elmo_dims = 1024

embeddings = np.asarray(embd, np.float32)

with open('../../Processed_Data/Processed_Data.pkl', 'rb') as fp:
    data = pickle.load(fp)

train_tweets = data[0]
train_tweets_vec = data[1]
train_tweets_window = data[2]
train_labels_1 = data[3]
train_labels_2 = data[4]
train_pos = data[5]
train_ipa = data[6]
train_phono = data[7]
train_disaster_flags = data[8]
val_tweets = data[9]
val_tweets_vec = data[10]
val_tweets_window = data[11]
val_labels_1 = data[12]
val_labels_2 = data[13]
val_pos = data[14]
val_ipa = data[15]
val_phono = data[16]
val_disaster_flags = data[17]
test_tweets = data[18]
test_tweets_vec = data[19]
test_tweets_window = data[20]
test_labels_1 = data[21]
test_labels_2 = data[22]
test_pos = data[23]
test_ipa = data[24]
test_phono = data[25]
test_disaster_flags = data[26]
max_char_len = 20

phonological_features = ['syl', 'son', 'cons', 'cont', 'delrel',
                         'lat', 'nas', 'strid', 'voi', 'sg', 'cg',
                         'ant', 'cor', 'distr', 'lab', 'hi', 'lo',
                         'back', 'round', 'velaric', 'tense', 'long']
phono_dim = len(phonological_features)


with open('SAVED_MODEL/test_hyperparameters', 'rb') as fp:
    hyperparameters = pickle.load(fp)

window_size = hyperparameters[0]
hidden_size = hyperparameters[1]
learning_rate = hyperparameters[2]
momentum = hyperparameters[3]
decay = hyperparameters[4]
fine_tune_lr = hyperparameters[5]
fine_tune = hyperparameters[6]
gross_tune = hyperparameters[7]
l2 = hyperparameters[8]
dropout = hyperparameters[9]
optimizer_to_use = hyperparameters[10]
variational_dropout = hyperparameters[11]
embedding_dropout = hyperparameters[12]
alpha = hyperparameters[13]
grad_max_norm = hyperparameters[14]
model_name = hyperparameters[15]
variational_dropout = hyperparameters[16]

tf_sentences = tf.placeholder(tf.int32, [None, None, 3])
#tf_sentences_string = tf.placeholder(tf.string, [None, None])
tf_pos = tf.placeholder(tf.int32, [None, None])
tf_ipa = tf.placeholder(tf.int32, [None, None, max_char_len])
tf_phono = tf.placeholder(tf.float32, [None, None, max_char_len, phono_dim])
tf_labels_1 = tf.placeholder(tf.int32, [None, None])
tf_labels_2 = tf.placeholder(tf.int32, [None, None])
tf_true_seq_lens = tf.placeholder(tf.int32, [None])
tf_train = tf.placeholder(tf.bool)
tf_learning_rate = tf.placeholder(tf.float32)
tf_fine_tune_lr = tf.placeholder(tf.float32)

model = Joint_Encoders(sentences=tf_sentences,
                       pos=tf_pos,
                       ipa=tf_ipa,
                       phono=tf_phono,
                       elmo_dims=elmo_dims,
                       labels_1=tf_labels_1,
                       labels_2=tf_labels_2,
                       true_seq_lens=tf_true_seq_lens,
                       train=tf_train,
                       learning_rate=tf_learning_rate,
                       fine_tune_lr=fine_tune_lr,
                       word_embeddings=embeddings,
                       tags_set=labels_set,
                       model=model_name,
                       alpha=alpha,
                       hidden_size=hidden_size,
                       dropout=dropout,
                       l2=l2,
                       momentum=momentum,
                       fine_tune=fine_tune,
                       gross_tune=gross_tune,
                       optimizer_to_use=optimizer_to_use,
                       embedding_dropout=embedding_dropout,
                       grad_max_norm=grad_max_norm,
                       variational_dropout=variational_dropout)

eval = eval_NER_exact_entity_match()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

for disaster_type in test_tweets:

    name = disaster_type

    with tf.Session(config=config) as sess:  # Start Tensorflow Session

        saver = tf.train.Saver()

        print('Loading pre-trained weights for the model...')

        saver.restore(sess, 'SAVED_MODEL/model.ckpt')
        sess.run(tf.global_variables())

        print("\nModel Restored\n\n")

        test_batches_sen, test_batches_string, test_batches_pos, \
            test_batches_ipa, test_batches_phono, test_batches_labels_1, test_batches_labels_2,\
            test_batches_true_seq_lens = bucket_and_batch.bucket_and_batch(
                test_tweets_window[disaster_type], test_tweets[disaster_type],
                test_pos[disaster_type], test_ipa[disaster_type],
                test_phono[disaster_type], test_labels_1[disaster_type], test_labels_2[disaster_type],
                vocab2idx, 35, labels_set)

        total_actual_keys = 0
        total_guess = 0
        total_correct_guess = 0
        total_test_acc = 0
        total_test_loss = 0

        tag_predictions = []
        tag_labels = []

        f = open("disaster_output_"+str(name)+".html", "w")
        f.write('<html><body>')
        f.close()

        f2 = open("disaster_output_gold_for_wordcloud_"+str(name)+".txt", "w")
        f2.write('')
        f2.close()

        f3 = open("disaster_output_for_wordcloud_"+str(name)+".txt", "w")
        f3.write('')
        f3.close()

        f4 = open("eval_disaster_"+str(name)+".txt", "w")
        f4.write('')
        f4.close()

        print("Testing")

        for i in range(0, len(test_batches_sen)):

            print("Testing batch {}....".format(i+1))

            cost, prediction, acc, = sess.run([model.loss,
                                               model.predictions_2,
                                               model.accuracy],
                                              feed_dict={tf_sentences: test_batches_sen[i],
                                                         tf_pos: test_batches_pos[i],
                                                         tf_ipa: test_batches_ipa[i],
                                                         tf_phono: test_batches_phono[i],
                                                         tf_labels_1: test_batches_labels_1[i],
                                                         tf_labels_2: test_batches_labels_2[i],
                                                         tf_true_seq_lens: test_batches_true_seq_lens[i],
                                                         tf_train: False})

            actual_keys, guess, correct_guess = eval.tpfnfp(
                test_batches_labels_2[i],
                prediction,
                test_batches_true_seq_lens[i],
                labels_set)

            f = open("disaster_output_"+str(name)+".html", "a")
            f2 = open("disaster_output_gold_for_wordcloud_"+str(name)+".txt", "a")
            f3 = open("disaster_output_for_wordcloud_"+str(name)+".txt", "a")
            f4 = open("eval_disaster_"+str(name)+".txt", "a")

            k = 0
            for sentence in test_batches_string[i]:

                sentence_to_write = ""
                sentence_to_write_2 = ""
                sentence_to_write_3 = ""
                sentence_to_write_4 = ""
                sentence_to_write_4_pred = ""

                j = 0
                for word in sentence:
                    if j < test_batches_true_seq_lens[i][k]:
                        if word[0] in ['<'] and word[-1] in ['>']:
                            word = word[1:-1]
                        prepend = ""
                        append = ""
                        if prediction[k][j] == 1:
                            prepend += "<u>"
                            if j+1 != test_batches_true_seq_lens[i][k]:
                                if prediction[k][j+1] not in [2, 3]:
                                    append += "</u>"
                        elif prediction[k][j] == 3:
                            append += "</u>"
                            if j != 0:
                                if prediction[k][j-1] not in [1, 2]:
                                    prepend += "<u>"
                            else:
                                prepend = "<u>"
                        elif prediction[k][j] == 2:
                            if j != 0:
                                if prediction[k][j-1] not in [1, 2]:
                                    prepend += "<u>"
                            else:
                                prepend = "<u>"
                            if j+1 != test_batches_true_seq_lens[i][k]:
                                if prediction[k][j+1] not in [2, 3]:
                                    append += "</u>"
                            else:
                                append += "</u>"
                        elif prediction[k][j] == 4:
                            prepend += "<u>"
                            append += "</u>"

                        if test_batches_labels_2[i][k][j] == 1:
                            prepend += "<b>[</b>"
                            if j+1 != test_batches_true_seq_lens[i][k]:
                                if test_batches_labels_2[i][k][j+1] not in [2, 3]:
                                    append += "<b>]</b>"
                        elif test_batches_labels_2[i][k][j] == 3:
                            append += "<b>]</b>"
                            if j != 0:
                                if test_batches_labels_2[i][k][j-1] not in [1, 2]:
                                    prepend += "<b>[</b>"
                            else:
                                prepend = "<b>[</b>"
                        elif test_batches_labels_2[i][k][j] == 2:
                            if j != 0:
                                if test_batches_labels_2[i][k][j-1] not in [1, 2]:
                                    prepend += "<b>[</b>"
                            else:
                                prepend = "<b>[</b>"
                            if j+1 != test_batches_true_seq_lens[i][k]:
                                if test_batches_labels_2[i][k][j+1] not in [2, 3]:
                                    append += "<b>]</b>"
                            else:
                                append += "<b>]</b>"
                        elif test_batches_labels_2[i][k][j] == 4:
                            prepend += "<b>[</b>"
                            append += "<b>]</b>"

                        if test_batches_labels_2[i][k][j] == 1:
                            sentence_to_write_2 = sentence_to_write_2+"\n"+str(word)+" "
                        if test_batches_labels_2[i][k][j] == 2:
                            sentence_to_write_2 = sentence_to_write_2+str(word)+" "
                        elif test_batches_labels_2[i][k][j] == 3:
                            sentence_to_write_2 = sentence_to_write_2+str(word)+"\n"
                        elif test_batches_labels_2[i][k][j] == 4:
                            sentence_to_write_2 = sentence_to_write_2+"\n"+str(word)+"\n"

                        if prediction[k][j] == 1:
                            sentence_to_write_3 = sentence_to_write_3+"\n"+str(word)+" "
                        if prediction[k][j] == 2:
                            sentence_to_write_3 = sentence_to_write_3+str(word)+" "
                        elif prediction[k][j] == 3:
                            sentence_to_write_3 = sentence_to_write_3+str(word)+"\n"
                        elif prediction[k][j] == 4:
                            sentence_to_write_3 = sentence_to_write_3+"\n"+str(word)+"\n"

                        sentence_to_write = sentence_to_write+prepend+str(word)+append+" "
                        j += 1

                j = 0
                for word in sentence:
                    if j < test_batches_true_seq_lens[i][k]:
                        if test_batches_labels_2[i][k][j] == 1:
                            sentence_to_write_4 = sentence_to_write_4+str(word)+" "
                        elif test_batches_labels_2[i][k][j] == 2:
                            sentence_to_write_4 = sentence_to_write_4+str(word)+" "
                        elif test_batches_labels_2[i][k][j] == 3:
                            sentence_to_write_4 = sentence_to_write_4+str(word)+", "
                        elif test_batches_labels_2[i][k][j] == 4:
                            sentence_to_write_4 = sentence_to_write_4+str(word)+", "
                        j += 1

                # remove the last two characters space and ,
                sentence_to_write_4 = sentence_to_write_4[0:len(sentence_to_write_4)-2]
                sentence_to_write_4 = sentence_to_write_4 + " | "

                j = 0
                modified_prediction = np.copy(prediction)
                for word in sentence:
                    if j < test_batches_true_seq_lens[i][k]:

                        # Fix boundary errors

                        if prediction[k][j] == 0:
                            if j+1 < test_batches_true_seq_lens[i][k]:
                                if prediction[k][j+1] in [2, 3]:
                                    modified_prediction[k][j] = 1

                        if prediction[k][j] == 1:
                            if j+1 < test_batches_true_seq_lens[i][k]:
                                if prediction[k][j+1] in [0, 4]:
                                    modified_prediction[k][j] = 4
                            else:
                                modified_prediction[k][j] = 4

                        elif prediction[k][j] == 2:
                            if j+1 < test_batches_true_seq_lens[i][k]:
                                if prediction[k][j+1] in [0, 1]:
                                    modified_prediction[k][j] = 3
                            else:
                                modified_prediction[k][j] = 3

                        elif prediction[k][j] == 3:
                            if j+1 < test_batches_true_seq_lens[i][k]:
                                if prediction[k][j+1] in [2]:
                                    modified_prediction[k][j] = 2
                        else:
                            modified_prediction[k][j] = prediction[k][j]

                        prediction[k][j] = modified_prediction[k][j]

                        if prediction[k][j] == 1:
                            sentence_to_write_4_pred = sentence_to_write_4_pred+str(word)+" "
                        if prediction[k][j] == 2:
                            sentence_to_write_4_pred = sentence_to_write_4_pred+str(word)+" "
                        elif prediction[k][j] == 3:
                            sentence_to_write_4_pred = sentence_to_write_4_pred+str(word)+", "
                        elif prediction[k][j] == 4:
                            sentence_to_write_4_pred = sentence_to_write_4_pred+str(word)+", "
                        j += 1

                    # remove the last two characters space and ,
                sentence_to_write_4_pred = sentence_to_write_4_pred[0:len(
                    sentence_to_write_4_pred)-2]
                sentence_to_write_4_pred = sentence_to_write_4_pred + "\n"
                f4.write("{}".format(sentence_to_write_4 + sentence_to_write_4_pred))
                k += 1
                f.write("{}</br></br>".format(sentence_to_write))
                f2.write("{}\n\n".format(sentence_to_write_2))
                f3.write("{}\n\n".format(sentence_to_write_3))

            f2.write("\n\n")
            f3.write("\n\n")
            f.write("</br></br>")
            total_actual_keys += actual_keys
            total_guess += guess
            total_correct_guess += correct_guess
            total_test_loss += cost
            total_test_acc += acc

        print("\n")
        f.write("</body></html>")
        f.close()
        f2.close()
        f3.close()
        f4.close()

        F1, prec, rec = eval.F1(total_actual_keys, total_guess, total_correct_guess)
        #_, _, F1_2 = evaluate(tag_labels, tag_predictions, verbose=True)

        test_len = len(test_batches_sen)

        avg_test_loss = total_test_loss/test_len
        avg_test_acc = total_test_acc/test_len

        print(str(name)+" Test Loss= " +
              "{:.3f}".format(avg_test_loss) + ", Test Accuracy= " +
              "{:.3f}%".format(avg_test_acc*100)+", Test F1= " +
              "{:.5f}".format(F1)+", Test Precision= " +
              "{:.5f}".format(prec)+",Test Recall= " +
              "{:.5f}".format(rec), end="\n")
