# %%
from models.JRNN import Joint_Encoders
import pickle
import numpy as np
import configparser
import tensorflow as tf
from bucket_and_batch import bucket_and_batch
from random import shuffle
from eval import eval_NER_exact_entity_match
import math
import tensorflow_hub as hub

# %%
cp = configparser.ConfigParser()
bucket_and_batch = bucket_and_batch()


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


learning_rate_init = learning_rate

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


tf_sentences = tf.placeholder(tf.int32, [None, None])
tf_sentences_string = tf.placeholder(tf.string, [None, None])
tf_labels_1 = tf.placeholder(tf.int32, [None, None])
tf_labels_2 = tf.placeholder(tf.int32, [None, None])
tf_true_seq_lens = tf.placeholder(tf.int32, [None])
tf_train = tf.placeholder(tf.bool)
tf_learning_rate = tf.placeholder(tf.float32)
tf_fine_tune_lr = tf.placeholder(tf.float32)

model = Joint_Encoders(sentences=tf_sentences,
                       sentences_string=tf_sentences_string,
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

Hyperparameters = [window_size, hidden_size, learning_rate, momentum, decay,
                   fine_tune_lr, fine_tune, gross_tune, l2, dropout,
                   optimizer_to_use, variational_dropout, embedding_dropout,
                   alpha, grad_max_norm, model_name, variational_dropout]

with open('SAVED_MODEL/test_hyperparameters', 'wb') as fp:
    pickle.dump(Hyperparameters, fp)

eval = eval_NER_exact_entity_match()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

with tf.Session(config=config) as sess:  # Start Tensorflow Session

    display_step = 50
    patience = 5

    load = input("\nLoad checkpoint? y/n: ")
    print("")
    saver = tf.train.Saver()

    if load.lower() == 'y':

        print('Loading pre-trained weights for the model...')

        sess.run(tf.tables_initializer())

        saver.restore(sess, 'SAVED_MODEL/model.ckpt')
        sess.run(tf.global_variables())

        with open('SAVED_MODEL/train_vals', 'rb') as fp:
            train_data = pickle.load(fp)

        epoch_ = train_data[3]
        loss_list = train_data[5]
        acc_list = train_data[6]
        F1_list = train_data[7]
        val_loss_list = train_data[8]
        val_acc_list = train_data[9]
        val_F1_list = train_data[10]
        best_val_loss = train_data[0]
        best_val_F1 = train_data[2]
        best_val_F1_loss = train_data[1]
        impatience = train_data[4]

        print('\nRESTORATION COMPLETE\n')

    else:

        loss_list = []
        acc_list = []
        F1_list = []
        val_loss_list = []
        val_acc_list = []
        val_F1_list = []
        best_val_loss = 2**30
        best_val_F1 = 0
        best_val_F1_loss = 2**30
        impatience = 0
        epoch_ = 0

        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.tables_initializer())

    # epoch_ == Number of pretrained epochs (if loading from saved model)

    for epoch in range(0, epochs):

        if fine_tune is True:

            if (epoch+epoch_) >= freeze_epochs:
                fine_tune_lr_tf = fine_tune_lr
            else:
                fine_tune_lr_tf = 0
        else:
            fine_tune_lr_tf = 0

        train_batches_sen, train_batches_string, train_batches_labels_1, train_batches_labels_2,\
            train_batches_true_seq_lens = bucket_and_batch.bucket_and_batch(
                train_tweets_vec, train_tweets, train_labels_1, train_labels_2,
                vocab2idx, batch_size, labels_set)

        batches_indices = [i for i in range(0, len(train_batches_sen))]
        shuffle(batches_indices)

        total_train_acc = 0
        total_train_loss = 0

        for i in range(0, len(train_batches_sen)):

            j = int(batches_indices[i])

            cost, prediction, acc, _, _ = sess.run([model.loss,
                                                    model.predictions_2,
                                                    model.accuracy,
                                                    model.fine_tune_op,
                                                    model.train_op],
                                                   feed_dict={tf_sentences: train_batches_sen[j],
                                                              tf_sentences_string: train_batches_string[j],
                                                              tf_labels_1: train_batches_labels_1[j],
                                                              tf_labels_2: train_batches_labels_2[j],
                                                              tf_true_seq_lens: train_batches_true_seq_lens[j],
                                                              tf_fine_tune_lr: fine_tune_lr_tf,
                                                              tf_learning_rate: learning_rate,
                                                              tf_train: True})

            actual_keys, guess, correct_guess = eval.tpfnfp(
                train_batches_labels_2[j],
                prediction,
                train_batches_true_seq_lens[j],
                labels_set)

            F1, _, _ = eval.F1(actual_keys, guess, correct_guess)

            total_train_acc += acc
            total_train_loss += cost

            if i % display_step == 0:

                print("Iter "+str(i)+", Loss= " +
                      "{:.3f}".format(cost)+", Accuracy= " +
                      "{:.3f}%".format(acc*100)+", F1= " +
                      "{:.3f}".format(F1))

            # break
        val_batches_sen, val_batches_string, val_batches_labels_1, val_batches_labels_2,\
            val_batches_true_seq_lens = bucket_and_batch.bucket_and_batch(
                val_tweets_vec, val_tweets, val_labels_1, val_labels_2,
                vocab2idx, batch_size, labels_set)

        total_actual_keys = 0
        total_guess = 0
        total_correct_guess = 0
        total_val_acc = 0
        total_val_loss = 0

        for i in range(0, len(val_batches_sen)):

            cost, prediction, acc, = sess.run([model.loss,
                                               model.predictions_2,
                                               model.accuracy],
                                              feed_dict={tf_sentences: val_batches_sen[i],
                                                         tf_sentences_string: val_batches_string[i],
                                                         tf_labels_1: val_batches_labels_1[i],
                                                         tf_labels_2: val_batches_labels_2[i],
                                                         tf_true_seq_lens: val_batches_true_seq_lens[i],
                                                         tf_train: False})

            actual_keys, guess, correct_guess = eval.tpfnfp(
                val_batches_labels_2[i],
                prediction,
                val_batches_true_seq_lens[i],
                labels_set)

            total_actual_keys += actual_keys
            total_guess += guess
            total_correct_guess += correct_guess
            total_val_loss += cost
            total_val_acc += acc

        val_F1, _, _ = eval.F1(total_actual_keys, total_guess, total_correct_guess)
        val_len = len(val_batches_sen)
        train_len = len(train_batches_sen)

        avg_val_loss = total_val_loss/val_len
        avg_val_acc = total_val_acc/val_len
        avg_train_loss = total_train_loss/train_len
        avg_train_acc = total_train_acc/train_len

        loss_list.append(avg_train_loss)
        acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)
        val_F1_list.append(val_F1)

        print("\n\nEpoch " + str(epoch+epoch_) + ", Average Training Loss= " +
              "{:.3f}".format(avg_train_loss) + ", Average Training Accuracy= " +
              "{:.3f}%".format(avg_train_acc*100)+"")

        print("Epoch " + str(epoch+epoch_) + ", Validation Loss= " +
              "{:.3f}".format(avg_val_loss) + ", validation Accuracy= " +
              "{:.3f}%".format(avg_val_acc*100)+", validation F1= " +
              "{:.3f}".format(val_F1))

        e = 0.0009
        flag = 0
        impatience += 1

        if val_F1 >= best_val_F1:

            best_val_F1 = val_F1
            impatience = 0
            flag = 1

        if flag == 1:

            saver.save(sess, 'SAVED_MODEL/model.ckpt')

            PICKLE_list = [best_val_loss, best_val_F1_loss, best_val_F1, epoch+epoch_+1, impatience,
                           loss_list, acc_list, F1_list, val_loss_list, val_acc_list, val_F1_list]

            with open('SAVED_MODEL/train_vals', 'wb') as fp:
                pickle.dump(PICKLE_list, fp)

            print("Checkpoint created!")

        print("\n")

        if impatience > patience:
            print("\nEarly Stopping since best validation loss not decreasing for "+str(patience)+" epochs.")
            break

    print("\nOptimization Finished!\n")

    print("Best Validation Loss: %.3f" % ((best_val_loss)))
