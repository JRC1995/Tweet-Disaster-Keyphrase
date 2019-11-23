import tensorflow as tf
import numpy as np
import math
import tensorflow_hub as hub
import pickle


class Joint_Encoders:

    def __init__(self, sentences, sentences_string, pos, ipa, phono, elmo_dims, labels_1, labels_2,
                 true_seq_lens, train, learning_rate, fine_tune_lr,
                 word_embeddings, tags_set, model='BIBILSTM', alpha=0.5, hidden_size=100,
                 dropout=0.5, l2=1e-6, momentum=0.9,
                 fine_tune=True, gross_tune=False, optimizer_to_use='nadam',
                 embedding_dropout=False, grad_max_norm=5, variational_dropout=False):

        with open('../../Processed_Data/word_to_ipa_vec.pkl', 'rb') as fp:
            data = pickle.load(fp)

        self.phono_dims = data[2]
        self.ipa_len = data[3]

        self.sentences = sentences
        self.sentences_string = sentences_string
        self.pos = pos
        self.ipa = ipa
        self.phono = phono
        self.elmo_dims = elmo_dims
        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.true_seq_lens = true_seq_lens
        self.train = train
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.word_embeddings = word_embeddings
        self.tags_set = tags_set
        self.model = model
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.l2 = l2
        self.momentum = momentum
        self.fine_tune = fine_tune
        self.gross_tune = gross_tune
        self.optimizer_to_use = optimizer_to_use
        self.embedding_dropout = embedding_dropout
        self.max_grad_norm = grad_max_norm
        self.variational_dropout = variational_dropout
        self.max_char_len = 20

        tweet_pos_vocab = ['N', 'O', 'S', '^', 'Z', 'V', 'L', 'M', 'A', 'R', '!',
                           'D', 'P', '&', 'T', 'X', 'Y', '~', 'U', 'E', '$', ',', 'G']

        self.pos_len = len(tweet_pos_vocab)

        init = tf.zeros_initializer()

        with tf.variable_scope("fine_tune"):

            with tf.variable_scope("embedding"):

                if self.fine_tune is True:

                    self.embeddings = tf.get_variable("embedding", trainable=True,
                                                      dtype=tf.float32,
                                                      initializer=tf.constant(self.word_embeddings.tolist()))
                elif self.gross_tune is False:

                    self.embeddings = tf.get_variable("embedding", trainable=False,
                                                      dtype=tf.float32,
                                                      initializer=tf.constant(self.word_embeddings.tolist()))

        with tf.variable_scope("embedding"):

            self.pos_embeddings = tf.get_variable("pos_embedding", shape=[self.pos_len, 64], trainable=True,
                                                  dtype=tf.float32)
            self.ipa_embeddings = tf.get_variable("ipa_embedding", shape=[self.ipa_len, self.phono_dims], trainable=True,
                                                  dtype=tf.float32)

            if self.gross_tune is True:

                self.tf_embeddings = tf.get_variable("embedding", trainable=True,
                                                     dtype=tf.float32,
                                                     initializer=tf.constant(self.word_embeddings.tolist()))

        with tf.variable_scope("char_representation"):

            self.convW = tf.get_variable("convw", shape=[3, 2*self.phono_dims, 128],
                                         trainable=True,
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("softmax_1"):

            self.W_score_1 = tf.get_variable("W_score_1", shape=[2*self.hidden_size, 2],
                                             trainable=True,
                                             initializer=tf.contrib.layers.xavier_initializer())

            self.B_score_1 = tf.get_variable("Bias_score_1", dtype=tf.float32,
                                             shape=[2],
                                             trainable=True, initializer=init)

        with tf.variable_scope("softmax_2"):

            self.W_score_2 = tf.get_variable("W_score_2", shape=[2*self.hidden_size, len(self.tags_set)],
                                             trainable=True,
                                             initializer=tf.contrib.layers.xavier_initializer())

            self.B_score_2 = tf.get_variable("Bias_score_2", dtype=tf.float32,
                                             shape=[len(self.tags_set)],
                                             trainable=True, initializer=init)

        self.initiate_graph()
        self.compute_cost()
        self.fine_tuner()
        self.optimizer()
        self.predict()

    # %%

    def char_CNN(self, CNN_in):
        """CNN_in_reshaped = tf.reshape(CNN_in,
                                     [self.N*self.S,
                                      self.max_char_len,
                                      2*self.phono_dims])"""

        CNN_out = tf.nn.conv1d(CNN_in, self.convW, stride=1,
                               padding='SAME')

        # CNN_out = tf.nn.bias_add(CNN_out, self.convB)

        # CNN_out shape = [N*S, max_char_len, filters]

        # global avg pooling
        """
        pooled = tf.nn.relu(tf.layers.average_pooling1d(CNN_out, pool_size=self.max_char_len,
                                                        strides=self.max_char_len, padding='valid'))
        """
        pooled = tf.reduce_max(CNN_out, axis=1)

        # pooled  shape = [N*S, filters]

        pooled_reshaped = tf.reshape(pooled, [self.N, self.S, 128])

        # pooled_reshaped  shape = [N,S, filters]

        return pooled_reshaped

    def char_LSTM(self, LSTM_in):
        cell_fw_1 = tf.contrib.rnn.LSTMCell(64,
                                            initializer=tf.contrib.layers.xavier_initializer(),
                                            forget_bias=1.0,
                                            dtype=tf.float32,
                                            state_is_tuple=True)

        cell_bw_1 = tf.contrib.rnn.LSTMCell(64,
                                            initializer=tf.contrib.layers.xavier_initializer(),
                                            forget_bias=1.0,
                                            dtype=tf.float32,
                                            state_is_tuple=True)

        input_depth = 2*self.phono_dims

        keep_prob = 1

        cell_fw_1 = tf.nn.rnn_cell.DropoutWrapper(cell_fw_1,
                                                  input_keep_prob=keep_prob,
                                                  output_keep_prob=keep_prob,
                                                  state_keep_prob=1,
                                                  variational_recurrent=self.variational_dropout,
                                                  input_size=input_depth,
                                                  dtype=tf.float32)

        cell_bw_1 = tf.nn.rnn_cell.DropoutWrapper(cell_bw_1,
                                                  input_keep_prob=keep_prob,
                                                  output_keep_prob=keep_prob,
                                                  state_keep_prob=1,
                                                  variational_recurrent=self.variational_dropout,
                                                  input_size=input_depth,
                                                  dtype=tf.float32)

        init_state_fw_1 = cell_fw_1.zero_state(self.N, dtype=tf.float32)
        init_state_bw_1 = cell_bw_1.zero_state(self.N, dtype=tf.float32)

        (outputs_fw_1, outputs_bw_1), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw_1,
            cell_bw_1,
            LSTM_in,
            sequence_length=self.true_seq_lens,
            initial_state_fw=init_state_fw_1,
            initial_state_bw=init_state_bw_1,
            dtype=tf.float32,
            time_major=False)

        output = tf.concat([outputs_fw_1, outputs_bw_1], axis=-1)
        return output[:, -1]

    def BIULSTM(self, LSTM_in):

        with tf.variable_scope("rnn_1"):

            cell_fw_1 = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                forget_bias=1.0,
                                                dtype=tf.float32,
                                                state_is_tuple=True)

            cell_bw_1 = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                forget_bias=1.0,
                                                dtype=tf.float32,
                                                state_is_tuple=True)

            input_depth = self.D

            keep_prob = tf.cond(self.train, lambda: (1-self.dropout), lambda: 1.0)

            cell_fw_1 = tf.nn.rnn_cell.DropoutWrapper(cell_fw_1,
                                                      input_keep_prob=keep_prob,
                                                      output_keep_prob=keep_prob,
                                                      state_keep_prob=1,
                                                      variational_recurrent=self.variational_dropout,
                                                      input_size=input_depth,
                                                      dtype=tf.float32)

            cell_bw_1 = tf.nn.rnn_cell.DropoutWrapper(cell_bw_1,
                                                      input_keep_prob=keep_prob,
                                                      output_keep_prob=keep_prob,
                                                      state_keep_prob=1,
                                                      variational_recurrent=self.variational_dropout,
                                                      input_size=input_depth,
                                                      dtype=tf.float32)

            init_state_fw_1 = cell_fw_1.zero_state(self.N, dtype=tf.float32)
            init_state_bw_1 = cell_bw_1.zero_state(self.N, dtype=tf.float32)

            (outputs_fw_1, outputs_bw_1), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw_1,
                cell_bw_1,
                LSTM_in,
                sequence_length=self.true_seq_lens,
                initial_state_fw=init_state_fw_1,
                initial_state_bw=init_state_bw_1,
                dtype=tf.float32,
                time_major=False)

            output_1 = tf.concat([outputs_fw_1, outputs_bw_1], axis=-1)

        with tf.variable_scope("rnn_2"):

            cell_2 = tf.contrib.rnn.LSTMCell(2*self.hidden_size,
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             forget_bias=1.0,
                                             dtype=tf.float32,
                                             state_is_tuple=True)

            input_depth = 2*self.hidden_size

            keep_prob = tf.cond(self.train, lambda: (1-self.dropout), lambda: 1.0)

            cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell_2,
                                                   input_keep_prob=1,
                                                   output_keep_prob=keep_prob,
                                                   state_keep_prob=1,
                                                   variational_recurrent=self.variational_dropout,
                                                   input_size=input_depth,
                                                   dtype=tf.float32)

            init_state_2 = cell_2.zero_state(self.N, dtype=tf.float32)

            output_2, _ = tf.nn.dynamic_rnn(
                cell_2,
                output_1,
                sequence_length=self.true_seq_lens,
                initial_state=init_state_2,
                dtype=tf.float32,
                time_major=False,
            )

        return output_1, output_2

    def BIBILSTM(self, LSTM_in):

        with tf.variable_scope("rnn_1"):

            cell_fw_1 = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                forget_bias=1.0,
                                                dtype=tf.float32,
                                                state_is_tuple=True)

            cell_bw_1 = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                forget_bias=1.0,
                                                dtype=tf.float32,
                                                state_is_tuple=True)

            input_depth = self.D

            keep_prob = tf.cond(self.train, lambda: (1-self.dropout), lambda: 1.0)

            cell_fw_1 = tf.nn.rnn_cell.DropoutWrapper(cell_fw_1,
                                                      input_keep_prob=keep_prob,
                                                      output_keep_prob=keep_prob,
                                                      state_keep_prob=1,
                                                      variational_recurrent=self.variational_dropout,
                                                      input_size=input_depth,
                                                      dtype=tf.float32)

            cell_bw_1 = tf.nn.rnn_cell.DropoutWrapper(cell_bw_1,
                                                      input_keep_prob=keep_prob,
                                                      output_keep_prob=keep_prob,
                                                      state_keep_prob=1,
                                                      variational_recurrent=self.variational_dropout,
                                                      input_size=input_depth,
                                                      dtype=tf.float32)

            init_state_fw_1 = cell_fw_1.zero_state(self.N, dtype=tf.float32)
            init_state_bw_1 = cell_bw_1.zero_state(self.N, dtype=tf.float32)

            (outputs_fw_1, outputs_bw_1), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw_1,
                cell_bw_1,
                LSTM_in,
                sequence_length=self.true_seq_lens,
                initial_state_fw=init_state_fw_1,
                initial_state_bw=init_state_bw_1,
                dtype=tf.float32,
                time_major=False)

            output_1 = tf.concat([outputs_fw_1, outputs_bw_1], axis=-1)

        with tf.variable_scope("rnn_2"):

            cell_fw_2 = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                forget_bias=1.0,
                                                dtype=tf.float32,
                                                state_is_tuple=True)

            cell_bw_2 = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                forget_bias=1.0,
                                                dtype=tf.float32,
                                                state_is_tuple=True)

            input_depth = 2*self.hidden_size

            keep_prob = tf.cond(self.train, lambda: (1-self.dropout), lambda: 1.0)

            cell_fw_2 = tf.nn.rnn_cell.DropoutWrapper(cell_fw_2,
                                                      input_keep_prob=1,
                                                      output_keep_prob=keep_prob,
                                                      state_keep_prob=1,
                                                      variational_recurrent=self.variational_dropout,
                                                      input_size=input_depth,
                                                      dtype=tf.float32)

            cell_bw_2 = tf.nn.rnn_cell.DropoutWrapper(cell_bw_2,
                                                      input_keep_prob=1,
                                                      output_keep_prob=keep_prob,
                                                      state_keep_prob=1,
                                                      variational_recurrent=self.variational_dropout,
                                                      input_size=input_depth,
                                                      dtype=tf.float32)

            init_state_fw_2 = cell_fw_2.zero_state(self.N, dtype=tf.float32)
            init_state_bw_2 = cell_bw_2.zero_state(self.N, dtype=tf.float32)

            (outputs_fw_2, outputs_bw_2), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw_2,
                cell_bw_2,
                output_1,
                sequence_length=self.true_seq_lens,
                initial_state_fw=init_state_fw_2,
                initial_state_bw=init_state_bw_2,
                dtype=tf.float32,
                time_major=False)

            output_2 = tf.concat([outputs_fw_2, outputs_bw_2], axis=-1)

        return output_1, output_2

    def initiate_graph(self):

        self.N = tf.shape(self.sentences)[0]
        self.S = tf.shape(self.sentences)[1]
        self.D = len(self.word_embeddings[0]) + self.elmo_dims + 64 + 128

        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        tokens_input = self.sentences_string
        tokens_length = self.true_seq_lens
        elmo_sentences = elmo(
            inputs={
                "tokens": tokens_input,
                "sequence_len": tokens_length
            },
            signature="tokens",
            as_dict=True)["elmo"]

        embd_sentences = tf.nn.embedding_lookup(self.embeddings, self.sentences)
        pos_embd = tf.nn.embedding_lookup(self.pos_embeddings, self.pos)
        ipa_embd = tf.nn.embedding_lookup(self.ipa_embeddings, self.ipa)

        ipa_embd = tf.reshape(ipa_embd, [self.N, self.S, self.max_char_len, self.phono_dims])

        char_embd = tf.concat([ipa_embd, self.phono], axis=-1)

        ipa_phono_cnn_in = tf.reshape(
            char_embd, [self.N*self.S, self.max_char_len, 2*self.phono_dims])
        ipa_phono_embd = self.char_CNN(ipa_phono_cnn_in)
        #ipa_phono_embd = tf.reshape(ipa_phono_embd, [self.N, self.S, 128])

        # Concatenate words in the window:
        embd_sentences = tf.reshape(embd_sentences, [self.N, self.S, len(self.word_embeddings[0])])
        elmo_sentences = tf.reshape(elmo_sentences, [self.N, self.S, self.elmo_dims])
        pos_embd = tf.reshape(pos_embd, [self.N, self.S, 64])

        ultimate_represent = tf.concat(
            [embd_sentences, elmo_sentences, pos_embd, ipa_phono_embd], axis=-1)  # , ipa_embd

        # if self.embedding_dropout is False:

        # ultimate_represent = tf.layers.dropout(ultimate_represent,
        # rate=self.dropout,
        # training=self.train)

        if self.model == 'BIBILSTM':

            output_1, output_2 = self.BIBILSTM(ultimate_represent)

        else:  # Use BI-UI-LSTM

            output_1, output_2 = self.BIULSTM(ultimate_represent)

        with tf.variable_scope("softmax_1"):

            output_1_squeeze = tf.reshape(output_1, [-1, 2*self.hidden_size])

            score = tf.matmul(output_1_squeeze, self.W_score_1) + self.B_score_1

            self.score_1 = tf.reshape(score, [self.N, self.S, 2])

        with tf.variable_scope("softmax_2"):

            output_2_squeeze = tf.reshape(output_2, [-1, 2*self.hidden_size])

            score = tf.matmul(output_2_squeeze, self.W_score_2) + self.B_score_2

            self.score_2 = tf.reshape(score, [self.N, self.S, len(self.tags_set)])

    # %%

    def compute_cost(self):

        filtered_trainables = [var for var in tf.trainable_variables() if
                               not("Bias" in var.name or "bias" in var.name
                                   or "noreg" in var.name)]

        regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var
                                        in filtered_trainables])

        with tf.variable_scope("loss"):

            loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_1,
                                                                    logits=self.score_1)
            loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_2,
                                                                    logits=self.score_2)
            total_loss = self.alpha*loss_1 + (1-self.alpha)*loss_2

            pad_mask = tf.sequence_mask(self.true_seq_lens,
                                        maxlen=self.S,
                                        dtype=tf.float32)
            pad_mask = tf.reshape(pad_mask, [self.N, self.S])

            masked_total_loss = tf.multiply(total_loss, pad_mask)

            self.loss = tf.reduce_mean(masked_total_loss) + self.l2*regularization

    # %%

    def fine_tuner(self):

        # cost = self.cost_function()
        if self.fine_tune is True:

            fine_tune_embd = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               "fine_tune")
            self.fine_tune_vars = fine_tune_embd

            if self.optimizer_to_use.lower() == 'nadam':
                fine_tuning_optimizer = tf.contrib.opt.NadamOptimizer(
                    learning_rate=self.fine_tune_lr)
            else:
                fine_tuning_optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self.fine_tune_lr,
                    momentum=self.momentum,
                    use_nesterov=True)

            gvs_fine_tune = fine_tuning_optimizer.compute_gradients(
                self.loss, var_list=fine_tune_embd)

            # gradient clipping by norm
            capped_gvs_fine_tune = [(tf.clip_by_norm(grad, self.max_grad_norm), var)
                                    for grad, var in gvs_fine_tune]

            self.fine_tune_op = fine_tuning_optimizer.apply_gradients(
                capped_gvs_fine_tune)

        else:

            self.fine_tune_op = tf.constant(0)

    # %%

    def optimizer(self):

        # cost = self.cost_function()

        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if self.fine_tune is True:
            rest_vars = [v for v in all_vars if v not in self.fine_tune_vars]
        else:
            rest_vars = all_vars

        if self.optimizer_to_use.lower() == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(
                learning_rate=self.learning_rate)
        else:
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                use_nesterov=True)

        gvs = optimizer.compute_gradients(self.loss, var_list=rest_vars)

        capped_gvs = [(tf.clip_by_norm(grad, self.max_grad_norm), var) for grad, var in gvs]

        self.train_op = optimizer.apply_gradients(capped_gvs)

    # %%

    def predict(self):

        self.predictions_1 = tf.argmax(self.score_1,
                                       axis=-1,
                                       output_type=tf.int32)

        self.predictions_2 = tf.argmax(self.score_2,
                                       axis=-1,
                                       output_type=tf.int32)

        # Comparing predicted sequence with labels

        comparison = tf.cast(tf.equal(self.predictions_2, self.labels_2),
                             tf.float32)

        # Masking to ignore the effect of pads while calculating accuracy
        pad_mask = tf.sequence_mask(self.true_seq_lens,
                                    maxlen=self.S,
                                    dtype=tf.bool)

        masked_comparison = tf.boolean_mask(comparison, pad_mask)

        # Accuracy
        self.accuracy = tf.reduce_mean(masked_comparison)
