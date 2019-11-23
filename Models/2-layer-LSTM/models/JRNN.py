import tensorflow as tf
import configparser
import numpy as np
import math


class Joint_Encoders:

    def __init__(self, sentences, labels_1, labels_2,
                 true_seq_lens, train, learning_rate, fine_tune_lr,
                 word_embeddings, tags_set, class_weights_1, class_weights_2, model='BIULSTM', alpha=0.5, hidden_size=100,
                 dropout=0.5, l2=1e-8, momentum=0.9,
                 fine_tune=True, gross_tune=False, optimizer_to_use='nadam', variational_dropout=False,
                 embedding_dropout=False, grad_max_norm=5):

        self.cp = configparser.ConfigParser()
        self.cp.read('configs.ini')
        self.window_size = int(self.cp['HYPERPARAMETERS']['window_size'])

        self.sentences = sentences
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
        self.variational_dropout = variational_dropout
        self.l2 = l2
        self.momentum = momentum
        self.fine_tune = fine_tune
        self.gross_tune = gross_tune
        self.optimizer_to_use = optimizer_to_use
        self.embedding_dropout = embedding_dropout
        self.max_grad_norm = grad_max_norm
        self.class_weights_1 = class_weights_1
        self.class_weights_2 = class_weights_2

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

            if self.gross_tune is True:

                self.tf_embeddings = tf.get_variable("embedding", trainable=True,
                                                     dtype=tf.float32,
                                                     initializer=tf.constant(self.word_embeddings.tolist()))


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
        self.D = self.window_size*len(self.word_embeddings[0])

        # EMBEDDING DROPOUTS (#type level)

        if self.embedding_dropout is True:

            self.embeddings = tf.layers.dropout(self.embeddings,
                                                rate=self.dropout,
                                                noise_shape=[len(self.word_embeddings), 1],
                                                training=self.train)

        embd_sentences = tf.nn.embedding_lookup(self.embeddings, self.sentences)
        # Concatenate words in the window:
        embd_sentences = tf.reshape(embd_sentences, [self.N, self.S, self.D])

        if self.model == 'BIBILSTM':

            output_1, output_2 = self.BIBILSTM(embd_sentences)

        else:  # Use BI-UI-LSTM

            output_1, output_2 = self.BIULSTM(embd_sentences)


        with tf.variable_scope("softmax_2"):

            output_2_squeeze = tf.reshape(output_2, [-1, 2*self.hidden_size])

            score = tf.matmul(output_2_squeeze, self.W_score_2) + self.B_score_2

            self.score_2 = tf.reshape(score, [self.N, self.S, len(self.tags_set)])

    # %%

    def compute_cost(self):

        filtered_trainables = [var for var in tf.trainable_variables() if
                               not("Bias" in var.name or "bias" in var.name
                                   or "noreg" in var.name)]

        regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var in filtered_trainables])
        with tf.variable_scope("loss"):

            weights_1 = tf.gather(self.class_weights_1, self.labels_1)
            weights_2 = tf.gather(self.class_weights_2, self.labels_2)

        
            loss_2 = tf.losses.sparse_softmax_cross_entropy(
                labels=self.labels_2,
                logits=self.score_2,
                weights=weights_2,
            )
            total_loss = loss_2

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

            fine_tuning_optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.fine_tune_lr)

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

        """if self.optimizer_to_use.lower() == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(
                learning_rate=self.learning_rate)
        else:
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                use_nesterov=True)"""

        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate)

        gvs = optimizer.compute_gradients(self.loss, var_list=rest_vars)

        capped_gvs = [(tf.clip_by_norm(grad, self.max_grad_norm), var) for grad, var in gvs]

        self.train_op = optimizer.apply_gradients(capped_gvs)

    # %%

    def predict(self):

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
