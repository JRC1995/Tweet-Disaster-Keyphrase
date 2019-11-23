import numpy as np


class bucket_and_batch:

    def __init__(self):
        self.window_size = 3

    def bucket_and_batch(self, sentences, char_sentences,
                         vocab2idx, batch_size):

        true_seq_lens = np.zeros((len(sentences)), dtype=int)
        PAD_vocab_index = vocab2idx['<pad']

        for i in range(len(sentences)):
            true_seq_lens[i] = len(sentences[i])

        # sorted in descending order after flip
        sorted_by_len_indices = np.flip(np.argsort(true_seq_lens), 0)

        sorted_sentences = []

        for i in range(len(sentences)):

            sorted_sentences.append(
                sentences[sorted_by_len_indices[i]])

        i = 0
        batches_sentences = []
        batches_true_seq_lens = []

        while i < len(sentences):

            if i+batch_size > len(sentences):
                batch_size = len(sentences)-i

            batch_sentences = []
            batch_true_seq_lens = []

            max_len = len(sorted_sentences[i])

            for j in range(i, i + batch_size):

                line_sentences = []

                for k1 in range(0, max_len):

                    if k1 >= len(sorted_sentences[j]):
                        line_sentences.append(PAD_vocab_index)
                    else:
                        line_sentences.append(sorted_sentences[j][k1])

                batch_sentences.append(line_sentences)
                batch_true_seq_lens.append(len(sorted_sentences[j]))

            batch_sentences = np.asarray(batch_sentences, dtype=int)

            batches_sentences.append(batch_sentences)
            batches_true_seq_lens.append(batch_true_seq_lens)

            i += batch_size

        return batches_sentences, batches_true_seq_lens
