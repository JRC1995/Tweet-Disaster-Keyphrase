
import configparser
import numpy as np


class bucket_and_batch:

    def __init__(self):
        self.cp = configparser.ConfigParser()
        self.cp.read('configs.ini')

    def bucket_and_batch(self, sentences, char_sentences, pos, ipa, phono, labels_1, labels_2,
                         vocab2idx, batch_size, tags_set):

        phonological_features = ['syl', 'son', 'cons', 'cont', 'delrel',
                                 'lat', 'nas', 'strid', 'voi', 'sg', 'cg',
                                 'ant', 'cor', 'distr', 'lab', 'hi', 'lo',
                                 'back', 'round', 'velaric', 'tense', 'long']
        phono_dim = len(phonological_features)

        true_seq_lens = np.zeros((len(sentences)), dtype=int)
        PAD_vocab_index = vocab2idx['<pad>']
        Negative_label = tags_set.index('O')
        PAD_tag_idx = Negative_label
        tweet_pos_vocab = ['N', 'O', 'S', '^', 'Z', 'V', 'L', 'M', 'A', 'R', '!',
                           'D', 'P', '&', 'T', 'X', 'Y', '~', 'U', 'E', '$', ',', 'G']
        ipa_pad = np.zeros((20), dtype=np.float32)
        pos_pad = tweet_pos_vocab.index(',')
        phono_pad = np.zeros((20, phono_dim), dtype=np.float32)

        for i in range(len(sentences)):
            true_seq_lens[i] = len(sentences[i])

        # sorted in descending order after flip
        sorted_by_len_indices = np.flip(np.argsort(true_seq_lens), 0)

        sorted_sentences = []
        sorted_sentences_char = []
        sorted_pos = []
        sorted_ipa = []
        sorted_phono = []
        sorted_labels_1 = []
        sorted_labels_2 = []

        for i in range(len(sentences)):

            sorted_sentences.append(
                sentences[sorted_by_len_indices[i]])

            sorted_sentences_char.append(
                char_sentences[sorted_by_len_indices[i]])

            sorted_pos.append(pos[sorted_by_len_indices[i]])

            sorted_ipa.append(ipa[sorted_by_len_indices[i]])

            sorted_phono.append(phono[sorted_by_len_indices[i]])

            sorted_labels_1.append(
                labels_1[sorted_by_len_indices[i]])

            sorted_labels_2.append(
                labels_2[sorted_by_len_indices[i]])

        i = 0
        batches_sentences = []
        batches_sentences_char = []
        batches_pos = []
        batches_ipa = []
        batches_phono = []
        batches_labels_1 = []
        batches_labels_2 = []
        batches_true_seq_lens = []

        while i < len(sentences):

            if i+batch_size > len(sentences):
                batch_size = len(sentences)-i

            batch_sentences = []
            batch_sentences_char = []
            batch_pos = []
            batch_ipa = []
            batch_phono = []
            batch_labels_1 = []
            batch_labels_2 = []
            batch_true_seq_lens = []

            max_len = len(sorted_sentences[i])

            for j in range(i, i + batch_size):

                line_sentences = []
                line_chars = []
                line_pos = []
                line_ipa = []
                line_phono = []
                # print(len(sorted_sentences_char[j]))
                # print(len(sorted_sentences[j]))
                line_labels_1 = []
                line_labels_2 = []

                for k1 in range(0, max_len):

                    if k1 >= len(sorted_sentences[j]):

                        # line_sentences.append(
                            # [PAD_vocab_index for l in range(self.window_size)])
                        line_sentences.append([PAD_vocab_index for l in range(3)])
                        line_chars.append("0")
                        line_pos.append(pos_pad)
                        line_ipa.append(ipa_pad)
                        line_phono.append(phono_pad)
                        line_labels_1.append(0)
                        line_labels_2.append(PAD_tag_idx)

                    else:

                        line_sentences.append(sorted_sentences[j][k1])
                        line_chars.append(sorted_sentences_char[j][k1])
                        line_pos.append(sorted_pos[j][k1])
                        line_ipa.append(sorted_ipa[j][k1])
                        line_phono.append(sorted_phono[j][k1])
                        line_labels_1.append(sorted_labels_1[j][k1])
                        line_labels_2.append(sorted_labels_2[j][k1])

                batch_sentences.append(line_sentences)
                batch_sentences_char.append(line_chars)
                batch_pos.append(line_pos)
                batch_ipa.append(line_ipa)
                batch_phono.append(line_phono)
                batch_labels_1.append(line_labels_1)
                batch_labels_2.append(line_labels_2)
                batch_true_seq_lens.append(len(sorted_sentences[j]))

            batch_sentences = np.asarray(batch_sentences, dtype=int)
            batch_sentences_char = batch_sentences_char
            batch_pos = np.asarray(batch_pos, dtype=int)
            batch_ipa = np.asarray(batch_ipa, dtype=int)
            batch_phono = np.asarray(batch_phono, dtype=np.float32)
            #batch_ipa = np.asarray(batch_ipa, dtype=np.float32)
            # for i in range(len(batch_sentences_char)):
            # print(len(batch_sentences_char[i]))
            # print("\n\n")
            batch_labels_1 = np.asarray(batch_labels_1,
                                        dtype=int)
            batch_labels_2 = np.asarray(batch_labels_2, dtype=int)

            batches_sentences.append(batch_sentences)
            batches_sentences_char.append(batch_sentences_char)
            batches_pos.append(batch_pos)
            batches_ipa.append(batch_ipa)
            batches_phono.append(batch_phono)
            batches_labels_1.append(batch_labels_1)
            batches_labels_2.append(batch_labels_2)
            batches_true_seq_lens.append(batch_true_seq_lens)

            i += batch_size

        return batches_sentences, batches_sentences_char, batches_pos, batches_ipa, batches_phono, batches_labels_1, batches_labels_2,\
            batches_true_seq_lens
