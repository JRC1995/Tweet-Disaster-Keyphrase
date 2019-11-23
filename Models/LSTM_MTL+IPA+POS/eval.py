from __future__ import division
import numpy as np


class eval_NER_exact_entity_match:

    def tpfnfp(self, batch_labels, batch_predictions, true_seq_lens, tags_set):

        correct_guess = 0
        guess = 0
        actual_keys = 0
        for i in range(0, len(batch_labels)):
            j = 0
            while j < true_seq_lens[i]:

                # print(j)
                # print(tags_set[batch_labels[i][j]][0])
                if tags_set[batch_labels[i][j]] == 'B':
                    actual_keys += 1
                    chunk_label = []
                    chunk_prediction = []
                    chunk_label.append(batch_labels[i][j])
                    chunk_prediction.append(batch_predictions[i][j])

                    j += 1

                    while j < true_seq_lens[i] and\
                            tags_set[batch_labels[i][j]] in ['E', 'I']:
                        chunk_label.append(batch_labels[i][j])
                        chunk_prediction.append(batch_predictions[i][j])
                        j += 1

                    flag = 0

                    if j < true_seq_lens[i]:
                        if tags_set[batch_predictions[i][j-1]] != 'E':
                            if tags_set[batch_predictions[i][j]] == 'I':
                                flag = 1

                    if flag == 0:

                        label_entity = np.asarray(chunk_label, dtype=np.int32)
                        prediction_entity = np.asarray(chunk_prediction, dtype=np.int32)

                        if np.all(np.equal(label_entity, prediction_entity)):
                            correct_guess += 1

                elif tags_set[batch_labels[i][j]] == 'S':
                    actual_keys += 1

                    label_entity = np.asarray(batch_labels[i][j], dtype=np.int32)
                    prediction_entity = np.asarray(batch_predictions[i][j], dtype=np.int32)

                    if np.all(np.equal(label_entity, prediction_entity)):
                        correct_guess += 1

                    j += 1

                elif tags_set[batch_labels[i][j]] == 'O':
                    j += 1

            j = 0

            while j < true_seq_lens[i]:

                if tags_set[batch_predictions[i][j]] in ['B', 'S']:
                    guess += 1

                """elif tags_set[batch_predictions[i][j]][0] == 'I':
                    if j == 0:
                        guess += 1
                    elif tags_set[batch_predictions[i][j-1]][0] in ['S', 'E', 'O']:
                        guess += 1"""
                j += 1

        return actual_keys, guess, correct_guess

    def F1(self, actual_keys, guess, correct_guess):

        if guess == 0:
            precision = 0
        else:
            precision = correct_guess/guess

        if actual_keys == 0:
            recall = 0
        else:
            recall = correct_guess/actual_keys

        if (recall+precision) == 0:
            F1 = 0
        else:
            F1 = (2*recall*precision)/(recall+precision)

        return F1,precision,recall
