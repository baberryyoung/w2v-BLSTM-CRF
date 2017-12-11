import math
import helper
import numpy as np
import tensorflow as tf
import time


class BILSTM_CRF(object):
    def __init__(self, num_chars, num_classes, num_steps=100, num_epochs=100, embedding_matrix=None, is_training=True,
                 is_crf=True, weight=False):
        # Parameter
        self.max_f1 = 0
        self.learning_rate = 0.002
        self.dropout_rate = 0.5
        self.batch_size = 128
        self.num_layers = 1
        self.emb_dim = 128
        self.hidden_dim = 100
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_classes = num_classes

        # placeholder of x, y and weight
        self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])   #确认输入是否应该钰batch_size一致
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])  #确认输出是否应该钰batch_size一致
        self.targets_weight = tf.placeholder(tf.float32, [self.batch_size, self.num_steps])
        self.targets_transition = tf.placeholder(tf.int32, [None])

        # char embedding
        if embedding_matrix.any() != None:
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)#input is a index,look up index in the embedding
        #self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        #self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])  #不明白为什么要平铺开
        #self.inputs_emb = tf.split(self.inputs_emb, self.num_steps, 0)

        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # dropout
        if is_training:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        # forward and backward
        self.outputs, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            self.inputs_emb,
            sequence_length=self.length,
            dtype=tf.float32

        )

        # softmax
        self.outputs = tf.reshape(tf.concat( self.outputs,2), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        if not is_crf:
            pass
        else:
            self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_steps, self.num_classes])
            self.transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])

            dummy_val = -1000
            class_pad = tf.Variable(dummy_val * np.ones((self.batch_size, self.num_steps, 1)), dtype=tf.float32)
            self.observations = tf.concat([self.tags_scores, class_pad], 2)

            begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]),
                                    trainable=False, dtype=tf.float32)
            end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]),
                                  trainable=False, dtype=tf.float32)
            begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
            end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])

            self.observations = tf.concat([begin_vec, self.observations, end_vec], 1)

            self.mask = tf.cast(tf.reshape(tf.sign(self.targets), [self.batch_size * self.num_steps]), tf.float32)

            # point score
            self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]),
                                         tf.range(0, self.batch_size * self.num_steps) * self.num_classes + tf.reshape(
                                             self.targets, [self.batch_size * self.num_steps]))
            self.point_score *= self.mask

            # transition score
            self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), self.targets_transition)

            # real score
            self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)

            # all path score
            self.total_path_score, self.max_scores, self.max_scores_pre = self.forward(self.observations,
                                                                                       self.transitions, self.length)

            # loss
            self.loss = - (self.target_path_score - self.total_path_score)

        # summary
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.val_summary = tf.summary.scalar("loss", self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size, 0), [self.batch_size, self.num_classes + 1, self.num_classes + 1 ])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, self.num_classes+1, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, self.num_classes + 1, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, self.num_classes + 1])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, dimension=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, self.num_classes + 1, 1])
            alphas.append(alpha_t)
            previous = alpha_t

        alphas = tf.reshape(tf.concat(alphas, 0), [self.num_steps + 2, self.batch_size, self.num_classes + 1, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), self.num_classes + 1, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.num_classes + 1, 1])

        max_scores = tf.reshape(tf.concat(max_scores, 0), (self.num_steps + 1, self.batch_size, self.num_classes + 1))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre, 0), (self.num_steps + 1, self.batch_size, self.num_classes + 1))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre

    def train(self, sess, save_file, X_train, y_train, X_val, y_val):
        saver = tf.train.Saver()

        char2id, id2char = helper.loadMap("char2id")
        label2id, id2label = helper.loadMap("label2id")

        merged = tf.summary.merge_all()
        summary_writer_train = tf.summary.FileWriter('loss_log/train_loss', sess.graph)
        summary_writer_val = tf.summary.FileWriter('loss_log/val_loss', sess.graph)

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))

        cnt = 0
        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]
            print("current epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                # train
                X_train_batch, y_train_batch = helper.nextBatch(X_train, y_train,
                                                                start_index=iteration * self.batch_size,
                                                                batch_size=self.batch_size)
                y_train_weight_batch = 1 + np.array((y_train_batch == label2id['disorder']),float)
                transition_batch = helper.getTransition(y_train_batch, self.num_classes)

                _, loss_train, max_scores, max_scores_pre, length, train_summary = \
                    sess.run([
                        self.optimizer,
                        self.loss,
                        self.max_scores,
                        self.max_scores_pre,
                        self.length,
                        self.train_summary
                    ],
                        feed_dict={
                            self.targets_transition: transition_batch,
                            self.inputs: X_train_batch,
                            self.targets: y_train_batch,
                            self.targets_weight: y_train_weight_batch
                        })

                predicts_train = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
                if iteration % 10 == 0:
                    cnt += 1
                    _,_,_,precision_train, recall_train, f1_train = self.evaluate(X_train_batch, y_train_batch,
                                                                            predicts_train, id2char, id2label)
                    summary_writer_train.add_summary(train_summary, cnt)
                    print("Train iteration: %5d, train loss: %5d, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (iteration, loss_train, precision_train, recall_train, f1_train))

                    # validation
                if iteration % 100 == 0:
                    X_val_batch, y_val_batch = helper.nextRandomBatch(X_val, y_val, batch_size=self.batch_size)
                    y_val_weight_batch = 1 + np.array((y_train_batch == label2id['disorder']),float)
                    transition_batch = helper.getTransition(y_val_batch, self.num_classes)

                    loss_val, max_scores, max_scores_pre, length, val_summary = \
                        sess.run([
                            self.loss,
                            self.max_scores,
                            self.max_scores_pre,
                            self.length,
                            self.val_summary
                        ],
                            feed_dict={
                                self.targets_transition: transition_batch,
                                self.inputs: X_val_batch,
                                self.targets: y_val_batch,
                                self.targets_weight: y_val_weight_batch
                            })

                    predicts_val = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
                    pre_num, true_num, hit_num,precision_val, recall_val, f1_val = self.evaluate(X_val_batch, y_val_batch, predicts_val, id2char,
                                                                      id2label)
                    summary_writer_val.add_summary(val_summary, cnt)
                    print("Valid iteration:",iteration,"Valid loss:", loss_val, "Valid predicted num:%d, true num:%d, hit num:%d, precision: %.5f, valid recall: %.5f, valid f1: %.5f"%(pre_num, true_num, hit_num,precision_val, recall_val, f1_val))

                    if f1_val > self.max_f1:
                        self.max_f1 = f1_val
                        save_path = saver.save(sess, save_file)
                        print("saved the best model with f1: %.5f" % (self.max_f1))

    def test(self, sess, X_test, X_test_str, output_path, y_test=None):
        char2id, id2char = helper.loadMap("char2id")
        label2id, id2label = helper.loadMap("label2id")
        precision_mean = 0
        recall_mean = 0
        y_predicts = []
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print("number of iteration: " + str(num_iterations))
        with open(output_path, "wb") as outfile:
            for i in range(num_iterations):
                print("Test iteration: " + str(i + 1))
                results = []
                X_test_batch = X_test[i * self.batch_size: (i + 1) * self.batch_size]
                X_test_str_batch = X_test_str[i * self.batch_size: (i + 1) * self.batch_size]
                y_test_batch = y_test[i * self.batch_size: (i + 1) * self.batch_size]
                if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                    X_test_batch = list(X_test_batch)
                    X_test_str_batch = list(X_test_str_batch)
                    last_size = len(X_test_batch)
                    X_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_str_batch += [['x' for j in range(self.num_steps)] for i in
                                         range(self.batch_size - last_size)]
                    X_test_batch = np.array(X_test_batch)
                    X_test_str_batch = np.array(X_test_str_batch)
                    predicts, results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)
                    results = results[:last_size]
                else:
                    X_test_batch = np.array(X_test_batch)
                    predicts, results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)
                    results = results[:len(X_test_batch)]
                    predicts = predicts[:len(X_test_batch)]#最後一批
                y_predicts += predicts
                # precision_val, recall_val, f1_val = self.evaluate_test( y_test_batch, results, id2label)
            pre_num, true_num, hit_num, precision_test, recall_test, f1_test = self.evaluate(X_test, y_test, y_predicts, id2char, id2label)

            print("predicted num:%d, true num:%d, hit num:%d, test precision: %.5f, test recall: %.5f, test f1: %.5f" % ( pre_num, true_num, hit_num,precision_test, recall_test, f1_test))
            # print("mean test precision: %.5f, mean test recall: %.5f, test f1: %.5f" % (precision_mean, recall_mean, (2*precision_mean*recall_mean)/(precision_mean+recall_mean)))

    def predict(self, sess, X_predict, X_predict_str, output_path ):
        char2id, id2char = helper.loadMap("char2id")
        label2id, id2label = helper.loadMap("label2id")
        num_iterations = int(math.ceil(1.0 * len(X_predict) / self.batch_size))
        print("number of iteration: " + str(num_iterations))
        with open(output_path+'.ann', "w") as outfile:
            for i in range(num_iterations):
                print("predict iteration: " + str(i + 1))
                results = []
                X_predict_batch = X_predict[i * self.batch_size: (i + 1) * self.batch_size]
                X_predict_str_batch = X_predict_str[i * self.batch_size: (i + 1) * self.batch_size]
                if i == num_iterations - 1 and len(X_predict_batch) < self.batch_size:
                    X_predict_batch = list(X_predict_batch)
                    X_predict_str_batch = list(X_predict_str_batch)
                    last_size = len(X_predict_batch)
                    X_predict_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_predict_str_batch += [['None' for j in range(self.num_steps)] for i in
                                         range(self.batch_size - last_size)]
                    X_predict_batch = np.array(X_predict_batch)
                    X_predict_str_batch = np.array(X_predict_str_batch)
                    results = self.predictBatch(sess, X_predict_batch, X_predict_str_batch, id2label)
                    results = results[:last_size]
                else:
                    X_predict_batch = np.array(X_predict_batch)
                    predicts, results = self.predictBatch(sess, X_predict_batch, X_predict_str_batch, id2label)
                    results = results[:len(X_predict_batch)]
                rawtext=''
                count=0
                for j in range(len(X_predict_str_batch)):

                    try:
                        for word, label in zip(X_predict_str_batch[j], results[1][j]):
                            count += 1
                            merge_line = ''
                            merge_line = 'T' + str(count) + '\t' + str(label) + ' ' + str(len(rawtext)) + ' ' + str(
                                len(rawtext) + len(str(word))) + '\t' + str(word) + '\n'
                            rawtext += str(word) + ' '
                            if str(label) != 'others':
                                outfile.write(merge_line)
                                print(merge_line)
                    except Exception:
                        continue
                outfile.close()
                with open(output_path + '.txt', 'w') as f:
                    f.write(rawtext.strip())
                    f.close()

    def viterbi(self, max_scores, max_scores_pre, length, predict_size=128):
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][length[m]])
            # last_max_node = 0
            for t in range(1, length[m] + 1)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]   #reverse
            best_paths.append(path)
        return best_paths

    def predictBatch(self, sess, X, X_str, id2label):
        results = []
        length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores, self.max_scores_pre], feed_dict={self.inputs: X})
        predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
        for i in range(len(predicts)):
            #x = ''.join(X_str[i])
            y_pred = [id2label[val] for val in predicts[i] if val != self.num_classes and val != 0]
            #entitys = helper.extractEntity(x, y_pred)
            #results.append(entitys)
            results.append(y_pred)
        return predicts, results

    def evaluate_test(self, true_labels, pred_labels):
        hit_num = 0
        pred_num = 0
        true_num = 0
        hit_num += len(set(true_labels) & set(pred_labels))
        pred_num += len(set(pred_labels))
        true_num += len(set(true_labels))

        if pred_num != 0:
            precision = 1.0 * hit_num / true_num
        if true_num != 0:
            recall = 1.0 * hit_num / pred_num
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    def evaluate(self, X, y_true, y_pred, id2char, id2label):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        hit_num = 0
        pred_num = 0
        true_num = 0
        total_hit_labels = []
        total_true_labels = []
        total_pred_labels = []
        for i in range(len(y_true)):
            x = "\t".join([str(id2char[val]) for val in X[i]])
            y = [str(id2label[val]) for val in y_true[i]]
            y_hat = [id2label[val] for val in y_pred[i] if val != 0]  # result of predict
            true_labels = helper.extractEntity(x.split('\t'), y)
            pred_labels = helper.extractEntity(x.split('\t'), y_hat)
            hit_num += len(set(true_labels) & set(pred_labels))
            pred_num += len(set(pred_labels))
            true_num += len(set(true_labels))
            total_hit_labels += set(true_labels) & set(pred_labels)
            total_pred_labels += pred_labels
            total_true_labels += true_labels
        self.summary(total_true_labels, total_pred_labels,total_hit_labels, id2label)
        if pred_num != 0:
            precision = 1.0 * hit_num / pred_num
        if true_num != 0:
            recall = 1.0 * hit_num / true_num
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        return pred_num, true_num, hit_num, precision, recall, f1

    def summary(self, true_labels, pred_labels, hit_labels, id2label):

        labeldict = list(set(list(label for label in id2label.values())))
        with open('test_log.txt','a') as f:
            f.write('\n\n********'+time.strftime("%Y-%m-%d %X", time.localtime())+'********\n')
            log =''
            for label in labeldict:
                precision = -1.0
                recall = -1.0
                f1 = -1.0
                hit_num = 0
                pred_num = 0
                true_num = 0
                tru = list(filter(lambda x: x[0]== label, true_labels))
                pre = list(filter(lambda x: x[0] == label, pred_labels))
                hit =  list(filter(lambda x: x[0] == label, hit_labels))
                hit_num = len(hit)
                pred_num = len(pre)
                true_num = len(tru)
                if pred_num != 0:
                    precision = 1.0 * hit_num / pred_num
                if true_num != 0:
                    recall = 1.0 * hit_num / true_num
                if precision > 0 and recall > 0:
                    f1 = 2.0 * (precision * recall) / (precision + recall)
                print('\n********%s********\n' % label)
                print('predicted number:%d, true number:%d, hit number:%d' % (pred_num, true_num, hit_num))
                print('precision:%.5f,recall:%.5f,f1:%.5f' % (precision, recall, f1))
                f.write('%s,%d,%d,%d,%.5f,%.5f,%.5f\n' % (label,pred_num, true_num, hit_num,precision, recall, f1))

