from __future__ import division
import time
import random
import progressbar
import editdistance
import dynet as dy
import numpy as np

import util
import datasets
from defaults import SANITY_SIZE

from transducer import cost_actions, edit_cost_matrix

OPTIMIZERS = {'ADAM'    : #dy.AdamTrainer,
                          lambda m: dy.AdamTrainer(m, alpha=0.0005,
                                                   beta_1=0.9, beta_2=0.999, eps=1e-8),
              'MOMENTUM': dy.MomentumSGDTrainer,
              'SGD'     : dy.SimpleSGDTrainer,
              'ADAGRAD' : dy.AdagradTrainer,
              'ADADELTA': dy.AdadeltaTrainer}


def INVERSE_SIGMOID(k):
    # e.g. k = 12
    assert k >= 1, 'Value of rate of decay parameter inappropriate: k = {}'.format(k)
    return lambda epoch: k / (k + np.exp(epoch / k))

def EXPONENTIAL(k):
    # e.g. k = 0.975
    assert k > 0 and k < 1, 'Value of rate of decay parameter inappropriate: k = {}'.format(k)
    return lambda epoch: k ** epoch

SCHEDULED_SAMPLING_DECAYS = {'inverse_sigmoid': INVERSE_SIGMOID,
                             'exponential':     EXPONENTIAL,
                             # 'linear': lambda (c, k): (lambda epoch: max(initial_decay, k - c*epoch) )  # 0<=initial_decay<1
                             }


def internal_eval(batches, transducer, vocab,
                  previous_predicted_actions,
                  check_condition=True, name='train'):

    then = time.time()
    print 'evaluating on {} data...'.format(name)

    number_correct = 0.
    total_loss = 0.
    predictions = []
    pred_acts = []
    i = 0  # counter of samples
    for j, batch in enumerate(batches):
        dy.renew_cg()
        batch_loss = []
        for sample in batch:
            feats = sample.pos, sample.feats
            loss, prediction, predicted_actions = transducer.transduce(sample.lemma, feats, external_cg=True)
            ###
            predictions.append(prediction)
            pred_acts.append(predicted_actions)
            batch_loss.extend(loss)

            # evaluation
            correct_prediction = False
            if (prediction in vocab.word and vocab.word.w2i[prediction] == sample.word):
                correct_prediction = True
                number_correct += 1

            if check_condition:
                # display prediction for this sample if it differs the prediction
                # of the previous epoch or its an error
                if predicted_actions != previous_predicted_actions[i] or not correct_prediction:
                    #
                    print 'BEFORE:    ', datasets.action2string(previous_predicted_actions[i], vocab)
                    print 'THIS TIME: ', datasets.action2string(predicted_actions, vocab)
                    print 'TRUE:      ', sample.act_repr
                    print 'PRED:      ', prediction
                    print 'WORD:      ', sample.word_str
                    print 'X' if correct_prediction else 'V'
            # increment counter of samples
            i += 1
        batch_loss = -dy.average(batch_loss)
        total_loss += batch_loss.scalar_value()
        # report progress
        if j > 0 and j % 100 == 0: print '\t\t...{} batches'.format(j)

    accuracy = number_correct / i
    print '\t...finished in {:.3f} sec'.format(time.time() - then)
    return accuracy, total_loss, predictions, pred_acts


def internal_eval_beam(batches, transducer, vocab,
                  beam_width, previous_predicted_actions,
                  check_condition=True, name='train'):
    assert callable(getattr(transducer, "beam_search_decode", None)), 'transducer does not implement beam search.'
    then = time.time()
    print 'evaluating on {} data with beam search (beam width {})...'.format(name, beam_width)
    number_correct = 0.
    total_loss = 0.
    predictions = []
    pred_acts = []
    i = 0  # counter of samples
    for j, batch in enumerate(batches):
        dy.renew_cg()
        batch_loss = []
        for sample in batch:
            feats = sample.pos, sample.feats
            hypotheses = transducer.beam_search_decode(sample.lemma, feats, external_cg=True,
                                                       beam_width=beam_width)
            # take top hypothesis
            loss, loss_expr, prediction, predicted_actions = hypotheses[0]

            predictions.append(prediction)
            pred_acts.append(predicted_actions)
            batch_loss.append(loss)
            # sanity check: Basically, this often is wrong...
            #assert round(loss, 3) == round(loss_expr.scalar_value(), 3), (loss, loss_expr.scalar_value())

            # evaluation
            correct_prediction = False
            if (prediction in vocab.word and vocab.word.w2i[prediction] == sample.word):
                correct_prediction = True
                number_correct += 1
                if check_condition:
                    # compare to greedy prediction:
                    _, greedy_prediction, _ = transducer.transduce(sample.lemma, feats, external_cg=True)
                    if greedy_prediction != prediction:
                        print 'Beam! Target: ', sample.word_str
                        print 'Greedy prediction: ', greedy_prediction
                        print u'Complete hypotheses:'
                        for log_p, _, pred_word, pred_actions in hypotheses:
                            print u'Actions {}, word {}, -log p {:.3f}'.format(
                                datasets.action2string(pred_actions, vocab), pred_word, -log_p)

            if check_condition:
                # display prediction for this sample if it differs the prediction
                # of the previous epoch or its an error
                if predicted_actions != previous_predicted_actions[i] or not correct_prediction:
                    #
                    print 'BEFORE:    ', datasets.action2string(previous_predicted_actions[i], vocab)
                    print 'THIS TIME: ', datasets.action2string(predicted_actions, vocab)
                    print 'TRUE:      ', sample.act_repr
                    print 'PRED:      ', prediction
                    print 'WORD:      ', sample.word_str
                    print 'X' if correct_prediction else 'V'
            # increment counter of samples
            i += 1
        batch_loss = -np.mean(batch_loss)
        total_loss += batch_loss
        # report progress
        if j > 0 and j % 100 == 0: print '\t\t...{} batches'.format(j)

    accuracy = number_correct / i
    print '\t...finished in {:.3f} sec'.format(time.time() - then)
    return accuracy, total_loss, predictions, pred_acts

class TrainingSession(object):
    def __init__(self, model, transducer, vocab,
                 train_data, dev_data,
                 batch_size,
                 optimizer=None,
                 decbatch_size=None,
                 dev_batches=None):

        self.model = model
        self.transducer = transducer
        self.optimizer = OPTIMIZERS.get(optimizer, 'ADADELTA')
        self.trainer = None  # initialized only in training
        self.vocab = vocab

        # DATA and BATCHES
        self.train_data = train_data
        self.dev_data = dev_data
        self.dev_batches = dev_batches

        self.batch_size = batch_size
        # use different (larger) batch size for decoding
        self.decbatch_size = decbatch_size if decbatch_size else batch_size
        self.dev_len    = len(self.dev_data)
        self.train_len  = len(self.train_data)
        if self.dev_batches is None:
            self.dev_batches = [self.dev_data.samples[i:i+self.decbatch_size]
                for i in range(0, self.dev_len, self.decbatch_size)]

        sanity_size = min(SANITY_SIZE, len(self.train_data))
        self.sanity_batches = [self.train_data.samples[:sanity_size][i:i+self.decbatch_size]
            for i in range(0, sanity_size, self.decbatch_size)]

        print 'Decoding batch size is {}.'.format(self.decbatch_size)
        print 'Training batch size is {}.'.format(self.batch_size)
        print 'There are {} train and {} dev samples.'.format(self.train_len, self.dev_len)
        print 'There are {} train batches and {} dev batches.'.format(
            (self.train_len / self.batch_size) + 1, len(self.dev_batches))

        # BOOKKEEPING OF PREDICTED ACTIONS
        self.dev_predicted_actions = [None]*self.dev_len
        self.train_predicted_actions = [None]*sanity_size

        # PERFORMANCE METRICS
        # dev performance stats
        self.best_avg_dev_loss = 999.
        self.best_dev_accuracy = 0.
        self.best_dev_loss_epoch = 0
        self.best_dev_acc_epoch  = 0
        # train performance stats
        self.avg_loss = 0.
        self.best_train_accuracy = 0.

    def reload(self, path2model, tmp_model_path=None):
        print 'Trying to reload model from: {}'.format(path2model)
        self.model.populate(path2model)
        print 'Computing dev accuracy of the reloaded model...'
        # initialize dev stats from the pretrained model
        self.best_dev_accuracy, self.best_avg_dev_loss = \
            self.dev_eval(check_condition=False)
        print 'Dev accuracy, dev loss: ', self.best_dev_accuracy, self.best_avg_dev_loss
        self.best_dev_loss_epoch = -1
        self.best_dev_acc_epoch  = -1
        if tmp_model_path and tmp_model_path != path2model:
            self.model.save(tmp_model_path)
            print 'saved reloaded model as best model to {}'.format(tmp_model_path)

    def action2string(self, acts):
        return datasets.action2string(acts, self.vocab)

    def dev_eval(self, check_condition=True):
        # call internal_eval with dev batches
        dev_accuracy, avg_dev_loss, _, self.dev_predicted_actions = \
            internal_eval(self.dev_batches, self.transducer, self.vocab,
                          self.dev_predicted_actions,
                          check_condition=check_condition, name='dev')
        return dev_accuracy, avg_dev_loss

    def train_eval(self, check_condition=True):
        # call internal_eval with train batches
        train_dev_accuracy, avg_loss, _, self.train_predicted_actions = \
            internal_eval(self.sanity_batches, self.transducer, self.vocab,
                          self.train_predicted_actions,
                          check_condition=check_condition, name='train')
        return train_dev_accuracy, avg_loss

    def run_MLE_training(self, **kwargs):

        print 'Running MLE training...'
        l2 = kwargs.get('l2')
        if l2:
            print 'Using l2-regularization with parameter {}'.format(l2)

        self.model.save(kwargs['tmp_model_path'])
        print 'saved initial model to {}'.format(kwargs['tmp_model_path'])

        def MLE_batch_update(batch, *args):
            # How to update model parameters from
            # a batch of training samples with MLE?
            dy.renew_cg()
            batch_loss = []
            for sample in batch:
                feats = sample.pos, sample.feats
                loss, prediction, predicted_actions = self.transducer.transduce(
                    sample.lemma, feats, sample.actions, external_cg=True)
                batch_loss.extend(loss)
            batch_loss = -dy.average(batch_loss)
            if l2: batch_loss += l2 * self.transducer.l2_norm(with_embeddings=False)
            loss = batch_loss.scalar_value()  # forward
            batch_loss.backward()             # backward
            self.trainer.update()
            return loss

        self.run_training(MLE_batch_update, **kwargs)


    def run_il_training(self, **kwargs):

        print 'Running IL training...'
        l2 = kwargs.get('l2')
        if l2:
            print 'Using l2-regularization with parameter {}'.format(l2)
        k = kwargs['il_k']
        c = None  # @TODO Add for linear dicay: kwargs.get('c'), currently simply ignore it
        decay_schedule = SCHEDULED_SAMPLING_DECAYS.get(kwargs.get('il_decay'), INVERSE_SIGMOID)
        decay              = decay_schedule(k)
        rollout_mixin_beta = kwargs['il_beta']
        global_rollout     = kwargs['il_global_rollout']
        loss_expression    = kwargs['il_loss']
        bias_inserts       = kwargs['il_bias_inserts']
        pretrain_epochs    = kwargs['il_tforcing_epochs']
        optimal_oracle     = kwargs['il_optimal_oracle']

        print 'Using {} roll-in decay schedule with parameters: k={}{}. Will apply decay after {} epoch.'.format(
            decay_schedule, k, ', c = {}'.format(c) if c else '', pretrain_epochs)
        print ('Using {} loss and beta={} ({}) to mix reference and learned roll-outs. Reference policy is {}{}.'
               .format(loss_expression, rollout_mixin_beta,
                       'global' if global_rollout else 'local',
                       'optimal' if optimal_oracle else 'sub-optimal',
                       ' (insert bias in learned roll-outs)' if bias_inserts else ''))
        self.model.save(kwargs['tmp_model_path'])
        print 'saved initial model to {}'.format(kwargs['tmp_model_path'])
        verbose = kwargs['check_condition']

        def il_training_batch_update(batch, *args):
            # How to update model parameters from
            # a batch of training samples with il training?
            dy.renew_cg()
            epoch = args[0]
            e = 1 - decay(epoch-pretrain_epochs) if epoch >= pretrain_epochs else 0.
            if verbose and e: print 'Sampling probability = {:.3f}'.format(e)
            batch_loss = []
            for sample in batch:
                feats = sample.pos, sample.feats
                # @TODO This will fail if a target character has never been seen
                # in lemmas and parameter tying is not used!
                loss, prediction, predicted_actions = self.transducer.transduce(
                    lemma=sample.lemma,
                    feats=feats,
                    oracle_actions={'loss'               : loss_expression,
                                    'rollout_mixin_beta' : rollout_mixin_beta,
                                    'global_rollout'     : global_rollout,
                                    'target_word'        : sample.actions,
                                    # support for legacy, buggy experiments
                                    'optimal'            : optimal_oracle,
                                    'bias_inserts'       : bias_inserts},
                    sampling=e,
                    external_cg=True,
                    verbose=verbose)
                batch_loss.extend(loss)
            batch_loss = -dy.average(batch_loss)
            if l2: batch_loss += l2 * self.transducer.l2_norm(with_embeddings=False)
            loss = batch_loss.scalar_value()  # forward
            batch_loss.backward()             # backward
            self.trainer.update()
            return loss

        self.run_training(il_training_batch_update, **kwargs)


    def run_RL_training(self, **kwargs):

        print 'Running RL training...'
        #print 'Trainer attributes: ', self.trainer.__dict__

        sample_size = kwargs['sample_size']
        scale_neg = kwargs['scale_negative']
        beta_ned = kwargs['beta']
        use_baseline = kwargs['baseline']
        verbose = True if kwargs['check_condition'] else False

        print 'Will draw {} samples per training sample.'.format(sample_size)
        print 'Will use greedy baseline for reward correction.' if use_baseline else 'Will not use baseline reward correction.'
        print 'Will apply negative scaling of {}.'.format(scale_neg)

        def compute_reward(word, word_str, prediction):
            # `word` is an integer code,
            # `word_str` is the string corresponding to this code,
            # `prediction` is a string
            if (prediction in self.vocab.word and
                self.vocab.word.w2i[prediction] == word):
                # correct prediction
                reward = 1.
            else:
                # the smaller the distance the better
                #reward = -1*int(editdistance.eval(word_str, prediction))/len(word_str)
                reward = -beta_ned * editdistance.eval(word_str, prediction) / max(len(word_str), len(prediction))
            return reward

        def RL_batch_update(batch, *args):

            dy.renew_cg()
            batch_loss = []
            rewards = []
            for sample in batch:

                lemma = sample.lemma
                word = sample.word
                word_str = sample.word_str
                feats = sample.pos, sample.feats

                if use_baseline:
                    # BASELINE PREDICTION
                    _, prediction_b, predicted_actions_b = \
                        self.transducer.transduce(lemma, feats, external_cg=True)
                    # BASELINE REWARD
                    reward_b = compute_reward(word, word_str, prediction_b)

                for _ in xrange(sample_size):

                    # SAMPLING-BASED PREDICTION
                    loss, prediction, predicted_actions = \
                        self.transducer.transduce(lemma, feats, sampling=True, external_cg=True)
                    # SAMPLING-BASED REWARD
                    reward = compute_reward(word, word_str, prediction)

                    if use_baseline:
                        sample_reward = reward - reward_b
                    else:
                        sample_reward = reward

                    if verbose and use_baseline and sample_reward and reward == 1.:
                        # i.e. sampling produced a correct prediction via a sequence of actions
                        # different from the argmax approach of the baseline.
                        assert predicted_actions != predicted_actions_b
                        print (u'Correct prediction by sampling for {}, {}:\n'
                                'Sampling: {}\t{}\n'
                                'Baseline: {}\t{}\n'.format(
                            sample.lemma_str, sample.feat_str,
                            prediction, self.action2string(predicted_actions),
                            prediction_b, self.action2string(predicted_actions_b)))

                    if scale_neg and sample_reward < 0:
                        sample_reward = scale_neg*sample_reward

                    if sample_reward:
                        rewards.append(sample_reward)
                        batch_loss.append(-dy.average(loss))
                    #print 'word, prediction_b, prediction: ', word_str, prediction_b, prediction
                    #print 'reward_b, reward, sample_reward: ', reward_b, reward, sample_reward
                    #print 'word, prediction: ', word_str, prediction
                    #print 'reward, sample_reward: ', reward, sample_reward
            if not use_baseline:
                rewards = np.array(rewards)
                if all(rewards == rewards[0]):
                    # all rewards are the same, assume we sampled the same thing from peaked distribution.
                    # Then don't update.
                    batch_loss = []
                else:
                    # mean normalize rewards
                    rewards = (rewards - np.mean(rewards)) / (np.max(rewards) - np.min(rewards))
            if batch_loss:
                num_nonzero_grad = len(batch_loss)
                # dy.concatenate(batch_loss) => make a vector out of Python list of dynet scalars
                # dy.cdiv => element-wise division, then .scalar_value() to get a scalar
                # division is not implemented.
                batch_loss = dy.cdiv(dy.dot_product(dy.inputVector(rewards),
                    dy.concatenate(batch_loss)), dy.scalarInput(num_nonzero_grad))
                loss = batch_loss.scalar_value()  # forward
                batch_loss.backward()
                self.trainer.update()
                if verbose:
                    print 'Batch loss, batch reward: ', loss, sum(rewards)/num_nonzero_grad
                    print 'Batch reward mean (std): %.3f (%.3f)' % (np.mean(rewards), np.std(rewards))
            else:
                loss = 0
                if verbose:
                    print 'Batch loss is zero.'
            return loss

        self.run_training(RL_batch_update, **kwargs)


    def run_MRT_training(self, **kwargs):

        print 'Running MRT training with sampling...'
        #print 'Trainer attributes: ', self.trainer.__dict__

        sample_size = kwargs['sample_size']
        alpha_p  = kwargs['alpha']  #0.05
        beta_ned = kwargs['beta']
        verbose = True if kwargs['check_condition'] else False

        action_penalty = 0  # 0.2


        print 'Alpha parameter will be {}'.format(alpha_p)
        print 'Beta scaling factor for NED will be {}'.format(beta_ned)
        print 'Sample size will be {}'.format(sample_size)

        def compute_reward(word, word_str, prediction):
            # `word` is an integer code,
            # `word_str` is the string corresponding to this code,
            # `prediction` is a string

            # This is a normalized edit distance cost
            # The better the prediction, the lower the reward.
            if (prediction in self.vocab.word and
                self.vocab.word.w2i[prediction] == word):
                # correct prediction
                reward = -1.
            else:
                # the smaller the distance the better
                reward = beta_ned * editdistance.eval(word_str, prediction) / max(len(word_str), len(prediction))
            return reward

        def MRT_batch_update(batch, epoch):

            dy.renew_cg()

            alpha = dy.scalarInput(alpha_p)

            batch_loss = []
            rewards = []
            for sample in batch:

                lemma = sample.lemma
                word = sample.word
                word_str = sample.word_str
                feats = sample.pos, sample.feats
                actions = sample.actions

                # ORACLE PREDICTION
                #loss, prediction_b, predicted_actions_b = \
                gold_loss, _, _ = \
                    self.transducer.transduce(lemma, feats, actions, external_cg=True)
                gold_loss = dy.esum(gold_loss)
                #if gold_loss.scalar_value() < -50.:  # Sum log P
                #    print 'Dangerously low prob of gold action seq: ', gold_loss.scalar_value(), word_str
                #    hypotheses = []
                #else:
                #    hypotheses = [ (_, gold_loss, word_str, actions) ]

                # BEAM-SEARCH-BASED PREDICTION
                #hypotheses += self.transducer.beam_search_decode(lemma, feats, external_cg=True,
                #                                                 beam_width=beam_width)
                if action_penalty:
                    # we will add edit cost to penalize long and intuitively wasteful actions
                    sample_actions = [cost_actions(actions)]
                sample_rewards = [-1.]
                sample_losses = [gold_loss]
                predictions = [word_str]
                seen_predicted_acts = {tuple(actions)}
                #for _, loss, prediction, predicted_actions in hypotheses:
                for _ in range(sample_size):
                    loss, prediction, predicted_actions = \
                        self.transducer.transduce(lemma, feats, sampling=True, external_cg=True)
                    predicted_actions = tuple(predicted_actions)
                    if predicted_actions in seen_predicted_acts:
                        #if verbose: print 'already sampled this action sequence: ', predicted_actions
                        continue
                    loss = dy.esum(loss)
                    if loss.scalar_value() < -20:  # log P
                        continue
                    else:
                        seen_predicted_acts.add(predicted_actions)
                #for _, loss, prediction, predicted_actions in hypotheses:

                    # COMPUTE REWARDS
                    reward = compute_reward(word, word_str, prediction)

                    sample_rewards.append(reward)
                    sample_losses.append(loss)
                    predictions.append(prediction)
                    if action_penalty:
                        sample_actions.append(cost_actions(predicted_actions))

                # SCALE & RENORMALIZE: (these are log P)
                if len(sample_rewards) == 1 and sample_rewards[0] == -1.:
                    if verbose: print 'Nothing to update with.'
                    continue
                else:
                    if action_penalty:
                        #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
                        # min-max scaling to [0, 1]
                        #print 'Sampled actions: ', sample_actions
                        if len(set(sample_actions)) == 1:
                            sample_actions = np.zeros_like(sample_actions)
                        else:
                            sample_actions = np.array(sample_actions)
                            min_score = np.min(sample_actions)
                            max_score = np.max(sample_actions)
                            sample_actions = (sample_actions - min_score) / (max_score - min_score)
                        #print 'Sampled actions: ', sample_actions
                        sample_rewards = (1 - action_penalty) * np.array(sample_rewards) + action_penalty * sample_actions
                        #print 'Sampled rewards: ', sample_rewards
                    #if verbose: print 'sample_losses', sample_losses
                    sample_losses = dy.concatenate(sample_losses)
                    sample_rewards = dy.inputVector(sample_rewards)
                    q_unnorm = dy.pow(dy.exp(sample_losses), alpha)
                    q = dy.cdiv(q_unnorm, dy.sum_elems(q_unnorm))

                    if verbose:
                        print 'q', q.npvalue()
                        print 'sample_rewards', sample_rewards.npvalue()
                        print 'word', word_str
                        print 'predictions: ', u', '.join(predictions)
                    batch_loss.append(dy.dot_product(q, sample_rewards))
            if batch_loss:
                batch_loss = dy.esum(batch_loss)
                loss = batch_loss.scalar_value()  # forward
                try:
                    batch_loss.backward()
                    self.trainer.update()
                except Exception, e:
                    print 'Batch loss: ', loss
                    print 'q', q.npvalue()
                    print 'q_unnorm', q_unnorm.npvalue()
                    print 'gold_loss', gold_loss.scalar_value()
                    print 'sample_rewards', sample_rewards.npvalue()
                    print 'word', word_str
                    print 'predictions: ', u', '.join(predictions)
                    raise e
                if verbose: print 'Batch loss: ', loss
            else:
                if verbose: print 'Batch loss is zero.'
                loss = 0.
            return loss

        self.run_training(MRT_batch_update, **kwargs)


    def run_training(self,
                     batch_update,
                     epochs,
                     max_patience,
                     pick_best_accuracy,
                     dropout,
                     log_file_path,
                     tmp_model_path,
                     check_condition,
                     train_until_accuracy=None,
                     optimizer=None,
                     **kwargs):

        if optimizer is None:
            optimizer = self.optimizer
        self.trainer = optimizer(self.model)
        print 'Initialized trainer with: {}.'.format(optimizer)

        if dropout:
            print 'Using dropout of {}.'.format(dropout)
        else:
            print 'Not using dropout.'

        if check_condition == 2:  # max verbose flag... @TODO
            check_condition = lambda e: e > 0
        else:
            check_condition = lambda e: False


        if train_until_accuracy and 0 < train_until_accuracy <= 1.:
            epochs = 10000
            max_patience = 10000
            print 'Will train until training set accuracy of {} is reached.'.format(train_until_accuracy)
        else:
            print 'Will train for a maximum of {} epochs with patience of {}.'.format(epochs, max_patience)
        print 'Will early stop based on dev {}.'.format('accuracy' if pick_best_accuracy else 'loss')

        # PROGRESS BAR INIT
        widgets = [progressbar.Bar('>'), ' ', progressbar.ETA()]
        train_progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=epochs).start()

        # LOG FILE INIT
        with open(log_file_path, 'a') as a:
            a.write('epoch\tavg_loss\ttrain_accuracy\tdev_accuracy\n')

        patience = 0

        for epoch in xrange(epochs):

            print 'training...'
            then = time.time()

            train_loss = 0.

            train = self.train_data.samples
            random.shuffle(train)
            batches = [train[i:i+self.batch_size] for i in range(0, self.train_len, self.batch_size)]
            print 'Number of train batches: {}.'.format(len(batches))

            # ENABLE DROPOUT
            if dropout: self.transducer.set_dropout(dropout)

            for j, batch in enumerate(batches):
                train_loss += batch_update(batch, epoch)
                if j > 0 and j % 100 == 0: print '\t\t...{} batches'.format(j)
            print '\t\t...{} batches'.format(j)

            # DISABLE DROPOUT AFTER TRAINING
            if dropout: self.transducer.disable_dropout()
            print '\t...finished in {:.3f} sec'.format(time.time() - then)
            self.avg_loss = train_loss / self.train_len
            print 'Average train loss: ', self.avg_loss

            # EVALUATE MODEL ON SUBSET OF TRAIN (SANITY)
            train_accuracy, avg_loss = self.train_eval(check_condition(epoch))
            if train_accuracy > self.best_train_accuracy:
                self.best_train_accuracy = train_accuracy

            patience += 1

            # EVALUATE MODEL ON DEV
            dev_accuracy, avg_dev_loss = self.dev_eval(check_condition(epoch))

            if dev_accuracy > self.best_dev_accuracy:
                self.best_dev_accuracy = dev_accuracy
                self.best_dev_acc_epoch = epoch
                # using dev acc for early stopping
                print 'Found best dev accuracy so far {:.7f}'.format(self.best_dev_accuracy)
                if pick_best_accuracy: patience = 0

            if avg_dev_loss < self.best_avg_dev_loss:
                self.best_avg_dev_loss = avg_dev_loss
                self.best_dev_loss_epoch = epoch
                # using dev loss for early stopping
                print 'Found best dev loss so far {:.7f}'.format(self.best_avg_dev_loss)
                if not pick_best_accuracy: patience = 0

            if patience == 0:
                # patience has been reset to 0, so save currently best model
                self.model.save(tmp_model_path)
                print 'saved new best model to {}'.format(tmp_model_path)

            print ('epoch: {} train loss: {:.4f} dev loss: {:.4f} dev acc: {:.4f} '
               'train acc: {:.4f} best train acc: {:.4f} best dev acc: {:.4f} (epoch {}) '
               'best dev loss: {:.7f} (epoch {}) patience = {}').format(
               epoch, self.avg_loss, avg_dev_loss, dev_accuracy, train_accuracy,
               self.best_train_accuracy, self.best_dev_accuracy, self.best_dev_acc_epoch,
               self.best_avg_dev_loss, self.best_dev_loss_epoch, patience)

            # LOG LATEST RESULTS
            with open(log_file_path, 'a') as a:
                a.write("{}\t{}\t{}\t{}\n".format(epoch, self.avg_loss, train_accuracy, dev_accuracy))

            if patience == max_patience:
                print 'out of patience after {} epochs'.format(epoch + 1)
                train_progress_bar.finish()
                break
            if train_until_accuracy and train_accuracy > train_until_accuracy:
                print 'reached required training accuracy level of {}'.format(train_until_accuracy)
                train_progress_bar.finish()
                break

            # UPDATE PROGRESS BAR
            train_progress_bar.update(epoch)


def withheld_data_eval(name, batches, transducer, vocab, beam_widths,
                       pred_path, gold_path, sigm2017format):

    """Runs internal greedy and beam-search evaluations as well as
       launches external eval script. Returns greedy accuracy (hm...?)"""

    # GREEDY PREDICTIONS FROM THIS MODEL
    greedy_accuracy, _, predictions, _ = internal_eval(batches,
        transducer, vocab, None, check_condition=False, name=name)
    if greedy_accuracy > 0:
        print '{} accuracy: {}'.format(name, greedy_accuracy)
    else:
        print 'Possibly covered test data. Accuracy zero.'
    # write out greedy predictions and scores
    util.external_eval(pred_path('greedy'), gold_path, batches, predictions, sigm2017format)

    # BEAM-SEARCH-BASED PREDICTIONS FROM THIS MODEL
    if beam_widths:
        print '\nDecoding with beam search...'
        #import hacm, hacm_sub, hard
        if not callable(getattr(transducer, "beam_search_decode", None)):
            print 'Transducer does not implement beam search.'
            raise NotImplementedError
            #isinstance(transducer, hacm_sub.MinimalTransducer):
        else:
            for beam_width in beam_widths:
                accuracy, _, predictions, _ = internal_eval_beam(batches,
                    transducer, vocab, beam_width, None, check_condition=False, name=name)
                if accuracy > 0:
                    print 'beam-{} accuracy {}'.format(beam_width, accuracy)
                else:
                    print 'Zero accuracy.'
                # write out predictions and scores more specifically
                beam_path = pred_path('beam' + str(beam_width))
                util.external_eval(beam_path, gold_path, batches, predictions, sigm2017format)

    return greedy_accuracy


def dev_external_eval(batches, transducer, vocab, paths,
                      data_arguments, model_arguments, optim_arguments):

    accuracy =  withheld_data_eval("dev", batches, transducer, vocab, optim_arguments['beam-widths'],
                       paths['dev_output'], paths['dev_path'], data_arguments['sigm2017format'])
    # WRITE STATS TO FILE (NOT IN TEST EXTERNAL EVAL)
    util.write_stats_file(accuracy, paths, data_arguments, model_arguments, optim_arguments)


def test_external_eval(batches, transducer, vocab, paths, beam_widths, sigm2017format):

    withheld_data_eval("test", batches, transducer, vocab, beam_widths,
                       paths['test_output'], paths['test_path'], sigm2017format)
