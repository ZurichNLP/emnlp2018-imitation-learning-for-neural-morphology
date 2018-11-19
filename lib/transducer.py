from __future__ import division

from collections import Counter

import dynet as dy
import numpy as np

from defaults import COPY, DELETE, BEGIN_WORD, END_WORD, UNK, MAX_ACTION_SEQ_LEN
from stack_lstms import Encoder
from datasets import action2string, lemma2string

NONLINS = {'tanh' : dy.tanh, 'ReLU' : dy.rectify}

def log_softmax_costs(logits, costs=None, valid_actions=None):
    """Compute log softmax-margin with arbitrary costs."""
    if costs is not None:
        # each action gets a cost, the higher the overall score the better.
        # Typically, when adding `costs`, no `valid_actions` are passed.
        logits += dy.inputVector(costs)
    return dy.log_softmax(logits, restrict=valid_actions)


def log_sum_softmax_margin_loss(indices, logits, logits_len, costs=None, valid_actions=None, verbose=False):
    """Compute dynamic-oracle log softmax-margin loss with arbitrary costs.
       Args:
           indices: Array-like object for selection of cost-augmented logits
               to be maximized."""
    # assumed: either valid_actions or costs, the latter should typically incorporate
    # validity of actions.
    if valid_actions is not None:
        try:
            costs = - np.ones(logits_len) * np.inf
            costs[valid_actions] = 0.
        except Exception, e:
            print "np.ones_like(logits), valid_actions, costs: ", np.ones_like(logits), valid_actions, costs
            raise e
    if costs is not None:
        if verbose == 2: print 'Indices, costs: ', indices, costs
        logits += dy.inputVector(costs)
    log_sum_selected_terms = dy.logsumexp([dy.pick(logits, index=e) for e in indices])
    normalization_term = dy.logsumexp([l for l in logits])
    return log_sum_selected_terms - normalization_term


def log_sum_softmax_loss(indices, logits, logits_len, valid_actions=None, verbose=False):
    """Compute dynamic-oracle negative log loss.
       Args:
           indices: Array-like object for selection of logits to be maximized."""
    # @TODO could just build cost vector from `valid_actions` and call `log_sum_softmax_margin_loss`
    if valid_actions is not None:
        # build cost vector expressing action invalidity
        costs = - np.ones(logits_len) * np.inf
        costs[valid_actions] = 0.
        if verbose == 2: print 'Indices, costs: ', indices, costs
        # add it to logits
        logits += dy.inputVector(costs)
    log_sum_selected_terms = dy.logsumexp([dy.pick(logits, index=e) for e in indices])
    normalization_term = dy.logsumexp([l for l in logits])
    return log_sum_selected_terms - normalization_term


def cost_sensitive_reg_loss(indices, logits, logit_len, costs=None, valid_actions=None, verbose=False):
    """Computes MSE loss over actions thereby solving cost-sensitive multiclass classification.
       Takes the same arguments as other losses for dynamic oracles for compatibility.
       The lower the cost the better, therefore all invalid actions (-np.inf) are remapped to
       large cost numbers."""
    # map costs between 0 and 1, with a margin for invalid actions
    costs[np.isinf(costs)] = np.max(costs) + 1
    costs = costs / np.max(costs) # given that 0. costs are in there, we end up perfectly in [0, 1]
    # NB! Note the minus to cancel out the minus added in the batch loss
    #if verbose and np.random.rand() > 0.99: print 'Costs, logits: ', costs, logits.npvalue()#[valid_actions]
    return -dy.squared_distance(logits, dy.inputVector(costs))
    #return -dy.squared_distance(dy.concatenate([logits[v] for v in valid_actions]),  # ...
    #                            dy.inputVector(costs))

def sample(log_probs_np, alpha=1.):
    """Sample an action from log-probability distribution (numpy array).
    Apply `alpha` hyper-parameter for distribution smoothing.
    """
    dist = np.exp(log_probs_np)**alpha
    dist = dist / np.sum(dist)
    # sample according to softmax
    rand = np.random.rand()
    for action, p in enumerate(dist):
        rand -= p
        if rand <= 0: break
    return action

def cost_actions(actions, del_cost=1, ins_cost=1, copy_cost=0):
    return sum(0 if a == COPY else count * del_cost if a == DELETE else count * ins_cost
               for a, count in Counter(actions).iteritems()) 

def edit_cost_matrix(source, target, del_cost=1, ins_cost=1, sub_cost=2, copy_cost=0):
    len_source_add1 = len(source) + 1
    len_target_add1 = len(target) + 1
    cost_matrix = np.zeros((len_source_add1, len_target_add1), dtype=np.int_)
    for i in range(1, len_source_add1):  # over rows
        cost_matrix[i, 0] = i
    for j in range(1, len_target_add1):  # over columns
        cost_matrix[0, j] = j
    for i in range(1, len_source_add1):
        for j in range(1, len_target_add1):
            dg_cost = copy_cost if source[i-1] == target[j-1] else sub_cost
            cost_matrix[i, j] = min(cost_matrix[i-1, j] + del_cost,
                                    cost_matrix[i, j-1] + ins_cost,
                                    cost_matrix[i-1, j-1] + dg_cost)
    return cost_matrix


def oracle_with_rollout(word, target_word, rest_of_input, valid_actions,
                        rollout, vocab, optimal=False, bias_inserts=False,
                        errors=None, verbose=False, del_cost=1, ins_cost=1,
                        copy_cost=0, accuracy_error_cost=5.):
    """Given the word form constructed so far, the target word, the buffer,
    and set of valid actions, what are the next optimal actions and the cost
    of all the actions? Under gold rollout, an action is optimal if the cost
    of taking it is the lowest, assuming that all future actions are optimal
    too. Biasing inserts in model roll-outs (due to a bug) gained performance."""

    bias_inserts_on = bias_inserts and np.random.rand() > 0.5
    if verbose:
        if rollout: print 'Rolling out with model...'
        if bias_inserts_on: print 'Will use bias inserts.'

    len_target_word = len(target_word)
    rest_of_input = rest_of_input[:-1]  # discount END WORD! @TODO undo choice of word wrapping
    len_word = len(word)
    
    if optimal:  # errors indicate that we use optimal reference policy
        if errors:
            # errors account for all possible errors except for last character
            num_errors = len(errors)
            if verbose:
                print u'Word contains at least {} errors: {}, {}, {}'.format(num_errors,
                    ''.join(word[:-1]) + '(' + word[-1] + ')',
                    ''.join([c for i, c in enumerate(word[:-1]) if i not in errors]) + '(' + word[-1] + ')',
                    action2string(target_word, vocab)), errors
            len_word -= num_errors
        try:
            if len_word and (len_word > len_target_word or                          # overgenerated !
                             word[-1] != vocab.char.i2w[target_word[len_word-1]]):  # generated a wrong char
                if verbose:
                    if len_word > len_target_word:
                        message = ''.join(word), action2string(target_word, vocab)
                    else:
                        message = word[-1], vocab.char.i2w[target_word[len_word-1]]
                    print u'Last action resulted in error: {}, {}'.format(*message)
                # there was an error, so in the following, ignore the last
                # generated char in accordance to the optimal policy, i.e.
                len_word -= 1
                errors.add(len(word)-1)
        except Exception, e:
            print 'len_word, word, target word: ', len_word, ''.join(word), action2string(target_word, vocab)
            raise e

    # (i) incorporate action validity into costs:
    costs = - np.ones(vocab.act_train) * np.inf
    # valid but suboptimal actions get high costs, e.g. actions leading to
    # wrong accuracy. @TODO This should be dependent on e.g. levenshtein
    costs[valid_actions] = accuracy_error_cost
 
    if len_word >= len_target_word:
        if DELETE in valid_actions:
            # maybe the buffer is still not empty
            optimal_actions = [DELETE]
            costs[END_WORD] = 0.
        else:
            assert END_WORD in valid_actions
            optimal_actions = [END_WORD]
            costs[END_WORD] = 0.
    else:
        # assume no sampling, therefore we are in edit distance
        # cost matrix. The lowest cost in is [len_word+1, len_target_word+1]
        # and action history defines position in the cost matrix. All costs
        # are then read off the cost matrix: INSERT(top_of_butter), DELETE,
        # COPY. All actions leading to an accuracy error get a -np.inf cost (?).
        # Optimal cost is (min_edit_distance - current_cost). Return optimal
        # cost actions and costs for all actions.
        target_char_i = target_word[len_word] # next target char, unicode => Works because of param_tying!!!
        target_char = vocab.char.i2w[target_char_i]
        top_of_buffer = rest_of_input[0] if len(rest_of_input) > 0 else END_WORD
        actions = []  # these actions are on a path to correct prediction
        action_costs = []  # their costs
        if DELETE in valid_actions:
            actions.append(DELETE)
            if rollout:
                _, prediction, predicted_actions = rollout(DELETE)
                # give cost to entire prediction. Alternatives would include:
                #  * computing cost from this DELETE action on (i.e. like dynamic oracle does),
                #  * take into account accuracy (dynamic oracle does not).
                cost = cost_actions(predicted_actions)
                if verbose == 2:
                    # prediction, predicted actions, cost
                    print u'DELETE COST (pred.): {}, {}, {}'.format(
                        prediction, action2string(predicted_actions, vocab), cost)
            else:
                cost = del_cost + edit_cost_matrix(rest_of_input[1:],  # delete one symbol
                                                   target_word[len_word:])[-1, -1]
                if verbose == 2:
                    # rest of lemma, rest of target, cost
                    print u'DELETE COST (ref.): {}, {}, {}'.format(action2string(rest_of_input[1:], vocab),
                           action2string(target_word[len_word:], vocab), cost)
            action_costs.append(cost)

        if COPY in valid_actions and target_char_i == top_of_buffer:
            # if valid, copy is on a path to target
            actions.append(COPY)
            if rollout:
                _, prediction, predicted_actions = rollout(COPY)
                cost = cost_actions(predicted_actions)
                if verbose == 2:
                    print u'COPY COST (pred.): {}, {}, {}'.format(
                        prediction, action2string(predicted_actions, vocab), cost)
            else:
                cost = copy_cost + edit_cost_matrix(rest_of_input[1:],  # delete one symbol
                                                    target_word[len_word+1:])[-1, -1]  # insert this symbol
                if verbose == 2:
                    print u'COPY COST (ref.): {}, {}, {}'.format(action2string(rest_of_input[1:], vocab),
                           action2string(target_word[len_word+1:], vocab), cost)
            action_costs.append(cost)
            
        if target_char in vocab.act.w2i:  # if such an action exists ...
            # if target char can be inserted by a corresponding insert action, allow that
            # @TODO speed this up by not going from dictionaries
            insert_target_char = vocab.act.w2i[target_char]
            actions.append(insert_target_char)
            if rollout:
                # @TODO BUG: SCORED WITH ROLLOUT COPY !!!
                _, prediction, predicted_actions = rollout(COPY if bias_inserts_on else insert_target_char)
                cost = cost_actions(predicted_actions)
                if verbose == 2:
                    print u'INSERT COST (pred.): {}, {}, {}'.format(
                        prediction, action2string(predicted_actions, vocab), cost)
            else:
                if bias_inserts_on:   # ENCOURAGE WITH ORACLE INSERTS
                    cost = copy_cost + edit_cost_matrix(rest_of_input[1:],  # delete one symbol
                                                        target_word[len_word+1:])[-1, -1]  # insert this symbol
                else:
                    cost = ins_cost + edit_cost_matrix(rest_of_input,
                                                       target_word[len_word+1:])[-1, -1]  # insert one symbol
                if verbose == 2:
                    print u'INSERT COST (ref.): {}, {}, {}'.format(action2string(rest_of_input, vocab),
                           action2string(target_word[len_word+1:], vocab), cost)
            action_costs.append(cost)

        if verbose == 2:
            print 'Target char:', target_char_i, target_char
            print 'Actions, action costs:', action2string(actions, vocab), action_costs
            print 'Top of the buffer:', top_of_buffer, action2string([top_of_buffer], vocab)
        
        # minimal cost according to gold oracle:
        optimal_cost = np.min(action_costs)
        optimal_actions = []
        for action, cost in zip(actions, action_costs):
            if cost == optimal_cost:
                optimal_actions.append(action)
            costs[action] = cost - optimal_cost

    if verbose == 2: 
        print 'Word:', u''.join(word)
        print 'Target word:', action2string(target_word, vocab)
        print 'Rest of input:', action2string(rest_of_input, vocab)
        print 'Valid actions:', valid_actions, action2string(valid_actions, vocab)
        print 'Optimal actions:', optimal_actions, action2string(optimal_actions, vocab)
        print 'Costs:', costs
    return optimal_actions, costs


def oracle(top_of_buffer, word, target_word, valid_actions, vocab):
    """Given the word form constructed so far, the target word, the buffer,
    and the set of valid actions, what are the next optimal actions?
    Optimality means that an action is on a path to target word. This
    does not take into account cost of an action, e.g. deleting all input
    and then inserting the whole target sequence is one path to target
    word (=> optimal), but it is costly, e.g. in terms of edit
    distance."""
    #print 'Top of buffer:', top_of_buffer, vocab.char.i2w[top_of_buffer]
    len_word = len(word)
    len_target_word = len(target_word)
    if len_word >= len_target_word:
        # target has been produced, otherwise we are done
        # @TODO when sampling, force transduction to terminate:
        # "greater than" is a hack to stop the transduction
        if DELETE in valid_actions:
            # maybe the buffer is still not empty
            actions = [DELETE]
        else:
            assert END_WORD in valid_actions
            actions = [END_WORD]
    else:
        actions = []
        try:
            target_char = target_word[len_word] # next target char, unicode
        except Exception, e:
            print u''.join(word), target_word, len_word
            raise e
        target_char_i = vocab.char.w2i[target_char]
        if DELETE in valid_actions:
            actions.append(DELETE)
        if COPY in valid_actions and target_char_i == top_of_buffer:
            # if valid, copy is on a path to target
            actions.append(COPY)
        if target_char in vocab.act.w2i:
            # if target char can be inserted by a corresponding insert action, allow that
            # @TODO speed this up by not going from dictionaries
            actions.append(vocab.act.w2i[target_char])
        #print 'Target char:', target_char_i, target_char

    #print 'Word:', u''.join(word)
    #print 'Target word:', target_word
    #print 'Valid actions:', valid_actions, action2string(valid_actions, vocab)
    #print 'Optimal actions: ', actions, action2string(actions, vocab)
    return actions

class Transducer(object):
    def __init__(self, model, vocab, char_dim=100, action_dim=100, feat_dim=20,
                 enc_hidden_dim=200, enc_layers=1, dec_hidden_dim=200, dec_layers=1,
                 vanilla_lstm=False, mlp_dim=0, nonlin='ReLU', lucky_w=55,
                 double_feats=False, param_tying=False, pos_emb=True, 
                 avm_feat_format=False, **kwargs):
        
        self.CHAR_DIM       = char_dim
        self.ACTION_DIM     = action_dim
        self.FEAT_DIM       = feat_dim
        self.ENC_HIDDEN_DIM = enc_hidden_dim
        self.ENC_LAYERS     = enc_layers
        self.DEC_HIDDEN_DIM = dec_hidden_dim
        self.DEC_LAYERS     = dec_layers
        self.LSTM           = dy.VanillaLSTMBuilder if vanilla_lstm else dy.CoupledLSTMBuilder
        self.MLP_DIM        = mlp_dim
        self.NONLIN         = NONLINS.get(nonlin, 'ReLU')
        self.LUCKY_W        = lucky_w
        self.double_feats   = double_feats
        self.param_tying    = param_tying
        self.pos_emb        = pos_emb
        self.avm_feat_format = avm_feat_format

        self.vocab = vocab

        # indices separating train elements from dev/test elements
        self.NUM_CHARS = self.vocab.char_train
        self.NUM_FEATS = self.vocab.feat_train
        self.NUM_POS   = self.vocab.pos_train
        self.NUM_ACTS  = self.vocab.act_train
        # an enumeration of all encoded insertions
        self.INSERTS   = range(self.vocab.number_specials, self.NUM_ACTS)
        
        # report stats
        print u'{} actions: {}'.format(self.NUM_ACTS,
            u', '.join(self.vocab.act.keys()))
        print u'{} features: {}'.format(self.NUM_FEATS,
            u', '.join(self.vocab.feat.keys()))
        print u'{} lemma chars: {}'.format(self.NUM_CHARS,
            u', '.join(self.vocab.char.keys()))

        if self.avm_feat_format:
            self.NUM_FEAT_TYPES = self.vocab.feat_type_train
            print u'{} feature types: {}'.format(self.NUM_FEAT_TYPES,
                u', '.join(self.vocab.feat_type.keys()))
            if self.pos_emb:
                print 'Assuming AVM features, therefore no specialized pos embedding'
                self.pos_emb = False
        
        self._build_model(model)
        # for printing
        self.hyperparams = {'CHAR_DIM'       : self.CHAR_DIM,
                            'FEAT_DIM'       : self.FEAT_DIM,
                            'ACTION_DIM'     : self.ACTION_DIM if not self.param_tying else self.CHAR_DIM,
                            'ENC_HIDDEN_DIM' : self.ENC_HIDDEN_DIM,
                            'ENC_LAYERS'     : self.ENC_LAYERS,
                            'DEC_HIDDEN_DIM' : self.DEC_HIDDEN_DIM,
                            'DEC_LAYERS'     : self.DEC_LAYERS,
                            'LSTM'           : self.LSTM,
                            'MLP_DIM'        : self.MLP_DIM,
                            'NONLIN'         : self.NONLIN,
                            'PARAM_TYING'    : self.param_tying,
                            'POS_EMB'        : self.pos_emb,
                            'AVM_FEATS'      : self.avm_feat_format}

    def _features(self, model):
        # trainable embeddings for characters and actions
        self.CHAR_LOOKUP = model.add_lookup_parameters((self.NUM_CHARS, self.CHAR_DIM))
        if self.param_tying:
            self.ACT_LOOKUP = self.CHAR_LOOKUP
            print 'NB! Using parameter tying: Chars and actions share embedding matrix.'
        else:
            self.ACT_LOOKUP  = model.add_lookup_parameters((self.NUM_ACTS, self.ACTION_DIM))
        # embed features or bag-of-word them?
        if not self.FEAT_DIM:
            print 'Using an n-hot representation for features.'
            # n-hot POS features are simply concatenated to feature vector
            self.FEAT_INPUT_DIM = self.NUM_FEATS + self.NUM_POS
        else:
            self.FEAT_LOOKUP = model.add_lookup_parameters((self.NUM_FEATS, self.FEAT_DIM))
            if self.pos_emb:
                self.POS_LOOKUP = model.add_lookup_parameters((self.NUM_POS, self.FEAT_DIM))
                # POS feature is the only feature with many values (=`self.NUM_POS`), hence + 1.
                # All other features are binary (e.g. SG and PL are disjoint binary features).
                self.FEAT_INPUT_DIM = self.NUM_FEATS*self.FEAT_DIM  # + 1 for POS and - 1 for UNK
                print 'All feature-value pairs are taken to be atomic, except for POS.'
            else:
                self.POS_LOOKUP = self.FEAT_LOOKUP  # self.POS_LOOKUP is probably not needed
                if self.avm_feat_format:
                    self.FEAT_INPUT_DIM = self.NUM_FEAT_TYPES*self.FEAT_DIM
                    print 'All feature-value pairs are taken to be non-atomic.'
                else:
                    self.FEAT_INPUT_DIM = (self.NUM_FEATS - 1)*self.FEAT_DIM  # -1 for UNK
                    print 'Every feature-value pair is taken to be atomic.'

        # BiLSTM encoding lemma
        self.fbuffRNN  = self.LSTM(self.ENC_LAYERS, self.CHAR_DIM, self.ENC_HIDDEN_DIM, model)
        self.bbuffRNN  = self.LSTM(self.ENC_LAYERS, self.CHAR_DIM, self.ENC_HIDDEN_DIM, model)

        # LSTM representing generated word
        self.WORD_REPR_DIM = self.ENC_HIDDEN_DIM*2 + self.ACTION_DIM + self.FEAT_INPUT_DIM
        self.wordRNN  = self.LSTM(self.DEC_LAYERS, self.WORD_REPR_DIM, self.DEC_HIDDEN_DIM, model)
        
        self.CLASSIFIER_IMPUT_DIM = self.DEC_HIDDEN_DIM
        if self.double_feats:
            self.CLASSIFIER_IMPUT_DIM += self.FEAT_INPUT_DIM

        print ' * LEMMA biLSTM:      IN-DIM: {}, OUT-DIM: {}'.format(2*self.CHAR_DIM, 2*self.ENC_HIDDEN_DIM)
        print ' * WORD LSTM:         IN-DIM: {}, OUT-DIM: {}'.format(self.WORD_REPR_DIM, self.DEC_HIDDEN_DIM)
        print ' LEMMA LSTMs have {} layer(s)'.format(self.ENC_LAYERS)
        print ' WORD LSTM has {} layer(s)'.format(self.DEC_LAYERS)
        print
        print ' * CHAR EMBEDDINGS:   IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_CHARS, self.CHAR_DIM)
        if not self.param_tying:
            print ' * ACTION EMBEDDINGS: IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_ACTS, self.ACTION_DIM)
        if self.FEAT_DIM:
            print ' * FEAT. EMBEDDINGS:  IN-DIM: {}, OUT-DIM: {}'.format(self.NUM_FEATS, self.FEAT_DIM)

            
    def _classifier(self, model):
        # single-hidden-layer classifier that works on feature presentation
        # from "self._features"
        if self.MLP_DIM:
            self.pW_s2h = model.add_parameters((self.MLP_DIM, self.CLASSIFIER_IMPUT_DIM))
            self.pb_s2h = model.add_parameters(self.MLP_DIM)
            feature_dim = self.MLP_DIM
            print ' * HIDDEN LAYER:      IN-DIM: {}, OUT-DIM: {}'.format(self.CLASSIFIER_IMPUT_DIM, feature_dim)
        else:
            feature_dim = self.CLASSIFIER_IMPUT_DIM
        # hidden to action
        self.pW_act = model.add_parameters((self.NUM_ACTS, feature_dim))
        self.pb_act = model.add_parameters(self.NUM_ACTS)
        print ' * SOFTMAX:           IN-DIM: {}, OUT-DIM: {}'.format(feature_dim, self.NUM_ACTS)
        
    def _build_model(self, model):
        # feature model
        self._features(model)
        # classifier
        self._classifier(model)
        
    def _build_lemma(self, lemma, unk_avg, is_training):
        # returns a list of character embedding for the lemma
        if is_training:
            lemma_enc = [self.CHAR_LOOKUP[c] for c in lemma]
        else:
            # then vectorize lemma with UNK
            if unk_avg:
                # UNK embedding is the average of trained embeddings (excluding UNK symbol=0)
                UNK_CHAR_EMB = dy.average([self.CHAR_LOOKUP[i] for i in xrange(1, self.NUM_CHARS)])
            else:
                # @TODO Pretrain it with "word dropout", otherwise
                # these are randomly initialized embeddings.
                UNK_CHAR_EMB = self.CHAR_LOOKUP[UNK]
            lemma_enc = [self.CHAR_LOOKUP[c] if c < self.NUM_CHARS else UNK_CHAR_EMB for c in lemma]
        return lemma_enc

    def _build_features(self, pos, feats):
        # represent morpho-syntactic features:
        if self.FEAT_DIM:
            feat_vecs = []
            if self.pos_emb:
                # POS gets a special treatment
                if pos < self.NUM_POS:
                    pos_emb = self.POS_LOOKUP[pos]
                else:
                    pos_emb = self.FEAT_LOOKUP[UNK]
                feat_vecs.append(pos_emb)

            if self.avm_feat_format:
                for ftype in range(self.NUM_FEAT_TYPES):
                    # each feature types gets represented with an embedding
                    # first, check if this feature type in `feats`
                    feat = feats.get(ftype, UNK)
                    # second, check if `feat` seen in training
                    if feat >= self.NUM_FEATS:
                        feat = UNK
                    feat_vecs.append(self.FEAT_LOOKUP[feat])

            else:                
                for feat in range(1, self.NUM_FEATS):  # skip UNK
                    if feat in feats:  # set of indices
                        feat_vecs.append(self.FEAT_LOOKUP[feat])
                    else:
                        feat_vecs.append(self.FEAT_LOOKUP[UNK])

            feats_enc = dy.concatenate(feat_vecs)
        else:
            # (upweighted) bag-of-features
            nhot = np.zeros(self.FEAT_INPUT_DIM)
            nhot[feats] = 1.
            if pos != UNK:
                # simply ignore UNK POS tag
                nhot[pos + self.NUM_FEATS] = 1.
            feats_enc = dy.inputVector(nhot * self.LUCKY_W)

        return feats_enc
    
    def set_dropout(self, dropout):
        self.wordRNN.set_dropout(dropout)

    def disable_dropout(self):
        self.wordRNN.disable_dropout()
    
    def l2_norm(self, with_embeddings=True):
        # specify regularization term: sum of Frobenius/L2-normalized weights
        # assume that we add to a computation graph
        reg = []
        # RNN weight matrices
        for rnn in (self.fbuffRNN, self.bbuffRNN, self.wordRNN):
            for exp in (e for layer in rnn.get_parameter_expressions() for e in layer):
                if len(exp.dim()[0]) != 1:
                    # this is not a bias term
                    reg.append(dy.l2_norm(exp))
        # classifier weight matices
        reg.append(dy.l2_norm(self.pW_act.expr()))
        if self.MLP_DIM:
            reg.append(dy.l2_norm(self.pW_s2h.expr()))
        if with_embeddings:
            # add embedding params
            reg.append(dy.l2_norm(self.FEAT_LOOKUP.expr()))
            reg.append(dy.l2_norm(self.CHAR_LOOKUP.expr()))
            if not self.param_tying:
                reg.append(dy.l2_norm(self.ACT_LOOKUP.expr()))
        return 0.5 * dy.esum(reg)

    def transduce(self, lemma, feats, oracle_actions=None, external_cg=True, sampling=False,
                  unk_avg=True, verbose=False):
        """
        Transduce an encoded lemma and features.
        Args:
            lemma: The input lemma, a list of integer character codes.
            feats: The features determining the morphological transformation. The most common
                   format is a list of integer codes, one code per feature-value pair.
            oracle_actions: `None` means prediction.
                            List of action codes is a static oracle.
                            A dictionary of keys (explained below) is the config for a dynamic oracle.
                                * "target_word": List of action codes for the target word form.
                                * "loss": Which loss function to use (softmax-margin, NLL, MSE).
                                * "rollout_mixin_beta": How to mix reference and learned roll-outs
                                    (1 is only reference, 0 is only model).
                                * "global_rollout": Whether to use one type of roll-out (expert or model)
                                    at the sequence level.
                                * "optimal": Whether to use an optimal or noisy (=buggy) expert
                                * "bias_inserts": Whether to use a buggy roll-out for inserts
                                    (which makes them as cheap as copies)
            external_cg: Whether or not an external computation graph is defined.
            sampling: Whether or not sampling should be used for decoding (e.g. for MRT) or
                      training (e.g. dynamic oracles with exploration / learned roll-ins).
            dynamic: Whether or not `oracle_actions` is a static oracle (list of actions) or a confuguration
                     for a dynamic oracle.
            unk_avg: Whether or not to average all char embeddings to produce UNK embedding
                     (see `self._build_lemma`).
            verbose: Whether or not to report on processing steps.
        """
        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)
        def _valid_actions(encoder):
            valid_actions = []
            if len(encoder) > 1:
                valid_actions += [COPY, DELETE]
            else:
                valid_actions += [END_WORD]
            valid_actions += self.INSERTS
            return valid_actions

        if not external_cg:
            dy.renew_cg()

        dynamic = None  # indicates prediction or static

        if oracle_actions:
            # if not, then prediction
            if isinstance(oracle_actions, dict):
                # dynamic oracle:
                # @TODO NB target word is not wrapped in boundary tags
                target_word = oracle_actions['target_word']
                generation_errors = set()
                dynamic = oracle_actions
            else:
                # static oracle:
                # reverse to enable simple popping
                oracle_actions = oracle_actions[::-1]
                oracle_actions.pop()  # COPY of BEGIN_WORD_CHAR

        # vectorize lemma
        lemma_enc = self._build_lemma(lemma, unk_avg, is_training=bool(oracle_actions))

        # vectorize features
        features = self._build_features(*feats)

        # add encoder and decoder to computation graph
        encoder = Encoder(self.fbuffRNN, self.bbuffRNN)
        decoder = self.wordRNN.initial_state()

        # add classifier to computation graph
        if self.MLP_DIM:
            # decoder output to hidden
            W_s2h = dy.parameter(self.pW_s2h)
            b_s2h = dy.parameter(self.pb_s2h)
        # hidden to action
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)

        # encoder is a stack which pops lemma characters and their
        # representations from the top. Thus, to get lemma characters
        # in the right order, the lemma has to be reversed.
        encoder.transduce(lemma_enc, lemma)

        encoder.pop()  # BEGIN_WORD_CHAR
        action_history = [COPY]
        word = []
        losses = []

        if verbose and not dynamic:
            count = 0
            print
            print action2string(oracle_actions, self.vocab)
            print lemma2string(lemma, self.vocab)
            
            
        if dynamic:
            # use model rollout for the whole of this sequence
            rollout_on = dynamic['global_rollout'] and np.random.rand() > dynamic['rollout_mixin_beta']
        
        while len(action_history) <= MAX_ACTION_SEQ_LEN:
            
            if verbose and not dynamic:
                print 'Action: ', count, self.vocab.act.i2w[action_history[-1]]
                print 'Encoder length, char: ', lemma, len(encoder), self.vocab.char.i2w[encoder.s[-1][-1]]
                print 'word: ', u''.join(word)
                print ('Remaining actions: ', oracle_actions, action2string(oracle_actions, self.vocab))
                count += 1
            
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            valid_actions = _valid_actions(encoder)
            encoder_embedding = encoder.embedding()
            # decoder
            decoder_input = dy.concatenate([encoder_embedding,
                                            features,
                                            self.ACT_LOOKUP[action_history[-1]]
                                           ])
            decoder = decoder.add_input(decoder_input)
            # classifier
            if self.double_feats:
                classifier_input = dy.concatenate([decoder.output(), features])
            else:
                classifier_input = decoder.output()
            if self.MLP_DIM:
                h = self.NONLIN(W_s2h * classifier_input + b_s2h)
            else:
                h = classifier_input
            logits = W_act * h + b_act
            # get action (argmax, sampling, or use oracle actions)
            if oracle_actions is None:
                # predicting by argmax or sampling
                log_probs = dy.log_softmax(logits, valid_actions)
                log_probs_np = log_probs.npvalue()
                if sampling:
                    action = sample(log_probs_np)
                else:
                    action = np.argmax(log_probs_np)
                losses.append(dy.pick(log_probs, action))
            elif dynamic:
                # training with dynamic oracle
                if rollout_on or (not dynamic['global_rollout'] and np.random.rand() > dynamic['rollout_mixin_beta']):
                    # the second disjunct allows for model roll-out applied locally
                    rollout = lambda action: self.rollout(action, dy.log_softmax(logits, valid_actions),
                                                          action_history, features, decoder, encoder, word,
                                                          W_act, b_act)  # @TODO W_s2h ...
                else:
                    rollout = None
                
                optim_actions, costs = oracle_with_rollout(word, target_word, encoder.get_extra(),
                                                           valid_actions, rollout, self.vocab,
                                                           optimal=dynamic['optimal'],
                                                           bias_inserts=dynamic['bias_inserts'],
                                                           errors=generation_errors,
                                                           verbose=verbose)

                log_probs = dy.log_softmax(logits, valid_actions)
                log_probs_np = log_probs.npvalue()
                if sampling == 1. or np.random.rand() <= sampling:
                    # action is picked by sampling
                    action = sample(log_probs_np)
                    # @TODO IL learned roll-ins are done with policy i.e. greedy / beam search decoding
                    if verbose: print 'Rolling in with model: ', action, self.vocab.act.i2w[action] 
                else:
                    # action is picked from optim_actions
                    action = optim_actions[np.argmax([log_probs_np[a] for a in optim_actions])]
                    #print [log_probs_np[a] for a in optim_actions]
                # loss is over all optimal actions.
                
                if dynamic['loss'] == 'softmax-margin':
                    loss = log_sum_softmax_margin_loss(optim_actions, logits, self.NUM_ACTS,
                                                       costs=costs, valid_actions=None, verbose=verbose)
                elif dynamic['loss'] == 'nll':
                    loss = log_sum_softmax_loss(optim_actions, logits, self.NUM_ACTS, 
                                                valid_actions=valid_actions, verbose=verbose)
                elif dynamic['loss'] == 'mse':
                    loss = cost_sensitive_reg_loss(optim_actions, logits, self.NUM_ACTS,
                                                   # NB expects both costs and valid actions!
                                                   costs=costs, valid_actions=valid_actions, verbose=verbose)
                ################
                else:
                    raise NotImplementedError
                losses.append(loss)
                #print 'Action'
                #print action
                #print self.vocab.act.i2w[action]
            else:
                # training with static oracle
                action = oracle_actions.pop()
                log_probs = dy.log_softmax(logits, valid_actions)
                losses.append(dy.pick(log_probs, action))

            action_history.append(action)

            #print 'action, log_probs: ', action, self.vocab.act.i2w[action], losses[-1].scalar_value(), log_probs.npvalue()
            
            # execute the action to update the transducer state
            if action == COPY:
                # 1. Increment attention index
                try:
                    char_ = encoder.pop()
                except IndexError, e:
                    print np.exp(log_probs.npvalue())
                    print 'COPY: ', action
                # 2. Append copied character to the output word
                word.append(self.vocab.char.i2w[char_])
            elif action == DELETE:               
                # 1. Increment attention index
                try:
                    encoder.pop()
                except IndexError, e:
                    print np.exp(log_probs.npvalue())
                    print 'DELETE: ', action
            elif action == END_WORD:
                # 1. Finish transduction
                break
            else:
                # one of the INSERT actions
                assert action in self.INSERTS
                # 1. Append inserted character to the output word
                char_ = self.vocab.act.i2w[action]
                word.append(char_)
                
        word = u''.join(word)

        return losses, word, action_history
    
    
    def rollout(self, action, log_probs, action_history, features, decoder, encoder, word, W_act, b_act):
        """Roll out the model to make greedy prediction."""
        # No new computation graph?
        def _valid_actions(encoder):
            valid_actions = []
            if len(encoder) > 1:
                valid_actions += [COPY, DELETE]
            else:
                valid_actions += [END_WORD]
            valid_actions += self.INSERTS
            return valid_actions
        
        # copy mutables
        action_history = list(action_history)
        encoder = encoder.copy()
        word = list(word)   

        action_history.append(action)
        losses = [dy.pick(log_probs, action)]

        if action != END_WORD:
            # Execute selected action and rollout
            if action == COPY:
                # 1. Increment attention index
                char_ = encoder.pop()
                # 2. Append copied character to the output word
                word.append(self.vocab.char.i2w[char_])
            elif action == DELETE:               
                # 1. Increment attention index
                encoder.pop()
            else:
                # one of the INSERT actions
                assert action in self.INSERTS, (action, self.vocab.act.i2w[action])
                # 1. Append inserted character to the output word
                char_ = self.vocab.act.i2w[action]
                word.append(char_)

            # rollout
            while len(action_history) <= MAX_ACTION_SEQ_LEN:

                valid_actions = _valid_actions(encoder)
                # decoder
                decoder_input = dy.concatenate([encoder.embedding(),
                                                features,
                                                self.ACT_LOOKUP[action_history[-1]]
                                               ])
                decoder = decoder.add_input(decoder_input)
                # classifier
                if self.double_feats:
                    classifier_input = dy.concatenate([decoder.output(), features])
                else:
                    classifier_input = decoder.output()
                if self.MLP_DIM:
                    raise NotImplementedError
                    #h = self.NONLIN(W_s2h * classifier_input + b_s2h)
                else:
                    h = classifier_input

                logits = W_act * h + b_act
                log_probs = dy.log_softmax(logits, valid_actions)
                action = np.argmax(log_probs.npvalue())

                losses.append(dy.pick(log_probs, action))
                action_history.append(action)

                # execute the action to update the transducer state
                if action == COPY:
                    # 1. Increment attention index
                    char_ = encoder.pop()
                    # 2. Append copied character to the output word
                    word.append(self.vocab.char.i2w[char_])
                elif action == DELETE:               
                    # 1. Increment attention index
                    encoder.pop()
                elif action == END_WORD:
                    # 1. Finish transduction
                    break
                else:
                    # one of the INSERT actions
                    assert action in self.INSERTS
                    # 1. Append inserted character to the output word
                    char_ = self.vocab.act.i2w[action]
                    word.append(char_)
        
        # from action history one can compute the cost
        word = u''.join(word)
        return losses, word, action_history


    def beam_search_decode(self, lemma, feats, external_cg=True, unk_avg=True, beam_width=4):
        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)
        def _valid_actions(encoder):
            valid_actions = []
            if len(encoder) > 1:
                valid_actions += [COPY, DELETE]
            else:
                valid_actions += [END_WORD]
            valid_actions += self.INSERTS
            return valid_actions

        if not external_cg:
            dy.renew_cg()

        # vectorize lemma
        lemma_enc = self._build_lemma(lemma, unk_avg, is_training=False)

        # vectorize features
        features = self._build_features(*feats)
            
        # add encoder and decoder to computation graph
        encoder = Encoder(self.fbuffRNN, self.bbuffRNN)
        decoder = self.wordRNN.initial_state()
        
        # encoder is a stack which pops lemma characters and their
        # representations from the top.
        encoder.transduce(lemma_enc, lemma)

        # add classifier to computation graph
        if self.MLP_DIM:
            # decoder output to hidden
            W_s2h = dy.parameter(self.pW_s2h)
            b_s2h = dy.parameter(self.pb_s2h)
        # hidden to action
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)
    
        encoder.pop()  # BEGIN_WORD_CHAR
        
        # a list of tuples:
        #    (decoder state, encoder state, list of previous actions,
        #     log prob of previous actions, log prob of previous actions as dynet object,
        #     word generated so far)
        beam = [(decoder, encoder, [COPY], 0., 0., [])]

        beam_length = 0
        complete_hypotheses = []
        
        while beam_length <= MAX_ACTION_SEQ_LEN:
            
            if not beam or beam_width == 0:
                break
            
            #if show_oracle_actions:
            #    print 'Action: ', count, self.vocab.act.i2w[action_history[-1]]
            #    print 'Encoder length, char: ', lemma, len(encoder), self.vocab.char.i2w[encoder.s[-1][-1]]
            #    print 'word: ', u''.join(word)
            #    print 'Remaining actions: ', oracle_actions, u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
            #    count += 1
            #elif action_history[-1] >= self.NUM_ACTS:
            #    print 'Will be adding unseen act embedding: ', self.vocab.act.i2w[action_history[-1]]
            
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            expansion = []
            #print 'Beam length: ', beam_length
            for decoder, encoder, prev_actions, log_p, log_p_expr, word in beam:
                #print 'Expansion: ', action2string(prev_actions, self.vocab), log_p, ''.join(word)
                valid_actions = _valid_actions(encoder)
                # decoder
                decoder_input = dy.concatenate([encoder.embedding(),
                                                features,
                                                self.ACT_LOOKUP[prev_actions[-1]]
                                               ])
                decoder = decoder.add_input(decoder_input)
                # classifier
                if self.double_feats:
                    classifier_input = dy.concatenate([decoder.output(), features])
                else:
                    classifier_input = decoder.output()
                if self.MLP_DIM:
                    h = self.NONLIN(W_s2h * classifier_input + b_s2h)
                else:
                    h = classifier_input
                logits = W_act * h + b_act
                log_probs_expr = dy.log_softmax(logits, valid_actions)
                log_probs = log_probs_expr.npvalue()
                top_actions = np.argsort(log_probs)[-beam_width:]
                #print 'top_actions: ', top_actions, action2string(top_actions, self.vocab) 
                #print 'log_probs: ', log_probs
                #print
                expansion.extend((
                    (decoder, encoder.copy(),
                     list(prev_actions), a, log_p + log_probs[a],
                     log_p_expr + log_probs_expr[a], list(word)) for a in top_actions))

            #print 'Overall, {} expansions'.format(len(expansion))
            beam = []
            expansion.sort(key=lambda e: e[4])
            for e in expansion[-beam_width:]:
                decoder, encoder, prev_actions, action, log_p, log_p_expr, word = e
            
                prev_actions.append(action)

                # execute the action to update the transducer state
                if action == END_WORD:
                    # 1. Finish transduction:
                    #  * beam width should be decremented
                    #  * expansion should be taken off the beam and
                    # stored to final hypotheses set
                    beam_width -= 1
                    complete_hypotheses.append((log_p, log_p_expr, u''.join(word), prev_actions))
                else:
                    if action == COPY:
                        # 1. Increment attention index
                        char_ = encoder.pop()
                        # 2. Append copied character to the output word
                        word.append(self.vocab.char.i2w[char_])
                    elif action == DELETE:               
                        # 1. Increment attention index
                        encoder.pop()
                    else:
                        # one of the INSERT actions
                        assert action in self.INSERTS
                        # 1. Append inserted character to the output word
                        char_ = self.vocab.act.i2w[action]
                        word.append(char_)
                    beam.append((decoder, encoder, prev_actions, log_p, log_p_expr, word))
            
            beam_length += 1

        if not complete_hypotheses:
            # nothing found because the model is so crappy
            complete_hypotheses = [(log_p, log_p_expr, u''.join(word), prev_actions)
                                   for _, _, prev_actions, log_p, log_p_expr, word in beam]

        complete_hypotheses.sort(key=lambda h: h[0], reverse=True)
        #print u'Complete hypotheses:'
        #for log_p, _, word, actions in complete_hypotheses:
        #    print u'Actions {}, word {}, log p {:.3f}'.format(action2string(actions, self.vocab), word, log_p)
            
        return complete_hypotheses
