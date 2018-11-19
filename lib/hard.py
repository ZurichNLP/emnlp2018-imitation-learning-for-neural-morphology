from __future__ import division

import dynet as dy
import numpy as np

from datasets import action2string
from defaults import STEP, BEGIN_WORD, END_WORD, UNK, MAX_ACTION_SEQ_LEN
from stack_lstms import Encoder
from transducer import Transducer

""" Reimplementation of Roee Aharoni & Joav Goldberg's hard-attention transducer model """

def lemma2string(lemma, vocab):
    return  u''.join(vocab.char.i2w[l] for l in lemma)

class Transducer(Transducer):
    
    def transduce(self, lemma, feats, oracle_actions=None, external_cg=True, sampling=False,
                  unk_avg=True, debug_mode=False):
        
        def _valid_actions(encoder):
            valid_actions = list(self.INSERTS)
            if len(encoder) > 1:
                valid_actions += [STEP]
            else:
                valid_actions += [END_WORD]
            return valid_actions

        if not external_cg:
            dy.renew_cg()

        if oracle_actions:
            # reverse to enable simple popping
            oracle_actions = oracle_actions[::-1]
            oracle_actions.pop()  # Deterministic insertion of BEGIN_WORD

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

        # encoder is a stack which pops lemma characters 
        # and their representations from the top
        encoder.transduce(lemma_enc, lemma)

        action_history = [BEGIN_WORD]
        word = []
        losses = []
        count = 0
        
        if debug_mode:
            print
            if oracle_actions: print action2string(oracle_actions, self.vocab)
            print lemma2string(lemma, self.vocab)
        
        while len(action_history) <= MAX_ACTION_SEQ_LEN:
            
            # what is at the top of encoder? 
            encoder_embedding, char_enc = encoder.embedding(extra=True)
            
            if debug_mode:
                print 'Action history: ', action_history, action2string(action_history, self.vocab)
                print 'Encoder length: ', len(encoder) 
                print 'Current char: ', char_enc, lemma2string([char_enc], self.vocab)
                print 'Word so far: ', u''.join(word)

            # decoder
            decoder_input = dy.concatenate([encoder_embedding,
                                            features,
                                            self.ACT_LOOKUP[action_history[-1]]
                                           ])
            decoder = decoder.add_input(decoder_input)
            decoder_output = decoder.output()
            # classifier
            if self.MLP_DIM:
                h = self.NONLIN(W_s2h * decoder_output + b_s2h)
            else:
                h = decoder_output
            
            valid_actions = _valid_actions(encoder)
            log_probs = dy.log_softmax(W_act*h + b_act, valid_actions)

            if oracle_actions is None:
                if sampling:
                    dist = np.exp(log_probs.npvalue())
                    # sample according to softmax
                    rand = np.random.rand()
                    for action, p in enumerate(dist):
                        rand -= p
                        if rand <= 0: break
                else:
                    action = np.argmax(log_probs.npvalue())
            else:
                action = oracle_actions.pop()

            losses.append(dy.pick(log_probs, action))
            action_history.append(action)

            if action == STEP:
                # Delete action
                encoder.pop()
                
            elif action == END_WORD:
                # Finish transduction
                break
            else:
                # Insert action
                assert action in self.INSERTS, (char_, action2string([char_], self.vocab), self.INSERTS)
                char_ = self.vocab.act.i2w[action]
                word.append(char_)
                
        word = u''.join(word)
        return losses, word, action_history
    
    def beam_search_decode(self, lemma, feats, external_cg=True, unk_avg=True, beam_width=4):
        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)
        def _valid_actions(encoder):
            valid_actions = list(self.INSERTS)
            if len(encoder) > 1:
                valid_actions += [STEP]
            else:
                valid_actions += [END_WORD]
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
        
        # a list of tuples:
        #    (decoder state, encoder state, list of previous actions,
        #     log prob of previous actions, log prob of previous actions as dynet object,
        #     word generated so far)
        beam = [(decoder, encoder, [BEGIN_WORD], 0., 0., [])]
        
        beam_length = 0
        complete_hypotheses = []
        
        while beam_length <= MAX_ACTION_SEQ_LEN:
            
            if not beam or beam_width == 0:
                break
        
            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            expansion = []
            # print 'Beam length: ', beam_length
            for decoder, encoder, prev_actions, log_p, log_p_expr, word in beam:
                # print 'Expansion: ', action2string(prev_actions, self.vocab), log_p, ''.join(word)
                encoder_embedding, char_enc = encoder.embedding(extra=True)
                # decoder
                decoder_input = dy.concatenate([encoder_embedding, features, self.ACT_LOOKUP[prev_actions[-1]]])
                decoder = decoder.add_input(decoder_input)
                decoder_output = decoder.output()
                # generate
                if self.MLP_DIM:
                    h = self.NONLIN(W_s2h * decoder_output + b_s2h)
                else:
                    h = decoder_output
                
                logits = W_act * h + b_act
                valid_actions = _valid_actions(encoder)
                log_probs_expr = dy.log_softmax(logits, valid_actions)
                log_probs = log_probs_expr.npvalue()
                top_actions = np.argsort(log_probs)[-beam_width:]
                # print 'top_actions: ', top_actions, action2string(top_actions, self.vocab)
                # print 'log_probs: ', log_probs
                # print
                expansion.extend(((decoder, encoder.copy(), list(prev_actions), a, log_p + log_probs[a], log_p_expr + log_probs_expr[a], list(word), char_enc) for a in top_actions))
            
            # print 'Overall, {} expansions'.format(len(expansion))
            beam = []
            expansion.sort(key=lambda e: e[4])
            for e in expansion[-beam_width:]:
                decoder, encoder, prev_actions, action, log_p, log_p_expr, word, char_enc = e
                
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
                    if action == STEP:
                        encoder.pop()
                    else:
                        # one of the INSERT actions
                        # 1. Append inserted character to the output word
                        assert action in self.INSERTS, (char_, action2string([char_], self.vocab), self.INSERTS)
                        char_ = self.vocab.act.i2w[action]
                        word.append(char_)
                    beam.append((decoder, encoder, prev_actions, log_p, log_p_expr, word))
            beam_length += 1
        
        if not complete_hypotheses:
            # nothing found because the model is so crappy
            complete_hypotheses = [(log_p, log_p_expr, u''.join(word), prev_actions) for _, _, prev_actions, log_p, log_p_expr, word in beam]

        complete_hypotheses.sort(key=lambda h: h[0], reverse=True)
        # print u'Complete hypotheses:'
        # for log_p, _, word, actions in complete_hypotheses:
        #   print u'Actions {}, word {}, log p {:.3f}'.format(action2string(actions, self.vocab), word, log_p)

        return complete_hypotheses



