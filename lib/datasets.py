from __future__ import division

import os
import codecs

from aligners import smart_align, dumb_align
from defaults import (ALIGN_SYMBOL, STEP, COPY, DELETE,
                      BEGIN_WORD_CHAR, END_WORD_CHAR,
                      BEGIN_WORD, END_WORD, DELETE_CHAR, COPY_CHAR,
                      UNK, SPECIAL_CHARS)
from vocabulary import EditVocab, MinimalVocab

#############################################################
# DATASETS
#############################################################

class BaseDataSample(object):
    # data sample with encoded features
    def __init__(self, lemma, lemma_str, word, word_str, pos, feats, feat_str, tag_wraps, vocab):
        self.vocab = vocab          # vocab of unicode strings
        self.lemma = lemma          # list of encoded lemma characters
        self.lemma_str = lemma_str  # unicode string
        self.word = word            # encoded word
        self.word_str = word_str    # unicode string
        self.pos = pos              # encoded pos feature
        self.feats = feats          # set of encoded features
        self.feat_str = feat_str    # original serialization of features, unicode
        # new serialization of features, unicode
        self.feat_repr = feats2string(self.pos, self.feats, self.vocab)
        self.tag_wraps = tag_wraps  # were lemma / word wrapped with word boundary tags '<' and '>'?

    def __repr__(self):
        return u'Lemma: {}, Word: {}, Features: {}, Wraps: {}'.format(self.lemma_str,
            self.word_str, self.feat_repr, self.tag_wraps)

    @classmethod
    def from_row(cls, vocab, tag_wraps, verbose, row, sigm2017format=True,
                 no_feat_format=False, pos_emb=True, avm_feat_format=False):
        if sigm2017format:
            lemma_str, word_str, feat_str = row
            feats_delimiter = u';'
        else:
            lemma_str, feat_str, word_str = row
            feats_delimiter = u','
        # encode features as integers
        # POS feature treated separately
        assert isinstance(lemma_str, unicode), lemma_str
        assert not any(c in lemma_str for c in SPECIAL_CHARS), (lemma_str, SPECIAL_CHARS)
        assert isinstance(word_str, unicode), word_str
        assert not any(c in word_str for c in SPECIAL_CHARS), (word_str, SPECIAL_CHARS)
        assert isinstance(feat_str, unicode), feat_str
        # `avm_feat_format=True` implies that `pos_emb=False`
        if avm_feat_format: assert not pos_emb
        # encode lemma characters
        lemma = [vocab.char[c] for c in lemma_str]
        # encode word
        word = vocab.word[word_str]
        feats = feat_str.split(feats_delimiter) if not no_feat_format else ['']
        if pos_emb:
            # encode features and, separately, pos
            pos = vocab.pos[feats[0]]
            feats = [vocab.feat[f] for f in set(feats[1:])]
        else:
            pos = None
            if avm_feat_format:
                # map from encoded feature names to encoded features
                feats = {vocab.feat_type[f.split('=')[0]] : vocab.feat[f] for f in set(feats)}
            else:
                feats = [vocab.feat[f] for f in set(feats)]
        # wrap encoded lemma with (encoded) boundary tags
        if tag_wraps == 'both':
            lemma = [BEGIN_WORD] + lemma + [END_WORD]
        elif tag_wraps == 'close':
            lemma = lemma + [END_WORD]
        # print features and lemma at a high verbosity level
        if verbose == 2:
            print u'POS & features from {}, {}, {}: {}, {}'.format(feat_str, word_str, lemma_str, pos, feats)
            print u'lemma encoding: {}'.format(lemma)
        return cls(lemma, lemma_str, word, word_str, pos, feats, feat_str, tag_wraps, vocab)


class NonAlignedDataSample(BaseDataSample):
    # Sample that additionally holds encodings of the word form.
    # Building its actions doesn't need alignment (which, in the general case, requires reading in all samples first)
    # and so it can be performs on a per-sample basis.
    def __init__(self, *args):
        super(NonAlignedDataSample, self).__init__(*args)
        # @TODO NB target word is not wrapped in boundary tags
        self.actions = [self.vocab.act[c] for c in self.word_str]
        self.act_repr = self.word


class AlignedDataSample(BaseDataSample):
    # data sample with encoded oracle actions derived from character alignment of lemma and word
    def set_actions(self, actions, aligned_lemma, aligned_word):
        self.actions = actions              # list of indices
        # serialization of actions as unicode string
        self.act_repr = action2string(self.actions, self.vocab)
        self.aligned_lemma = aligned_lemma  # unicode string
        self.aligned_word = aligned_word    # unicode string

    def __repr__(self):
        return u'Lemma: {}, Word: {}, Features: {}, Actions: {}'.format(
            self.lemma_str, self.word_str, self.feat_repr, self.act_repr)

def lemma2string(lemma, vocab):
    return u''.join(vocab.char.i2w[c] for c in lemma)

def action2string(actions, vocab):
    return u''.join(vocab.act.i2w[a] for a in actions)

def feats2string(pos, feats, vocab):
    if pos:
        pos_str = vocab.pos.i2w[pos] + u';'
    else:
        pos_str = u''
    return  pos_str + u';'.join(vocab.feat.i2w[f] for f in feats)

class BaseDataSet(object):
    # class to hold an encoded dataset
    def __init__(self, filename, samples, vocab, training_data, tag_wraps, verbose, **kwargs):
        self.filename = filename
        self.samples = samples
        self.vocab = vocab
        self.length = len(self.samples)
        self.training_data = training_data
        self.tag_wraps = tag_wraps
        self.verbose = verbose

    def __len__(self): return self.length

    @classmethod
    def from_file(cls, filename, vocab, DataSample=BaseDataSample,
                  encoding='utf8', delimiter=u'\t', sigm2017format=True, no_feat_format=False,
                  pos_emb=True, avm_feat_format=False, tag_wraps='both', verbose=True, **kwargs):
        # filename (str):   tab-separated file containing morphology reinflection data:
        #                   lemma word feat1;feat2;feat3...
        training_data = True if 'train' in os.path.basename(filename) else False
        datasamples = []

        print 'Loading data from file: {}'.format(filename)
        print 'These are {} data.'.format('training' if training_data else 'holdout')
        print 'Word boundary tags?', tag_wraps
        print 'Verbose?', verbose

        if avm_feat_format:
            # check that `avm_feat_format` and `pos_emb` does not clash
            if pos_emb:
                print 'Attribute-value feature matrix implies that no specialized pos embedding is used.'
                pos_emb = False

        with codecs.open(filename, encoding=encoding) as f:
            for row in f:
                split_row = row.strip().split(delimiter)
                sample = DataSample.from_row(vocab, tag_wraps, verbose, split_row,
                                             sigm2017format, no_feat_format, pos_emb, avm_feat_format)
                datasamples.append(sample)

        return cls(filename=filename, samples=datasamples, vocab=vocab,
                   training_data=training_data, tag_wraps=tag_wraps, verbose=verbose, **kwargs)


class NonAlignedDataSet(BaseDataSet):
    # this dataset does not support alignment between inputs and outputs
    @classmethod
    def from_file(cls, filename, vocab=None, pos_emb=True, avm_feat_format=False,
                  param_tying=False, **kwargs):
        if vocab:
            assert isinstance(vocab, EditVocab)
        else:
            vocab = EditVocab(pos_emb=pos_emb, avm_feat_format=avm_feat_format,
                                 param_tying=param_tying)
        print vocab
        #return super(NonAlignedDataSet, cls).from_file(filename, vocab, , **kwargs)
        ds = super(NonAlignedDataSet, cls).from_file(filename, vocab, DataSample=NonAlignedDataSample,
                                                       pos_emb=pos_emb, avm_feat_format=avm_feat_format, **kwargs)
        print 'Number of actions: {}'.format(len(ds.vocab.act))
        print u'Action set: {}'.format(' '.join(sorted(ds.vocab.act.keys())))
        return ds

class AlignedDataSet(BaseDataSet):
    # this dataset aligns its inputs
    def __init__(self, aligner=smart_align, **kwargs):
        super(AlignedDataSet, self).__init__(**kwargs)

        self.aligner = aligner
        # wrapping lemma / word with word boundary tags
        if self.tag_wraps == 'both':
            self.wrapper = lambda s: BEGIN_WORD_CHAR + s + END_WORD_CHAR
        elif self.tag_wraps == 'close':
            self.wrapper = lambda s: s + END_WORD_CHAR
        else:
            self.wrapper = lambda s: s

        print 'Started aligning with {} aligner...'.format(self.aligner)
        aligned_pairs = self.aligner([(s.lemma_str, s.word_str) for s in self.samples], **kwargs)
        print 'Finished aligning.'

        print 'Started building oracle actions...'
        for (al, aw), s in zip(aligned_pairs, self.samples):
            al = self.wrapper(al)
            aw = self.wrapper(aw)
            self._build_oracle_actions(al, aw, sample=s, **kwargs)
        print 'Finished building oracle actions.'
        print 'Number of actions: {}'.format(len(self.vocab.act))
        print u'Action set: {}'.format(' '.join(sorted(self.vocab.act.keys())))

        if self.verbose:
            print 'Examples of oracle actions:'
            for a in (s.act_repr for s in self.samples[:20]):
                print a #.encode('utf8')

    def _build_oracle_actions(self, al_lemma, al_word, sample, **kwargs):
        pass

    @classmethod
    def from_file(cls, filename, vocab, **kwargs):
        return super(AlignedDataSet, cls).from_file(filename, vocab, AlignedDataSample, **kwargs)


class MinimalDataSet(AlignedDataSet):
    # this dataset builds actions with
    # Algorithm of Aharoni & Goldberg 2017
    def _build_oracle_actions(self, lemma, word, sample, **kwargs):
        # Aharoni & Goldberg 2017 Algorithm 1
        actions = []
        alignment_len = len(lemma)
        for i, (l, w) in enumerate(zip(lemma, word)):
            if w == ALIGN_SYMBOL:
                actions.append(STEP)
            else:
                actions.append(self.vocab.act[w])  # encode w
                if i+1 < alignment_len and lemma[i+1] != ALIGN_SYMBOL:
                    actions.append(STEP)
        if self.verbose == 2:
            print u'{}\n{}\n{}\n'.format(word,
                action2string(actions, self.vocab), lemma)

        sample.set_actions(actions, lemma, word)

    @classmethod
    def from_file(cls, filename, vocab=None, pos_emb=True, avm_feat_format=False,
                  param_tying=False, **kwargs):
        if vocab:
            assert isinstance(vocab, MinimalVocab)
        else:
            vocab = MinimalVocab(pos_emb=pos_emb, avm_feat_format=avm_feat_format,
                                 param_tying=param_tying)
        print vocab
        return super(MinimalDataSet, cls).from_file(filename, vocab, pos_emb=pos_emb,
                                                    avm_feat_format=avm_feat_format, **kwargs)


class EditDataSet(AlignedDataSet):
    # this dataset uses COPY action
    def __init__(self, try_reverse=False, substitution=False, copy_as_substitution=False,
                 reorder_deletes=True, freq_check=(0.1, 0.3), **kwargs):
        # "try reverse" only makes sense with dumb aligner
        self.try_reverse = try_reverse and self.aligner == dumb_align  # @TODO Fix bug
        if self.try_reverse:
            print 'USING STRING REVERSING WITH DUMB ALIGNMENT...'
            print 'USING DEFAULT ALIGN SYMBOL ~'
        self.copy_as_substitution = copy_as_substitution
        self.substitution = substitution
        if copy_as_substitution is True:
            self.substitution = True
            print 'TREATING COPY AS SUBSTITUTIONS'
        if self.substitution is True:
            self.reorder_deletes = False
            print 'USING SUBSTITUTION ACTIONS, NOT REORDERING DELETES'
        else:
            self.reorder_deletes = reorder_deletes
        # "frequency check" for COPY and DELETE actions
        self.freq_check = freq_check

        super(EditDataSet, self).__init__(**kwargs)
        if self.freq_check:
            copy_low, delete_high = self.freq_check
            # some stats on actions
            action_counter = self.vocab.act.freq()
            #print action_counter.values()
            freq_delete = action_counter[DELETE] / sum(action_counter.values())
            freq_copy = action_counter[COPY] / sum(action_counter.values())
            print ('Alignment results: COPY action freq {:.3f}, '
                   'DELETE action freq {:.3f}'.format(freq_copy, freq_delete))
            if freq_copy < copy_low:
                print 'WARNING: Too few COPY actions!\n'
            if freq_delete > delete_high:
                print 'WARNING: Many DELETE actions!\n'

    def _build_oracle_actions(self, lemma, word, sample, **kwargs):
        # Makarov et al 2017 Algorithm 1
        def _build(lemma, word):
            actions = []
            alignment_len = len(lemma)
            has_copy = False
            for i, (l, w) in enumerate(zip(lemma, word)):
                if l == ALIGN_SYMBOL:
                    actions.append(self.vocab.act[w])
                elif w == ALIGN_SYMBOL:
                    actions.append(self.vocab.act[DELETE_CHAR])
                elif l == w:
                    if i+1 == alignment_len:
                        # end of string => insert </s>
                        actions.append(self.vocab.act[w])
                    elif self.copy_as_substitution:
                        # treat copy as another substitution action
                        actions.append(self.vocab.act[w+u'@'])
                    else:
                        # treat copy as a special action
                        actions.append(self.vocab.act[COPY_CHAR])
                        has_copy = True
                else:
                    # substitution
                    if self.substitution:
                        subt = self.vocab.act[w+u'@'],
                        #subt = (self.vocab.act[u'@' + l + w + u'@'],)
                    else:
                        subt = self.vocab.act[DELETE_CHAR], self.vocab.act[w]
                    actions.extend(subt)
            return actions, has_copy

        actions, has_copy = _build(lemma, word)

        if self.try_reverse and has_copy:
            # no copying is being done, probably
            # this sample uses prefixation. Try aligning
            # original pair from the end:
            reversed_pair = sample.lemma[::-1], sample.word[::-1]
            [(new_al_lemma, new_al_word)] = self.aligner([reversed_pair], ALIGN_SYMBOL)
            ractions, has_copy = _build(new_al_lemma[::-1], new_al_word[::-1])
            if has_copy:
                print (u'Reversed aligned: {} => {}\n'
                       u'Forward alignment: {}, REVERSED alignment: {}'.format(
                        new_al_lemma, new_al_word,
                        action2string(actions,  self.vocab),
                        action2string(ractions, self.vocab)))
                actions = ractions

        if self.reorder_deletes:
            reordered_actions = []
            suffix = []
            for i, c in enumerate(actions):
                if i == 0 or c == COPY:
                    reordered_actions.append(c)
                    # count deletes and store inserts
                    # between two copy actions
                    inserts = []
                    deletes = 0
                    for b in actions[i+1:]:
                        if b == COPY:
                            # copy
                            break
                        elif b == DELETE:
                            # delete
                            deletes += 1
                        else:
                            inserts.append(b)
                    between_copies = [DELETE]*deletes + inserts
                    reordered_actions.extend(between_copies)
            actions = reordered_actions + suffix

        if self.verbose == 2:
            print u'{}\n{}\n{}\n'.format(word,
                                         action2string(actions, self.vocab),
                                         lemma)

        sample.set_actions(actions, lemma, word)

    @classmethod
    def from_file(cls, filename, vocab=None, pos_emb=True, avm_feat_format=False,
                  param_tying=False, **kwargs):
        if vocab:
            assert isinstance(vocab, EditVocab)
        else:
            vocab = EditVocab(pos_emb=pos_emb, avm_feat_format=avm_feat_format,
                              param_tying=param_tying)
        print vocab
        return super(EditDataSet, cls).from_file(filename, vocab, pos_emb=pos_emb,
                                                 avm_feat_format=avm_feat_format, **kwargs)


if __name__ == "__main__":
    import os
    from defaults import DATA_PATH

    for dataset, name in ((MinimalDataSet, 'MinimalDataSet'), (EditDataSet, 'EditDataSet'),
                          (NonAlignedDataSet, 'NonAlignedDataSet')):
        print '\n\n*** DATASET TYPE (TRAIN): ***',  name
        train_fn = os.path.join(DATA_PATH, 'russian-train-low')
        ds = dataset.from_file(train_fn, verbose=0,
                               tag_wraps='both',
                               param_tying=False,
                               iterations=5)
        vocab = ds.vocab

        print '\n\n*** DATASET TYPE (DEV): ***', name
        dev_fn = os.path.join(DATA_PATH, 'russian-dev')
        ds = dataset.from_file(dev_fn, vocab=vocab,
                               verbose=0,
                               tag_wraps='both',
                               param_tying=False,
                               iterations=5)
