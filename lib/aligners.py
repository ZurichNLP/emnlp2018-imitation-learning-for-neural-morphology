#-*- coding: utf-8 -*-
import align

from defaults import ALIGN_SYMBOL

#############################################################
# ALIGNERS
#############################################################

def smart_align(pairs, align_symbol=ALIGN_SYMBOL,
                iterations=150, burnin=5, lag=1, mode='crp', **kwargs):
    return align.Aligner(pairs,
                         align_symbol=align_symbol,
                         iterations=iterations,
                         burnin=burnin,
                         lag=lag,
                         mode=mode).alignedpairs


def dumb_align(pairs, align_symbol=ALIGN_SYMBOL, multiword=True, **kwargs):
    def _dumb_align(ins, outs):
        length_diff = len(ins) - len(outs)
        if length_diff > 0:
            outs += align_symbol * length_diff
        elif length_diff < 0:
            ins += align_symbol * abs(length_diff)
        return ins, outs
    return multiword_align(pairs, _dumb_align, multiword)

def multiword_align(pairs, _align, multiword):
    if multiword:
        alignedpairs = []
        for ins, outs in pairs:
            if ins.count(u' ') == outs.count(u' '):
                # assume multiword expression
                aligned_ins = []
                aligned_outs = []
                for subins, subouts in zip(ins.split(u' '), outs.split(u' ')):
                    # align each subunit separately
                    aligned_subins, aligned_subouts = _align(subins, subouts)
                    aligned_ins.append(aligned_subins)
                    aligned_outs.append(aligned_subouts)
                aligned_ins  = u' '.join(aligned_ins)
                aligned_outs = u' '.join(aligned_outs)
            else:
                aligned_ins, aligned_outs = _align(ins, outs)
            alignedpairs.append((aligned_ins, aligned_outs))
    else:
        alignedpairs = [_align(*p) for p in pairs]
    return alignedpairs

def cls_align(pairs, align_symbol=ALIGN_SYMBOL, multiword=True, **kwargs):
    def _cls_align(ins, outs):
        len_ins  = len(ins)
        len_outs = len(outs)
        LCSuff = [[0 for k in range(len_outs+1)] for l in range(len_ins+1)]
        cls_length = 0
        pointer = 0, 0
        for i in range(len_ins + 1):
            for j in range(len_outs + 1):
                if i == 0 or j == 0:
                    LCSuff[i][j] == 0
                elif ins[i-1] == outs[j-1]:
                    LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                    if LCSuff[i][j] > cls_length:
                        cls_length = LCSuff[i][j]
                        pointer = i-1, j-1
                else:
                    LCSuff[i][j] = 0
        aligned_ins, aligned_outs = ins, outs
        # cls'es should be aligned, the rest aligned and padded
        # pad from the left
        offset = pointer[0] - pointer[1]
        if offset > 0:
            # the cls starts later in ins, and so outs
            # needs to be padded.
            aligned_outs = align_symbol * offset + aligned_outs
        elif offset < 0:
            aligned_ins = align_symbol * abs(offset) + aligned_ins
        # pad from the right
        length_diff = len_ins - len_outs - offset
        if length_diff > 0:
            aligned_outs += align_symbol * length_diff
        elif length_diff < 0:
            aligned_ins += align_symbol * abs(length_diff)
        return aligned_ins, aligned_outs
    return multiword_align(pairs, _cls_align, multiword)

if __name__ == "__main__":
    
    def seq_of_pairs_unicode(l):
        return u', '.join(u'({}, {})'.format(*p) for p in l)
        
    pairs = (('walk', 'walked'), ('fliegen', 'flog'), (u'береза', u'берёз'), (u'집', u'집'),
             ('sing', 'will sing'), (u'белый хлеб', u'белого хлеба'))
    dumb_targets_nomulti = (('walk~~', 'walked'), ('fliegen', 'flog~~~'), (u'береза', u'берёз~'),
                            (u'집', u'집'), ('sing~~~~~', 'will sing'), (u'белый хлеб~~', u'белого хлеба'))
    dumb_targets = (('walk~~', 'walked'), ('fliegen', 'flog~~~'), (u'береза', u'берёз~'),
                    (u'집', u'집'), ('sing~~~~~', 'will sing'), (u'белый~ хлеб~', u'белого хлеба'))
    cls_targets = (('walk~~', 'walked'), ('fliegen', 'flog~~~'), (u'береза', u'берёз~'), (u'집', u'집'),
               ('~~~~~sing', 'will sing'), (u'белый~ хлеб~', u'белого хлеба'))
    for alignment, aligned_pairs, targets in (('DUMB NO MULTI', dumb_align(pairs, multiword=False), dumb_targets_nomulti),
                                              ('DUMB MULTI', dumb_align(pairs, multiword=True), dumb_targets),
                                              ('CLS', cls_align(pairs), cls_targets)):
        print 'Alignment: {}'.format(alignment)
        print u'Pairs:         {}'.format(seq_of_pairs_unicode(pairs))
        print u'Aligned pairs: {}'.format(seq_of_pairs_unicode(aligned_pairs))
        print u'Targets:       {}'.format(seq_of_pairs_unicode(targets))
        for a, t in zip(aligned_pairs, targets):
            assert a == t, u'Mismatch: {} and {}'.format(a, t)
        print 'All matches.'