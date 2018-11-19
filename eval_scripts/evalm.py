#!/usr/bin/env python
"""
Official Evaluation Script for the SIGMORPHON 2016 Shared Task.

Returns accuracy, mean Levenhstein distance and mean reciprocal rank.

Author: Ryan Cotterell and Mans Hulden
Last Update: 12/01/2015

Changed: Peter Makarov
Last Update: 02/01/2018
"""

import sys
import numpy as np
import codecs
from collections import defaultdict as dd


def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2)+1, len(str1)+1])
    for x in xrange(1, len(str2) + 1):
        m[x][0] = m[x-1][0] + 1
    for y in xrange(1, len(str1) + 1):
        m[0][y] = m[0][y-1] + 1
    for x in xrange(1, len(str2) + 1):
        for y in xrange(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
    return int(m[len(str2)][len(str1)])


def evaluate(gold, guess):
    " Evaluates a single tag "
    
    # best guess
    best = guess[0]
    # compute the metrics
    acc = 1.0 if best == gold else 0
    # edit distance
    lev = distance(best, gold)    
    # reciprocal rank
    rank = 0.0

    try:
        rank = 1.0/(guess.index(gold)+1)
    except ValueError:
        # 1.0 / oo = 0
        rank = 0.0
    
    return float(acc), float(lev), float(rank)


def aggregate(golden, guesses):
    """ Aggregates over the results """
    
    breakdown = dd(lambda : (0.0, 0.0, 0.0, 0.0))
    breakdown_N = dd(int)
    N = 0
    A, L, NL, R = 0.0, 0.0, 0.0, 0.0
    for tag, gold in golden.items():
        
        if tag in guesses:
            guess = guesses[tag]
            # assumes only 1 golden analysis
            if len(gold) > 1:
                for elem in gold[1:]:
                    assert elem == gold[0], u"gold: {}, tag: {}".format(', '.join(gold), tag).encode('utf8')
                    
            acc, lev, rank = evaluate(gold[0], guess)
            A, L, NL, R = A+acc, L+lev, NL+(lev / len(gold[0])), R+rank
            # compute results broken down by POS tag
            pos = tag[-1].split(",")[0].replace("pos=", "")
            _A, _L, _NL, _R = breakdown[pos]
            breakdown[pos] = _A+acc, _L+lev, _NL+(lev / len(gold[0])), _R+rank
            breakdown_N[pos] += 1
            
        else:
            sys.stderr.write("warning: no guess provided for (%s)\n" % " ".join(tag))
        N += 1

    return A/N, L/N, NL/N, R/N, breakdown, breakdown_N


def read_file(file_in, is_format2016=False, merge_same_keys=False):
    """ Read in the users' guesses """
    
    if is_format2016:
        # format of sigmorphon 2016
        key_ids = 0, 1
        value_id = 2        
    else:
        key_ids = 0, 2
        value_id = 1
        
    with codecs.open(file_in, 'rb', encoding="utf-8") as f:
        guesses = dd(list)
        for i, line in enumerate(f):
            linest = line.strip()
            l = linest.split('\t')  # whitespace is a char!
            if len(l) != 3:
                print 'Deleted too much!'
                l = l + ['']*(3-len(l))
            guess_id = [l[k] for k in key_ids]
            if merge_same_keys:
                guess_id = tuple(guess_id)
            else:
                # keep same (string, feats) keys separate by adding line id to the key
                guess_id = tuple([i] + guess_id)
            guess = l[value_id]
            guesses[guess_id].append(guess)

    return dict(guesses)


def pp(A, L, NL, R):
    print "Accuracy:", A
    print "Mean Levenshtein:", L
    print "Mean Normalized Levenshtein:", NL
    print "Mean Reciprocal Rank:", R
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SIGMORPHON 2016 Shared Task Evaluation')
    parser.add_argument("--golden")
    parser.add_argument("--guesses")
    parser.add_argument("--format2016", default=False, action='store_true')
    parser.add_argument("--merge_same_keys", default=False, action='store_true')
    args = parser.parse_args()
    
    print 'Evaluation script arguments: ', args
    print
    golden = read_file(args.golden, is_format2016=args.format2016, merge_same_keys=args.merge_same_keys)
    guesses = read_file(args.guesses, is_format2016=args.format2016, merge_same_keys=args.merge_same_keys)

    try:
        A, L, NL, R, breakdown, breakdown_N = aggregate(golden, guesses)
        for tag, (_A, _L, _NL, _R) in breakdown.items():
            print tag
            _N = breakdown_N[tag]
            pp(_A / _N, _L / _N, _NL / _N, _R / _N)
            print
        print "Aggregate"
        pp(A, L, NL, R)
    except ZeroDivisionError, e:
        print 'Likely, using test file without gold predictions. Returning nothing!'