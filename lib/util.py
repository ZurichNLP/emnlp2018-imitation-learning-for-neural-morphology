import os
import codecs

#import datasets
from defaults import EVALM_PATH, COPY, DELETE, END_WORD, END_WORD_CHAR, BEGIN_WORD_CHAR

def write_stats_file(dev_accuracy, paths, data_arguments, model_arguments, optim_arguments):

    with open(paths['stats_file_path'], 'w') as w:

        print >> w, 'LANGUAGE: {}, REGIME: {}'.format(paths['lang'], paths['regime'])
        print >> w, 'Train path:   {}'.format(paths['train_path'])
        print >> w, 'Dev path:     {}'.format(paths['dev_path'])
        print >> w, 'Test path:    {}'.format(paths['test_path'])
        print >> w, 'Results path: {}'.format(paths['results_file_path'])

        for k, v in paths.iteritems():
            if k not in ('lang', 'regime', 'train_path', 'dev_path',
                         'test_path', 'results_file_path'):
                print >> w, '{:20} = {}'.format(k, v)
        print >> w

        for name, args in (('DATA ARGS:', data_arguments),
                           ('MODEL ARGS:', model_arguments),
                           ('OPTIMIZATION ARGS:', optim_arguments)):
            print >> w, name
            for k, v in args.iteritems():
                print >> w, '{:20} = {}'.format(k, v)
            print >> w

        print >> w, 'DEV ACCURACY (internal evaluation) = {}'.format(dev_accuracy)


def external_eval(output_path, gold_path, batches, predictions, sigm2017format, evalm_path=EVALM_PATH):

    pred_path = output_path + 'predictions'
    eval_path = output_path + 'eval'

    if sigm2017format is True:
        line = u'{LEM}\t{PRE}\t{FET}\n'
        format_flag = ''
        merge_keys_flag = '--merge_same_keys'
    else:
        line = u'{LEM}\t{FET}\t{PRE}\n'
        format_flag = '--format2016'
        merge_keys_flag = ''

    # WRITE FILE WITH PREDICTIONS
    with codecs.open(pred_path, 'w', encoding='utf8') as w:
        for sample, prediction in zip((s for b in batches for s in b), predictions):
            w.write(line.format(LEM=sample.lemma_str, PRE=prediction, FET=sample.feat_str))

    # CALL EXTERNAL EVAL SCRIPT
    os.system('python {} --gold {} --guesses {} {} {} | tee {}'.format(
            evalm_path, gold_path, pred_path, format_flag, merge_keys_flag, eval_path))


def alignment(lemma, prediction, actions, action_string, EPSILON=u"\u2002"):
    # lemma as str, prediction as str, actions as list of indices
    # pad prediction
    lemma = BEGIN_WORD_CHAR + lemma + END_WORD_CHAR
    prediction = BEGIN_WORD_CHAR + prediction + END_WORD_CHAR
    assert EPSILON not in lemma, lemma
    assert EPSILON not in prediction, prediction
    alignment = []
    x = 0
    y = 0
    len_actions = len(actions)
    for i, a in enumerate(actions):
        # there are as many alignment pairs as there are actions
        if a == COPY or ((i + 1) == len_actions and a == END_WORD):
            alignment_pair = lemma[x], prediction[y]
            x += 1
            y += 1
        elif a == DELETE:
            alignment_pair = lemma[x], EPSILON
            x += 1
        else:  # inertion
            alignment_pair = EPSILON, prediction[y]
            assert action_string[i] == prediction[y], (i, action_string, y, prediction)
            y += 1
        alignment.append(alignment_pair)
    # finished
    if actions[-1] == END_WORD:
        assert x == len(lemma), (x, len(lemma))
        assert y == len(prediction), (y, len(prediction))
    return alignment
