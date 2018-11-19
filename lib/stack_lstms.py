import dynet as dy

#############################################################
# Stack RNNs and biRNNs
#############################################################

# from Chris Dyer and Co.'s EMNLP 2016 tutorial:
class StackRNN(object):
    def __init__(self, rnn, p_empty_embedding = None):
        self.s = [(rnn.initial_state(), None)]
        self.empty = None
        if p_empty_embedding:
            self.empty = dy.parameter(p_empty_embedding)
    def push(self, expr, extra=None):
        self.s.append((self.s[-1][0].add_input(expr), extra))
    def pop(self):
        return self.s.pop()[1] # return "extra" (i.e., whatever the caller wants or None)
    def embedding(self):
        # work around since inital_state.output() is None
        return self.s[-1][0].output() if len(self.s) > 1 else self.empty
    def __len__(self):
        return len(self.s) - 1


class DeleteRNN(StackRNN):
    def clear_all(self):
        self.s = self.s[:1]


class StackBiRNN(object):
    def __init__(self, frnn, brnn, p_empty_embedding = None):
        self.frnn = frnn
        self.brnn = brnn
        self.empty = None
        if p_empty_embedding:
            self.empty = dy.parameter(p_empty_embedding)
    def transduce(self, embs, extras=None):
        fs = self.frnn.initial_state()
        bs = self.brnn.initial_state()
        fs_states = fs.add_inputs(embs)  # 1, 2, 3, 4
        bs_states = reversed(bs.add_inputs(reversed(embs)))   # 1, 2, 3, 4
        self.s = [(fs, bs, None)] + reversed(zip(fs_states, bs_states, extras))  # 0, 4, 3, 2, 1
    def pop(self):
        return self.s.pop()[-1] # return "extra" (i.e., whatever the caller wants or None)
    def embedding(self):
        if len(self.s) > 1:
            fs = self.s[-1][0].output()
            bs = self.s[-1][1].output()
            emb = dy.concatenate([fs, bs])
        else:
            # work around since inital_state.output() is None
            emb = self.empty
        return emb
    def __len__(self):
        return len(self.s) - 1

class Encoder(object):
    def __init__(self, frnn, brnn):
        self.frnn = frnn
        self.brnn = brnn
    def transduce(self, embs, extras=None):
        fs = self.frnn.initial_state()
        bs = self.brnn.initial_state()
        fs_states = fs.add_inputs(embs)   # 1, 2, 3, 4
        bs_states = reversed(bs.add_inputs(reversed(embs)))  # 1, 2, 3, 4
        self.s = list(reversed(zip(fs_states, bs_states, extras)))  # 4, 3, 2, 1
        # special treatment for the final element
        final_s = self.s[0]
        self.final_embedding = dy.concatenate([final_s[0].output(),
                                               final_s[1].output()])
        self.final_extra = final_s[2]
    def embedding(self, extra=False):
        if len(self.s) > 1:
            fs, bs, e = self.s[-1]
            output = dy.concatenate([fs.output(), bs.output()])
        else:
            e = self.final_extra
            output = self.final_embedding
        if extra:
            output = output, e
        return output
    def get_extra(self):
        return [e for _, _, e in reversed(self.s)]
    def pop(self):
        return self.s.pop()[-1] # return "extra" (i.e., whatever the caller wants or None)
    def __len__(self):
        return len(self.s)
    def copy(self):
        encoder = Encoder(self.frnn, self.brnn)
        encoder.s = list(self.s)  # copy
        encoder.final_embedding = self.final_embedding
        encoder.final_extra = self.final_extra
        return encoder