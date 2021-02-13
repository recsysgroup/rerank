import math
import numpy as np


class Record(object):
    def __init__(self, qid, rating, vec, score=0.0):
        self.qid = qid
        self.rating = rating
        self.vec = vec
        self.score = score


def to_dense(parts):
    vec = []
    for o in parts:
        if o == '#':
            break
        fid, value = o.split(':')
        vec.append(float(value))
    return vec


def to_sparse(vec):
    ret = []
    for i, o in enumerate(vec):
        ret.append('{}:{}'.format(i + 1, o))
    return ' '.join(ret)


def dis(f1, f2):
    return math.sqrt(np.sum((f1 - f2) * (f1 - f2)))


'''
output format: [(click,bounce),()]
'''


def gen_scan_seq(seq):
    feedback_seq = []
    _acc_min_dis = 0.0
    for index, val in enumerate(seq):
        val.vec = np.array(val.vec)
        val.vec = val.vec / (math.sqrt(np.sum(val.vec * val.vec)) + 1e-8)
        click = 0
        bounce = 0
        if val.rating >= 3:
            click = 1

        _min_dis = 1.0
        for j in range(index):
            o = seq[j]
            _dis = dis(val.vec, o.vec)
            _min_dis = min(_min_dis, _dis)

        _acc_min_dis += _min_dis * 0.9 + val.rating * 0.1
        decay_rate = _acc_min_dis / (index + 1.0)
        # print 'decay',index,decay_rate

        if decay_rate < 0.8:
            bounce = 1
        feedback_seq.append((click, bounce))
    # print 'decay ######'
    return feedback_seq


def gen_scan_seq_bak(seq):
    feedback_seq = []
    _acc_min_dis = 0.0
    for index, val in enumerate(seq):
        val.vec = np.array(val.vec)
        val.vec = val.vec / (math.sqrt(np.sum(val.vec * val.vec)) + 1e-8)
        click = 0
        bounce = 0
        if val.rating >= 3:
            click = 1

        _min_dis = 1.0
        for j in range(index):
            o = seq[j]
            _dis = dis(val.vec, o.vec)
            _min_dis = min(_min_dis, _dis)

        _acc_min_dis += _min_dis
        decay_rate = _acc_min_dis / (index + 1.0)
        # print 'decay',index,decay_rate

        if decay_rate < 0.7:
            bounce = 1
        feedback_seq.append((click, bounce))
    # print 'decay ######'
    return feedback_seq


def movie_lens_gen_scan_seq(seq):
    feedback_seq = []
    _acc_mmr = 0.0
    for index, val in enumerate(seq):
        click = 0
        bounce = 0
        if val.rating >= 4.0:
            click = 1

        _min_dis = 1.0
        for j in range(index):
            o = seq[j]
            _dis = dis(val.vec, o.vec)
            _min_dis = min(_min_dis, _dis)

        _acc_mmr += _min_dis
        decay_rate = _acc_mmr / (index + 1) ** 1.4

        # print 'decay', index, decay_rate

        if decay_rate < 0.25:
            bounce = 1

        feedback_seq.append((click, bounce))
    # print 'decay ####'

    return feedback_seq
