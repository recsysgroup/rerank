import sys
import numpy as np
import math
import utils



def map(qid2list):
    def ap(seq):
        pos = 0.0
        ap = 0.0
        view = 0.0
        fb_seq = utils.gen_scan_seq(seq)
        for index, (click, bounce) in enumerate(fb_seq):
            view += 1.0
            if click > 0:
                pos += click
                ap += pos / (index + 1)

            if bounce == 1:
                break
        return ap / view

    vals = []
    for qid, seq in qid2list.items():
        val = ap(seq)
        vals.append(val)

    return np.mean(vals)


def click_num(qid2list):
    def one_click_num(seq):
        _click_num = 0.0

        fb_seq = utils.gen_scan_seq(seq)
        for click, bounce in fb_seq:
            _click_num += click
            if bounce == 1:
                break
        return _click_num

    vals = []
    for qid, seq in qid2list.items():
        val = one_click_num(seq)
        vals.append(val)

    return np.mean(vals)


def view_deep(qid2list):
    def one_view_deep(seq):
        _view_deep = 0.0

        fb_seq = utils.gen_scan_seq(seq)
        for click, bounce in fb_seq:
            _view_deep += 1
            if bounce == 1:
                break
        return _view_deep

    vals = []
    for qid, seq in qid2list.items():
        val = one_view_deep(seq)
        vals.append(val)

    return np.mean(vals)


def main():
    qid2list = {}
    inc = 0
    for line in sys.stdin:
        parts = line.split(' ')
        score = float(parts[0])
        rating = int(parts[-1].split('=')[1])
        qid = parts[2]
        vec = utils.to_dense(parts[3:])
        seq = qid2list.get(qid)
        if seq is None:
            seq = []
            qid2list[qid] = seq

        record = utils.Record(qid, rating, vec, score=score)
        seq.append(record)
        inc += 1

    for qid, seq in qid2list.items():
        seq.sort(key=lambda x: x.score, reverse=True)

    print 'click_num', click_num(qid2list)
    print 'view_deep', view_deep(qid2list)
    print 'map', map(qid2list)


if __name__ == '__main__':
    main()
