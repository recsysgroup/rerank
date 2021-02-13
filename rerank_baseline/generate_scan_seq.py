import sys
import utils

qid2list = {}

ARG_type, ARG_do_bounce, ARG_seq_len = sys.argv[1], sys.argv[2], sys.argv[3]
ARG_seq_len = int(ARG_seq_len)

for line in sys.stdin:
    parts = line.split(' ')
    score = float(parts[0])
    rating = int(parts[1])
    qid = parts[2]
    vec = utils.to_dense(parts[3:])
    record = utils.Record(qid, rating, vec, score=score)
    seq = qid2list.get(qid)
    if seq is None:
        seq = []
        qid2list[qid] = seq

    seq.append(record)

for qid, seq in qid2list.items():
    if len(seq) < ARG_seq_len:
        continue
    seq.sort(key=lambda x: x.score, reverse=True)
    if len(seq) > ARG_seq_len:
        seq = seq[0:ARG_seq_len]

    fb_seq = utils.gen_scan_seq(seq)

    seq_len = len(seq)
    with_end = False
    for index, (_, bounce) in enumerate(fb_seq):
        if bounce == 1:
            seq_len = index + 1
            with_end = True
            break

    if ARG_do_bounce == 'true':
        do_bounce = True
    else:
        do_bounce = False

    if ARG_type == 'seq':
        ratings_str = ','.join([str(o.rating) for o in seq])
        vec_str = ','.join([':'.join(str(s) for s in o.vec) for o in seq])
        ret = []
        ret.append(qid)
        ret.append(str(seq_len))
        ret.append(ratings_str)
        ret.append(','.join([str(o[0]) for o in fb_seq]))
        ret.append(','.join([str(o[1]) for o in fb_seq]))
        ret.append(vec_str)
        print ';'.join(ret)

    elif ARG_type == 'single':
        for index, o in enumerate(seq):
            ret = []
            ret.append(str(fb_seq[index][0]))
            ret.append(qid)
            ret.append(utils.to_sparse(o.vec))
            ret.append('#')
            ret.append('rating=' + str(o.rating))
            print ' '.join(ret)

            if do_bounce and fb_seq[index][1] == 1:
                break
