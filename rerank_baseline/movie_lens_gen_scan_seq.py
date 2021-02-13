import numpy as np
import sys
import utils

movie_tag_file_name = 'movie_lens/genome_scores_format.txt'

movie_2_vec = {}
with open(movie_tag_file_name, 'r') as f:
    for line in f.readlines():
        movie_id, vec_str = line.split(',')
        vec = vec_str.split(':')
        movie_2_vec[int(movie_id)] = np.array([float(o) for o in vec])

user_2_behaviors = {}

for line in sys.stdin:
    user_id, movie_id, rating, time_stamp = line.split(',')
    user_id = int(user_id)
    movie_id = int(movie_id)
    rating = float(rating)
    time_stamp = int(time_stamp)

    behaviors = user_2_behaviors.get(user_id)
    if behaviors is None:
        behaviors = []
        user_2_behaviors[user_id] = behaviors

    vec = movie_2_vec.get(movie_id)
    if vec is not None:
        record = utils.Record(movie_id, rating, vec, score=time_stamp)
        behaviors.append(record)


def vec_to_string(vec):
    vec_list = vec.tolist()
    _vec_str = [str(o) for o in vec_list]
    return ':'.join(_vec_str)


for user_id, behaviors in user_2_behaviors.items():
    if len(behaviors) < 50:
        continue
    behaviors = sorted(behaviors, key=lambda x: x.score)

    last_behaviors = behaviors[0:30]
    last_click_ids = []
    for o in last_behaviors:
        if o.rating > 4.0:
            last_click_ids.append(o.qid)

    new_behaviors = behaviors[30:50]

    fb_seq = utils.movie_lens_gen_scan_seq(new_behaviors)
    seq_len = 0
    for o in fb_seq:
        seq_len += 1
        if o[1] == 1:
            break

    ret = []
    # last_clicks
    ret.append(','.join([str(o) for o in last_click_ids]))
    # seq_len
    ret.append(str(seq_len))
    # ratings
    ret.append(','.join([str(o.rating) for o in new_behaviors]))
    # clicks
    ret.append(','.join([str(o[0]) for o in fb_seq]))
    # bounces
    ret.append(','.join([str(o[1]) for o in fb_seq]))
    # ids
    ret.append(','.join([str(o.qid) for o in new_behaviors]))
    # vecs
    ret.append(','.join([vec_to_string(o.vec) for o in new_behaviors]))

    print ';'.join(ret)
