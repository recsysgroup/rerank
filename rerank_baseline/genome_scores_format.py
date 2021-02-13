import sys
import numpy as np
import math

movie_2_vec = {}

for line in sys.stdin:
    movie_id, tag_id, relevance = line.split(',')
    movie_id = int(movie_id)
    tag_index = int(tag_id) - 1
    relevance = float(relevance)

    vec = movie_2_vec.get(movie_id)
    if vec is None:
        vec = np.zeros([1128], dtype=float)
        movie_2_vec[movie_id] = vec

    vec[tag_index] = relevance

for movie_id, vec in movie_2_vec.items():
    vec = vec / (math.sqrt(np.sum(vec * vec)) + 1e-8)
    vec_str = [str(score) for score in vec.tolist()]

    print '{},{}'.format(movie_id, ':'.join(vec_str))
