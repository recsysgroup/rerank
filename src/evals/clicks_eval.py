# -*- coding: utf-8 -*-
from trainers.train_yahoo_with_rl import Reinforce
import numpy as np
import tensorflow as tf
import rerank_baseline.utils as re_utils
from trainers.common import cal_view_deep, cal_click_num, cal_expection, cal_map


class Evaluator(Reinforce):
    def __init__(self, config, features, PolicyNetwork, SimulatorNetwork, simulator_checkpoint_path):
        super(Evaluator, self).__init__(config, features, PolicyNetwork, SimulatorNetwork, simulator_checkpoint_path)
        self.config = config
        self.features = features
        self.batch_size = config.get('eval_batch_size')
        self.batch_num = config.get('eval_batch_num')

    def eval(self, session):

        import time
        expections = []
        fb_seq_list = []
        s1 = time.time()
        for round in range(self.batch_num):
            _features, auc = session.run([self.features, self.env_for_train.auc])

            # max生成序列
            selected_indexes, _ = self.step_by_model(session, _features, self.batch_size, self.max_sample)

            # 送入评估器，得到点击率和跳失率
            ctr_list, pbr_list = self.estimator_by_env(session, _features, selected_indexes, self.batch_size)

            rewards = cal_expection(ctr_list, pbr_list, _features.get('features_mask'))
            expections.extend(rewards)

            ratings = _features.get('ratings')
            feature_len = _features.get('features_mask')
            feature = _features.get('features')

            s2 = time.time()
            print 'round {}: {}'.format(round, s2 - s1)

            for b_index in range(self.batch_size):
                size = min(self.rank_size, int(feature_len[b_index]))
                seq = []
                for r_index in range(size):
                    s_index = selected_indexes[b_index][r_index]
                    vec = feature[b_index][s_index]
                    rating = int(ratings[b_index][s_index])
                    seq.append(re_utils.Record('', rating, vec))

                fb_seq = re_utils.gen_scan_seq(seq)
                fb_seq_list.append(fb_seq)

            s2 = time.time()
            print 'round {}: {}'.format(round, s2 - s1)

        print('{} is {}'.format('ctr auc', auc['ctr'][1]))
        print('{} is {}'.format('pbr auc', auc['pbr'][1]))
        print 'expection is {}'.format(np.mean(expections))
        print 'map is {}'.format(cal_map(fb_seq_list))
        print 'click is {}'.format(cal_click_num(fb_seq_list))
        print 'view_deep is {}'.format(cal_view_deep(fb_seq_list))

    def max_sample(self, action_distribution):
        sample_index = []
        for o in action_distribution:
            max_val = -1.0
            max_index = 0
            for i in range(len(o)):
                if o[i] > max_val:
                    max_val = o[i]
                    max_index = i
            sample_index.append(max_index)
        return sample_index
