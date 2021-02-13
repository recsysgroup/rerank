# -*- coding: utf-8 -*-
import tensorflow as tf
import constant as C
import numpy as np
from policy_network import fm_with_gru
from utils import get_assignment_map_from_checkpoint
import math
import rerank_baseline.utils as re_utils
from common import cal_view_deep, cal_click_num, cal_expection, cal_map


class Reinforce(object):
    def __init__(self, config, features, PolicyNetwork, SimulatorNetwork, simulator_checkpoint_path):
        self.config = config
        self.rank_size = config.get('rank_size')
        self.env_features = self.buildSimulationPlaceholder(config, self.rank_size)
        self.model_features = self.buildModelPlaceholder(config, self.rank_size)

        with tf.variable_scope("simulator", reuse=tf.AUTO_REUSE):
            self.env = SimulatorNetwork(config, self.env_features, config.get('rank_size'), training=False)
            self.env_for_train = SimulatorNetwork(config, features, config.get('rank_size'), training=True)

        with tf.variable_scope("reinforce"):
            self.model = PolicyNetwork(config, self.model_features, config.get('rank_size'), training=True)

        # if simulator_checkpoint_path is not None:
        #     t_vars = tf.trainable_variables()
        #     (assignment_map, initialized_variable_names
        #      ) = get_assignment_map_from_checkpoint(t_vars, simulator_checkpoint_path)
        #
        #     tf.train.init_from_checkpoint(simulator_checkpoint_path, assignment_map)
        #
        #     tf.logging.info("**** Trainable Variables ****")
        #     for var in t_vars:
        #         init_string = ""
        #         if var.name in initialized_variable_names:
        #             init_string = ", *INIT_FROM_CKPT*"
        #         tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                         init_string)

    def step_by_model_full(self, session, _features, batch_size, sample_fn):
        rank_size = self.rank_size
        features_mask = _features.get('features_mask')
        features = _features.get('features')

        _feed_dict = {}
        _feed_dict[self.model_features.get('features')] = features
        _feed_dict[self.model_features.get('features_mask')] = features_mask

        point_vec = session.run(self.model.point_vec, feed_dict=_feed_dict)

        rank_mask = np.ones([batch_size, rank_size], dtype=float)
        for b_index in range(batch_size):
            for r_index in range(int(features_mask[b_index]), rank_size):
                rank_mask[b_index][r_index] = 0.5
        _feed_dict[self.model_features.get('rank_mask')] = rank_mask

        zero = [0.0 for _ in range(699)]
        selected_vec = [[] for _ in range(batch_size)]
        selected_vec_mask = ['0' for _ in range(batch_size)]
        for b_index in range(batch_size):
            selected_vec_mask[b_index] = '0'
            for r_index in range(rank_size):
                selected_vec[b_index].append(zero)

        _feed_dict[self.model_features.get('selected_vec')] = selected_vec
        _feed_dict[self.model_features.get('selected_vec_mask')] = selected_vec_mask

        selected_index = [[] for _ in range(batch_size)]

        for index in range(rank_size):
            _action_distribution = session.run([self.model.action_distribution],
                                               feed_dict=_feed_dict)
            sample_index = sample_fn(_action_distribution)
            for b_index in range(batch_size):
                selected_index[b_index].append(sample_index[b_index])
                rank_mask[b_index][sample_index[b_index]] = 0.0
                selected_vec_mask[b_index] = str(index + 1)
                selected_vec[b_index][index] = features[b_index][sample_index[b_index]]

        return selected_index

    def step_by_model(self, session, _features, batch_size, sample_fn):
        rank_size = self.rank_size

        features_mask = _features.get('features_mask')
        features = _features.get('features')

        _feed_dict = {}
        _feed_dict[self.model_features.get('features')] = features
        _feed_dict[self.model_features.get('features_mask')] = features_mask

        point_vec = session.run(self.model.point_vec, feed_dict=_feed_dict)

        _feed_dict = {}
        rank_mask = np.ones([batch_size, rank_size], dtype=float)
        for b_index in range(batch_size):
            for r_index in range(int(features_mask[b_index]), rank_size):
                rank_mask[b_index][r_index] = 0.5
        _feed_dict[self.model_features.get('rank_mask')] = rank_mask

        last_state = [[0 for i in range(128)] for _ in range(batch_size)]
        _feed_dict[self.model_features.get('last_state')] = last_state
        _feed_dict[self.model_features.get('point_vec')] = point_vec

        selected_vec_mask = ['0' for _ in range(batch_size)]
        _feed_dict[self.model_features.get('selected_vec_mask')] = selected_vec_mask
        last_vec = [features[i][0] for i in range(batch_size)]
        _feed_dict[self.model_features.get('last_vec')] = last_vec

        selected_index = [[] for _ in range(batch_size)]
        selected_probability = [[] for _ in range(batch_size)]

        for index in range(rank_size):
            action_distribution, next_state = session.run(
                [self.model.inc_action_distribution, self.model.next_state],
                feed_dict=_feed_dict)
            sample_index = sample_fn(action_distribution)
            # 标记rank_mask
            for b_index in range(batch_size):
                selected_vec_mask[b_index] = str(index + 1)
                rank_mask[b_index][sample_index[b_index]] = 0.0
                selected_index[b_index].append(sample_index[b_index])
                selected_probability[b_index].append(action_distribution[b_index][sample_index[b_index]])
                last_state[b_index] = next_state[b_index]
                last_vec[b_index] = features[b_index][sample_index[b_index]]

        return selected_index, selected_probability

    def estimator_by_env(self, session, _features, selected_indexes, batch_size):
        rank_size = self.rank_size

        _feed_dict = {}

        features = _features.get('features')
        list_vector = [[] for _ in range(batch_size)]
        for b_index in range(batch_size):
            for r_index in range(rank_size):
                s_index = selected_indexes[b_index][r_index]
                list_vector[b_index].append(features[b_index][s_index])

        _feed_dict[self.env_features.get('features')] = list_vector
        _feed_dict[self.env_features.get('features_mask')] = _features.get('features_mask')
        _feed_dict[self.env_features.get('seq_len')] = _features.get('features_mask')
        _ctr, _pbr = session.run([self.env.ctr_prediction, self.env.pbr_prediction], feed_dict=_feed_dict)

        return _ctr, _pbr

    def buildModelPlaceholder(self, config, rank_size):
        pla_features = {}

        pla_features['features'] = tf.placeholder(tf.float32, shape=[None, rank_size, 699], name='features')
        pla_features['features_mask'] = tf.placeholder(tf.string, shape=[None], name='features_mask')

        pla_features['selected_vec'] = tf.placeholder(tf.float32, shape=[None, rank_size, 699], name='selected_vec')
        pla_features['selected_vec_mask'] = tf.placeholder(tf.string, shape=[None], name='selected_vec_mask')
        pla_features['rank_mask'] = tf.placeholder(tf.float32, shape=[None, rank_size], name='rank_mask')
        pla_features['point_vec'] = tf.placeholder(tf.float32, shape=[None, rank_size, 128], name='point_vec')
        pla_features['last_state'] = tf.placeholder(tf.float32, shape=[None, 128], name='last_state')
        pla_features['last_vec'] = tf.placeholder(tf.float32, shape=[None, 699], name='last_vec')

        return pla_features

    def buildSimulationPlaceholder(self, config, rank_size):
        pla_features = {}
        pla_features['seq_len'] = tf.placeholder(tf.string, shape=[None], name='seq_len')
        pla_features['features_mask'] = tf.placeholder(tf.string, shape=[None], name='features_mask')
        pla_features['features'] = tf.placeholder(tf.float32, shape=[None, rank_size, 699], name='features')

        pla_features['point_vec'] = tf.placeholder(tf.float32, shape=[None, 699], name='point_vec')
        pla_features['ctr_state'] = tf.placeholder(tf.float32, shape=[None, 128], name='ctr_state')
        pla_features['pbr_state'] = tf.placeholder(tf.float32, shape=[None, 128], name='pbr_state')

        return pla_features


class Trainer(Reinforce):
    def __init__(self, config, features, PolicyNetwork, SimulatorNetwork, global_step, simulator_checkpoint_path,
                 eval_features=None):
        super(Trainer, self).__init__(config, features, PolicyNetwork, SimulatorNetwork, simulator_checkpoint_path)
        self.global_step = global_step
        self.config = config
        self.step_op = tf.train.get_or_create_global_step()
        self.features = features
        self.eval_features = eval_features
        self.batch_size = config.get('train_batch_size')
        self.eval_batch_size = config.get('eval_batch_size')
        self.eval_batch_num = config.get('eval_batch_num')

        self.action_index = tf.placeholder(tf.int32, shape=[None], name='action_index')
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.old_prob = tf.placeholder(tf.float32, shape=[None], name='old_prob')

        with tf.variable_scope("simulator", reuse=tf.AUTO_REUSE):
            env_for_train = SimulatorNetwork(config, features, config.get('rank_size'), training=True)
            env_loss = env_for_train.loss
            self.env_train_op = tf.train.AdagradOptimizer(learning_rate=0.01) \
                .minimize(env_loss, global_step=self.global_step)

        with tf.variable_scope("reinforce"):
            self.train_op = self.build_train_op(self.model, self.action_index, self.reward, self.old_prob)

        self.pool_size = 1280
        self.pool_index = 0
        self.pool = [None for _ in range(self.pool_size)]
        self.eval_features_cache = []

    def build_train_op(self, train_model, action_indexs, baseline_rewards, old_prob):
        # build loss function
        action_ont_hot = tf.one_hot(indices=action_indexs, depth=self.rank_size, on_value=1.0, off_value=0.0)
        action_prob = tf.reduce_sum(train_model.action_distribution * action_ont_hot, axis=-1)
        neg_log_prob = -tf.log(tf.clip_by_value(action_prob, clip_value_max=1.0, clip_value_min=1e-5))

        # total loss
        loss = tf.reduce_mean(neg_log_prob * baseline_rewards)

        learning_rate = self.config["init_learning_rate"]
        # learning_rate = tf.train.polynomial_decay(
        #     learning_rate,
        #     self.global_step,
        #     10000,
        #     end_learning_rate=1e-4,
        #     power=1.0,
        #     cycle=False)
        train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss, global_step=self.global_step)

        return train_op

    def build_train_op_clip(self, train_model, action_indexs, baseline_rewards, old_prob):
        # build loss function
        action_ont_hot = tf.one_hot(indices=action_indexs, depth=self.rank_size, on_value=1.0, off_value=0.0)
        action_prob = tf.reduce_sum(train_model.action_distribution * action_ont_hot, axis=-1)
        neg_log_prob = -tf.log(tf.clip_by_value(action_prob, clip_value_max=1.0, clip_value_min=1e-5))
        neg_log_old_prob = -tf.log(tf.clip_by_value(old_prob, clip_value_max=1.0, clip_value_min=1e-5))

        ratio = tf.exp(neg_log_old_prob - neg_log_prob)

        CLIPRANGE = 0.2

        pg_losses = -baseline_rewards * ratio

        pg_losses2 = -baseline_rewards * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        loss = tf.maximum(pg_losses, pg_losses2)

        learning_rate = self.config["init_learning_rate"]
        train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss, global_step=self.global_step)

        return train_op

    def train_batch(self, session, _features):
        _feed_dict = {}

        _feed_dict[self.model_features.get('rank_mask')] = _features.get('rank_mask')

        _feed_dict[self.model_features.get('features')] = _features.get('features')
        _feed_dict[self.model_features.get('features_mask')] = _features.get('features_mask')

        _feed_dict[self.model_features.get('selected_vec')] = _features.get('selected_vec')
        _feed_dict[self.model_features.get('selected_vec_mask')] = _features.get('selected_vec_mask')

        _feed_dict[self.action_index] = _features.get('action_index')

        # reward
        _feed_dict[self.reward] = _features.get('reward')
        _feed_dict[self.old_prob] = _features.get('old_prob')

        _, _global_step, _action_distribution = session.run(
            [self.train_op, self.global_step, self.model.action_distribution],
            feed_dict=_feed_dict)
        print 'step@{}'.format(_global_step)

    # train_clicks
    def train(self, session):

        session.run(self.env_train_op)
        # return
        #
        # _features, _ = session.run([self.features, tf.train.get_global_step()])
        #
        # # 采样生成序列
        #
        # tile_num = 8
        # tile_batch_size = self.batch_size * tile_num
        # _tile_features = {}
        # _tile_features['features'] = np.tile(_features.get('features'), (tile_num, 1, 1))
        # _tile_features['features_mask'] = np.tile(_features.get('features_mask'), tile_num)
        # _tile_features['clicks'] = np.tile(_features.get('clicks'), (tile_num, 1))
        # _tile_features['seq_len'] = np.tile(_features.get('seq_len'), tile_num)
        #
        # # remove
        # _tile_features['features_mask'] = _tile_features['seq_len']
        #
        # selected_indexes, selected_prob = self.step_by_model(session, _tile_features, tile_batch_size,
        #                                                      self.weight_sample)
        # ctr_list, pbr_list = self.estimator_by_env(session, _tile_features, selected_indexes, tile_batch_size)
        #
        # features_mask = _tile_features.get('features_mask')
        # seq_len = _tile_features.get('seq_len')
        #
        # # remove
        # features_mask = seq_len
        #
        # batch_rewards = []
        # clicks = _tile_features.get('clicks')
        # for b_index in range(tile_batch_size):
        #     reward = 0.0
        #     rewards = []
        #     for index in range(int(features_mask[b_index]) - 1, -1, -1):
        #         reward = reward * (1.0 - pbr_list[b_index][index])
        #
        #         s_index = selected_indexes[b_index][index]
        #         if s_index < int(seq_len[b_index]):
        #             if clicks[b_index][s_index] != '':
        #                 reward += float(clicks[b_index][s_index])
        #         rewards.append(reward)
        #
        #     rewards = rewards[::-1]
        #     batch_rewards.append(rewards)
        #
        # _rand_features = {}
        # tile_features = _tile_features['features']
        # _rand_features['features'] = tile_features
        # _rand_features['features_mask'] = _tile_features['features_mask']
        # selected_vec = [[] for _ in range(tile_batch_size)]
        # for b_index in range(tile_batch_size):
        #     for r_index in range(self.rank_size):
        #         s_index = selected_indexes[b_index][r_index]
        #         selected_vec[b_index].append(tile_features[b_index][s_index])
        # _rand_features['selected_vec'] = selected_vec
        #
        # rank_mask = np.ones((tile_batch_size, self.rank_size), dtype=float)
        # for b_index in range(tile_batch_size):
        #     for index in range(int(features_mask[b_index]), self.rank_size):
        #         rank_mask[b_index][index] = 0.5
        # _rand_features['rank_mask'] = rank_mask
        #
        # for random_pos in range(self.rank_size):
        #     action_index = []
        #     old_prob = []
        #     for b_index in range(tile_batch_size):
        #         action_index.append(selected_indexes[b_index][random_pos])
        #         old_prob.append(selected_prob[b_index][random_pos])
        #
        #     baseline_rewards = [0.0 for _ in range(tile_batch_size)]
        #     for b_index in range(self.batch_size):
        #         tmp_rewards = []
        #         for t_index in range(tile_num):
        #             a_index = t_index * self.batch_size + b_index
        #             if random_pos < int(features_mask[a_index]):
        #                 tmp_rewards.append(batch_rewards[a_index][random_pos])
        #             else:
        #                 tmp_rewards.append(0.0)
        #         tmp_rewards = tmp_rewards - np.mean(tmp_rewards)
        #         for t_index in range(tile_num):
        #             baseline_rewards[t_index * self.batch_size + b_index] = tmp_rewards[t_index]
        #
        #     _rand_features['selected_vec_mask'] = [str(random_pos) for _ in range(tile_batch_size)]
        #     _rand_features['action_index'] = action_index
        #     _rand_features['reward'] = baseline_rewards
        #     _rand_features['old_prob'] = old_prob
        #
        #     self.train_batch(session, _rand_features)
        #
        #     for b_index in range(tile_batch_size):
        #         rank_mask[b_index][selected_indexes[b_index][random_pos]] = 0.0

    # train_tile_reinforce_batch
    def train_tile_reinforce_batch(self, session):
        _features = session.run(self.features)

        # 采样生成序列

        tile_num = 8
        tile_batch_size = self.batch_size * tile_num
        _tile_features = {}
        _tile_features['features'] = np.tile(_features.get('features'), (tile_num, 1, 1))
        _tile_features['features_mask'] = np.tile(_features.get('features_mask'), tile_num)

        selected_indexes, selected_prob = self.step_by_model(session, _tile_features, tile_batch_size,
                                                             self.weight_sample)
        ctr_list, pbr_list = self.estimator_by_env(session, _tile_features, selected_indexes, tile_batch_size)

        seq_len = _tile_features.get('features_mask')

        batch_keeps = []
        for b_index in range(tile_batch_size):
            keep = 1.0
            keeps = []
            for index in range(int(seq_len[b_index])):
                keeps.append(keep)
                keep *= (1.0 - pbr_list[b_index][index])
            batch_keeps.append(keeps)

        batch_rewards = []
        clicks = _features.get('clicks')
        for b_index in range(tile_batch_size):
            reward = 0.0
            rewards = []
            for index in range(int(seq_len[b_index]) - 1, -1, -1):
                reward = reward * (1.0 - pbr_list[b_index][index])
                reward += ctr_list[b_index][index]
                rewards.append(reward)

            rewards = rewards[::-1]
            batch_rewards.append(rewards)

        _rand_features = {}
        tile_features = _tile_features['features']
        _rand_features['features'] = tile_features
        _rand_features['features_mask'] = _tile_features['features_mask']
        selected_vec = [[] for _ in range(tile_batch_size)]
        for b_index in range(tile_batch_size):
            for r_index in range(self.rank_size):
                s_index = selected_indexes[b_index][r_index]
                selected_vec[b_index].append(tile_features[b_index][s_index])
        _rand_features['selected_vec'] = selected_vec

        rank_mask = np.ones((tile_batch_size, self.rank_size), dtype=float)
        for b_index in range(tile_batch_size):
            for index in range(int(seq_len[b_index]), self.rank_size):
                rank_mask[b_index][index] = 0.5
        _rand_features['rank_mask'] = rank_mask

        for random_pos in range(self.rank_size):
            action_index = []
            old_prob = []
            for b_index in range(tile_batch_size):
                action_index.append(selected_indexes[b_index][random_pos])
                old_prob.append(selected_prob[b_index][random_pos])

            baseline_rewards = [0.0 for _ in range(tile_batch_size)]
            for b_index in range(self.batch_size):
                tmp_rewards = []
                for t_index in range(tile_num):
                    a_index = t_index * self.batch_size + b_index
                    if random_pos < int(seq_len[a_index]):
                        tmp_rewards.append(batch_rewards[a_index][random_pos])
                    else:
                        tmp_rewards.append(0.0)
                tmp_rewards = tmp_rewards - np.mean(tmp_rewards)
                for t_index in range(tile_num):
                    baseline_rewards[t_index * self.batch_size + b_index] = tmp_rewards[t_index]

            _rand_features['selected_vec_mask'] = [str(random_pos) for _ in range(tile_batch_size)]
            _rand_features['action_index'] = action_index
            _rand_features['reward'] = baseline_rewards
            _rand_features['old_prob'] = old_prob

            self.train_batch(session, _rand_features)

            for b_index in range(tile_batch_size):
                rank_mask[b_index][selected_indexes[b_index][random_pos]] = 0.0

    # 最优reinforce_with sampled baseline
    def train_sampled_baseline(self, session):
        _features = session.run(self.features)

        # 采样生成序列

        tile_num = 4
        tile_batch_size = self.batch_size * tile_num
        _tile_features = {}
        _tile_features['features'] = np.tile(_features.get('features'), (tile_num, 1, 1))
        _tile_features['features_mask'] = np.tile(_features.get('features_mask'), tile_num)

        selected_indexes, selected_prob = self.step_by_model(session, _tile_features, tile_batch_size,
                                                             self.weight_sample)
        ctr_list, pbr_list = self.estimator_by_env(session, _tile_features, selected_indexes, tile_batch_size)

        seq_len = int(_features.get('features_mask')[0])

        batch_keeps = []
        for b_index in range(tile_batch_size):
            keep = 1.0
            keeps = []
            for index in range(seq_len):
                keeps.append(keep)
                keep *= (1.0 - pbr_list[b_index][index])
            batch_keeps.append(keeps)

        batch_rewards = []
        clicks = _features.get('clicks')
        for b_index in range(tile_batch_size):
            reward = 0.0
            rewards = []
            for index in range(seq_len - 1, -1, -1):
                reward = reward * (1.0 - pbr_list[b_index][index])
                reward += ctr_list[b_index][index]
                rewards.append(reward)

            rewards = rewards[::-1]
            batch_rewards.append(rewards)

        _rand_features = {}
        tile_features = _tile_features['features']
        _rand_features['features'] = tile_features
        _rand_features['features_mask'] = _tile_features['features_mask']
        selected_vec = [[] for _ in range(tile_batch_size)]
        for b_index in range(tile_batch_size):
            for r_index in range(self.rank_size):
                s_index = selected_indexes[b_index][r_index]
                selected_vec[b_index].append(tile_features[b_index][s_index])
        _rand_features['selected_vec'] = selected_vec

        rank_mask = np.ones((tile_batch_size, self.rank_size), dtype=float)
        for b_index in range(tile_batch_size):
            for index in range(seq_len, self.rank_size):
                rank_mask[b_index][index] = 0.5
        _rand_features['rank_mask'] = rank_mask

        for random_pos in range(seq_len):
            rewards = []
            action_index = []
            old_prob = []
            for b_index in range(tile_batch_size):
                rewards.append(batch_rewards[b_index][random_pos])
                action_index.append(selected_indexes[b_index][random_pos])
                old_prob.append(selected_prob[b_index][random_pos])
            baseline_rewards = (rewards - np.mean(rewards))
            _rand_features['selected_vec_mask'] = [str(random_pos) for _ in range(tile_batch_size)]
            _rand_features['action_index'] = action_index
            _rand_features['reward'] = baseline_rewards
            _rand_features['old_prob'] = old_prob

            self.train_batch(session, _rand_features)

            for b_index in range(tile_batch_size):
                rank_mask[b_index][selected_indexes[b_index][random_pos]] = 0.0

    # train_reinforce
    def train_reinforce(self, session):
        session.run(self.env_for_train.los)

        _features = session.run(self.features)

        # 采样生成序列

        seq_len = _features.get('features_mask')

        selected_indexes, selected_prob = self.step_by_model(session, _features, self.batch_size,
                                                             self.weight_sample)
        ctr_list, pbr_list = self.estimator_by_env(session, _features, selected_indexes, self.batch_size)

        batch_rewards = []
        for b_index in range(self.batch_size):
            reward = 0.0
            rewards = []
            for index in range(int(seq_len[b_index]) - 1, -1, -1):
                reward *= (1.0 - pbr_list[b_index][index])
                reward += ctr_list[b_index][index]
                rewards.append(reward)

            rewards = rewards[::-1]
            batch_rewards.append(rewards)

        _rand_features = {}
        features = _features['features']
        _rand_features['features'] = features
        _rand_features['features_mask'] = _features['features_mask']
        selected_vec = [[] for _ in range(self.batch_size)]
        for b_index in range(self.batch_size):
            for r_index in range(self.rank_size):
                s_index = selected_indexes[b_index][r_index]
                selected_vec[b_index].append(features[b_index][s_index])
        _rand_features['selected_vec'] = selected_vec

        rank_mask = np.ones((self.batch_size, self.rank_size), dtype=float)
        for b_index in range(self.batch_size):
            for index in range(int(seq_len[b_index]), self.rank_size):
                rank_mask[b_index][index] = 0.5
        _rand_features['rank_mask'] = rank_mask

        for random_pos in range(self.rank_size):

            rewards = []
            action_index = []
            old_prob = []
            for b_index in range(self.batch_size):
                action_index.append(selected_indexes[b_index][random_pos])
                if random_pos < int(seq_len[b_index]):
                    rewards.append(batch_rewards[b_index][random_pos])
                else:
                    rewards.append(0.0)
                old_prob.append(selected_prob[b_index][random_pos])

            # baseline_rewards = (rewards - np.mean(rewards)) / (np.var(rewards) + 1e-6)
            baseline_rewards = (rewards - np.mean(rewards))
            for b_index in range(self.batch_size):
                if random_pos >= int(seq_len[b_index]):
                    baseline_rewards[b_index] = 0.0

            baseline_rewards = (rewards - np.mean(rewards))
            _rand_features['selected_vec_mask'] = [str(random_pos) for _ in range(self.batch_size)]
            _rand_features['action_index'] = action_index
            _rand_features['reward'] = baseline_rewards
            _rand_features['old_prob'] = old_prob

            self.train_batch(session, _rand_features)

            for b_index in range(self.batch_size):
                rank_mask[b_index][selected_indexes[b_index][random_pos]] = 0.0

    # train_reinforce_whitening
    def train_reinforce_whitening(self, session):
        _features = session.run(self.features)

        # 采样生成序列

        seq_len = _features.get('features_mask')

        selected_indexes, selected_prob = self.step_by_model(session, _features, self.batch_size,
                                                             self.weight_sample)
        ctr_list, pbr_list = self.estimator_by_env(session, _features, selected_indexes, self.batch_size)

        batch_rewards = []
        for b_index in range(self.batch_size):
            reward = 0.0
            rewards = []
            for index in range(int(seq_len[b_index]) - 1, -1, -1):
                reward *= (1.0 - pbr_list[b_index][index])
                reward += ctr_list[b_index][index]
                rewards.append(reward)

            rewards = rewards[::-1]
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-6)
            batch_rewards.append(rewards)

        _rand_features = {}
        features = _features['features']
        _rand_features['features'] = features
        _rand_features['features_mask'] = _features['features_mask']
        selected_vec = [[] for _ in range(self.batch_size)]
        for b_index in range(self.batch_size):
            for r_index in range(self.rank_size):
                s_index = selected_indexes[b_index][r_index]
                selected_vec[b_index].append(features[b_index][s_index])
        _rand_features['selected_vec'] = selected_vec

        rank_mask = np.ones((self.batch_size, self.rank_size), dtype=float)
        for b_index in range(self.batch_size):
            for index in range(int(seq_len[b_index]), self.rank_size):
                rank_mask[b_index][index] = 0.5
        _rand_features['rank_mask'] = rank_mask

        for random_pos in range(self.rank_size):
            rewards = []
            action_index = []
            old_prob = []
            for b_index in range(self.batch_size):
                action_index.append(selected_indexes[b_index][random_pos])
                if random_pos < int(seq_len[b_index]):
                    rewards.append(batch_rewards[b_index][random_pos])
                else:
                    rewards.append(0.0)
                old_prob.append(selected_prob[b_index][random_pos])

            _rand_features['selected_vec_mask'] = [str(random_pos) for _ in range(self.batch_size)]
            _rand_features['action_index'] = action_index
            _rand_features['reward'] = rewards
            _rand_features['old_prob'] = old_prob

            self.train_batch(session, _rand_features)

            for b_index in range(self.batch_size):
                rank_mask[b_index][selected_indexes[b_index][random_pos]] = 0.0

    def train_ppo(self, session):
        _features, _ = session.run([self.features, tf.train.get_global_step()])

        # 采样生成序列

        tile_num = 512
        tile_batch_size = self.batch_size * tile_num
        _tile_features = {}
        _tile_features['features'] = np.tile(_features.get('features'), (tile_num, 1, 1))
        _tile_features['features_mask'] = np.tile(_features.get('features_mask'), tile_num)

        selected_indexes, selected_prob = self.step_by_model(session, _tile_features, tile_batch_size,
                                                             self.weight_sample)
        ctr_list, pbr_list = self.estimator_by_env(session, _tile_features, selected_indexes, tile_batch_size)

        seq_len = int(_features.get('features_mask')[0])

        batch_keeps = []
        for b_index in range(tile_batch_size):
            keep = 1.0
            keeps = []
            for index in range(seq_len):
                keeps.append(keep)
                keep *= (1.0 - pbr_list[b_index][index])
            batch_keeps.append(keeps)

        batch_rewards = []
        clicks = _features.get('clicks')
        for b_index in range(tile_batch_size):
            reward = 0.0
            rewards = []
            for index in range(seq_len - 1, -1, -1):
                reward = reward * (1.0 - pbr_list[b_index][index])
                reward += ctr_list[b_index][index]
                rewards.append(reward)

            rewards = rewards[::-1]
            batch_rewards.append(rewards)

        _rand_features = {}
        tile_features = _tile_features['features']
        _rand_features['features'] = tile_features
        _rand_features['features_mask'] = _tile_features['features_mask']
        selected_vec = [[] for _ in range(tile_batch_size)]
        for b_index in range(tile_batch_size):
            for r_index in range(self.rank_size):
                s_index = selected_indexes[b_index][r_index]
                selected_vec[b_index].append(tile_features[b_index][s_index])
        _rand_features['selected_vec'] = selected_vec

        rank_mask = np.ones((tile_batch_size, self.rank_size), dtype=float)
        for b_index in range(tile_batch_size):
            for index in range(seq_len, self.rank_size):
                rank_mask[b_index][index] = 0.5
        _rand_features['rank_mask'] = rank_mask

        for random_pos in range(seq_len):
            rewards = []
            action_index = []
            old_prob = []
            for b_index in range(tile_batch_size):
                rewards.append(batch_rewards[b_index][random_pos])
                action_index.append(selected_indexes[b_index][random_pos])
                old_prob.append(selected_prob[b_index][random_pos])
            # baseline_rewards = (rewards - np.mean(rewards)) / (np.var(rewards) + 1e-6)
            baseline_rewards = (rewards - np.mean(rewards))
            _rand_features['selected_vec_mask'] = [str(random_pos) for _ in range(tile_batch_size)]
            _rand_features['action_index'] = action_index
            _rand_features['reward'] = baseline_rewards
            _rand_features['old_prob'] = old_prob

            self.train_batch(session, _rand_features)

            for b_index in range(tile_batch_size):
                rank_mask[b_index][selected_indexes[b_index][random_pos]] = 0.0

    # train_static
    def train_static(self, session):
        _features = session.run(self.features)

        # 采样生成序列

        seq_len = _features.get('seq_len')

        selected_indexes = [[i for i in range(self.rank_size)] for i in range(self.batch_size)]
        # ctr_list, pbr_list = self.estimator_by_env(session, _tile_features, selected_indexes, tile_batch_size)

        batch_rewards = []
        clicks = _features.get('clicks')
        bounces = _features.get('bounces')
        for b_index in range(self.batch_size):
            reward = 0.0
            rewards = []
            for index in range(int(seq_len[b_index]) - 1, -1, -1):
                s_index = selected_indexes[b_index][index]
                reward *= 0.8
                if clicks[b_index][s_index] != '':
                    reward += float(clicks[b_index][s_index])
                if bounces[b_index][s_index] != '':
                    reward -= float(bounces[b_index][s_index])
                rewards.append(reward)

            rewards = rewards[::-1]
            batch_rewards.append(rewards)

        _rand_features = {}
        features = _features['features']
        _rand_features['features'] = features
        _rand_features['features_mask'] = _features['seq_len']
        selected_vec = [[] for _ in range(self.batch_size)]
        for b_index in range(self.batch_size):
            for r_index in range(self.rank_size):
                s_index = selected_indexes[b_index][r_index]
                selected_vec[b_index].append(features[b_index][s_index])
        _rand_features['selected_vec'] = selected_vec

        rank_mask = np.ones((self.batch_size, self.rank_size), dtype=float)
        for b_index in range(self.batch_size):
            for index in range(int(seq_len[b_index]), self.rank_size):
                rank_mask[b_index][index] = 0.5
        _rand_features['rank_mask'] = rank_mask

        for random_pos in range(self.rank_size):
            rewards = []
            action_index = []
            old_prob = []
            for b_index in range(self.batch_size):
                action_index.append(selected_indexes[b_index][random_pos])
                if random_pos < int(seq_len[b_index]):
                    rewards.append(batch_rewards[b_index][random_pos])
                else:
                    rewards.append(0.0)
            # baseline_rewards = (rewards - np.mean(rewards)) / (np.var(rewards) + 1e-6)
            baseline_rewards = (rewards - np.mean(rewards))
            for b_index in range(self.batch_size):
                if random_pos >= int(seq_len[b_index]):
                    baseline_rewards[b_index] = 0.0
            _rand_features['selected_vec_mask'] = [str(random_pos) for _ in range(self.batch_size)]
            _rand_features['action_index'] = action_index
            _rand_features['reward'] = baseline_rewards
            _rand_features['old_prob'] = old_prob

            self.train_batch(session, _rand_features)

            for b_index in range(self.batch_size):
                rank_mask[b_index][selected_indexes[b_index][random_pos]] = 0.0

    def train_new(self, session):
        _features = session.run(self.features)

        # 采样生成序列

        tile_num = 128
        tile_batch_size = self.batch_size * tile_num
        _tile_features = {}
        _tile_features['features'] = np.tile(_features.get('features'), (tile_num, 1, 1))
        _tile_features['features_mask'] = np.tile(_features.get('features_mask'), tile_num)

        selected_indexes, selected_prob = self.step_by_model(session, _tile_features, tile_batch_size,
                                                             self.weight_sample)
        ctr_list, pbr_list = self.estimator_by_env(session, _tile_features, selected_indexes, tile_batch_size)

        seq_len = int(_features.get('features_mask')[0])

        batch_rewards = []
        clicks = _features.get('clicks')
        for b_index in range(tile_batch_size):
            reward = 0.0
            rewards = []
            for index in range(seq_len - 1, -1, -1):
                reward = reward * (1.0 - pbr_list[b_index][index])
                reward += ctr_list[b_index][index]
                rewards.append(reward)

            rewards = rewards[::-1]
            batch_rewards.append(rewards)

        for r_index in range(seq_len):
            rewards = []
            for b_index in range(tile_batch_size):
                rewards.append(batch_rewards[b_index][r_index])
            rewards = (rewards - np.mean(rewards)) / (np.var(rewards) + 1e-6)
            for b_index in range(tile_batch_size):
                batch_rewards[b_index][r_index] = rewards[b_index]

        for b_index in range(tile_batch_size):
            one_features = _tile_features['features'][b_index]
            one_features_mask = _tile_features['features_mask'][b_index]
            one_selected_vec = []

            for r_index in range(self.rank_size):
                s_index = selected_indexes[b_index][r_index]
                one_selected_vec.append(one_features[s_index])

            for random_pos in range(seq_len):
                rank_mask = [1.0 for _ in range(self.rank_size)]
                for index in range(0, random_pos):
                    rank_mask[selected_indexes[b_index][index]] = 0.0
                for index in range(seq_len, self.rank_size):
                    rank_mask[index] = 0.5

                action_index = selected_indexes[b_index][random_pos]
                reward = batch_rewards[b_index][random_pos]
                prob = selected_prob[b_index][random_pos]
                self.pool[self.pool_index] = (
                    one_features, one_features_mask, one_selected_vec, str(random_pos), rank_mask, action_index, reward,
                    prob)
                self.pool_index = (self.pool_index + 1) % self.pool_size

        if self.pool[self.pool_index] is not None:
            for _ in range(100):
                rand_indexs = np.random.randint(self.pool_size, size=128)
                tmp_features = [[] for _ in range(8)]
                for index in rand_indexs.tolist():
                    for f_index in range(8):
                        tmp_features[f_index].append(self.pool[index][f_index])

                rand_features = {
                    'features': tmp_features[0],
                    'features_mask': tmp_features[1],
                    'selected_vec': tmp_features[2],
                    'selected_vec_mask': tmp_features[3],
                    'rank_mask': tmp_features[4],
                    'action_index': tmp_features[5],
                    'reward': tmp_features[6],
                    'old_prob': tmp_features[7],
                }

                self.train_batch(session, rand_features)

    def weight_sample(self, action_distribution):
        sample_index = []
        for o in action_distribution:
            r = np.random.uniform()
            find = False
            for i in range(len(o)):
                if o[i] > 1e-5:
                    r -= o[i]
                    if r <= 0:
                        sample_index.append(i)
                        find = True
                        break
            if not find:
                sample_index.append(len(o) - 1)
        return sample_index

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

    def eval(self, session):
        expections = []
        fb_seq_list = []
        for round in range(self.eval_batch_num):
            if len(self.eval_features_cache) <= round:
                _features = session.run(self.eval_features)
                self.eval_features_cache.append(_features)
            else:
                _features = self.eval_features_cache[round]

            # max生成序列
            selected_indexes, _ = self.step_by_model(session, _features, self.eval_batch_size, self.max_sample)

            # 送入评估器，得到点击率和跳失率
            ctr_list, pbr_list = self.estimator_by_env(session, _features, selected_indexes, self.eval_batch_size)

            rewards = cal_expection(ctr_list, pbr_list, _features.get('features_mask'))
            expections.extend(rewards)

            ratings = _features.get('ratings')
            feature_len = _features.get('features_mask')
            feature = _features.get('features')

            for b_index in range(self.eval_batch_size):
                size = min(self.rank_size, int(feature_len[b_index]))
                seq = []
                for r_index in range(size):
                    s_index = selected_indexes[b_index][r_index]
                    vec = feature[b_index][s_index]
                    rating = int(ratings[b_index][s_index])
                    seq.append(re_utils.Record('', rating, vec))

                fb_seq = re_utils.gen_scan_seq(seq)
                fb_seq_list.append(fb_seq)

        print 'expection is {}'.format(np.mean(expections))
        print 'map is {}'.format(cal_map(fb_seq_list))
        print 'click is {}'.format(cal_click_num(fb_seq_list))
        print 'view_deep is {}'.format(cal_view_deep(fb_seq_list))


class Predictor(Reinforce):
    def __init__(self, config, features, PolicyNetwork, SimulatorNetwork, simulator_checkpoint_path):
        self.features = features
        self.config = config
        self.batch_size = config.get('predict_batch_size')
        self.rank_size = config.get('rank_size')

        self.env_features = self.buildSimulationPlaceholder(config, self.rank_size)
        self.env = SimulatorNetwork(config, self.env_features, config.get('rank_size'), training=False)

        self.model_features = self.buildModelPlaceholder(config, self.rank_size)
        with tf.variable_scope("reinforce"):
            self.model = PolicyNetwork(config, self.model_features, config.get('rank_size'), training=True)

    def predict(self, session, writer):

        expections = []
        fb_seq_list = []

        while True:
            try:
                _features = session.run(self.features)
            except:
                break

            # max生成序列
            qid = _features.get('qid')
            feature_mask = _features.get('features_mask')
            batch_size = len(_features.get('features'))

            # rl
            selected_indexes, _ = self.step_by_model(session, _features, batch_size, self.max_sample)

            # simulator
            # selected_indexes = self.step_by_simulator(session, _features, batch_size)

            # context ctr
            # selected_indexes = [[i for i in range(self.rank_size)] for _ in range(batch_size)]
            # _ctr, _pbr = self.estimator_by_env(session, _features, selected_indexes, batch_size)
            # selected_indexes = []
            # for b_index in range(batch_size):
            #     seq = [(_ctr[b_index][i] if i < int(feature_mask[b_index]) else 0.0, i) for i in range(self.rank_size)]
            #     seq = sorted(seq, key=lambda x: x[0], reverse=True)
            #     selected_indexes.append([o[1] for o in seq])

            # shuffle
            # selected_indexes = []
            # for b_index in range(batch_size):
            #     tmp = [i for i in range(int(feature_mask[b_index]))]
            #     np.random.shuffle(tmp)
            #     for r_index in range(int(feature_mask[b_index]), self.rank_size):
            #         tmp.append(r_index)
            #     selected_indexes.append(tmp)

            ctr_list, pbr_list = self.estimator_by_env(session, _features, selected_indexes, batch_size)
            rewards = cal_expection(ctr_list, pbr_list, _features.get('features_mask'))
            expections.extend(rewards)

            ratings = _features.get('ratings')
            feature_len = _features.get('features_mask')
            feature = _features.get('features')

            for b_index in range(batch_size):
                size = min(self.rank_size, int(feature_len[b_index]))
                seq = []
                for r_index in range(size):
                    s_index = selected_indexes[b_index][r_index]
                    vec = feature[b_index][s_index]
                    rating = int(ratings[b_index][s_index])
                    seq.append(re_utils.Record('', rating, vec))

                fb_seq = re_utils.gen_scan_seq(seq)
                fb_seq_list.append(fb_seq)

        m_map = cal_map(fb_seq_list)
        m_clicks = cal_click_num(fb_seq_list)
        m_views = cal_view_deep(fb_seq_list)
        m_exception = np.mean(expections)
        write_record = [len(fb_seq_list), m_map, m_clicks, m_views, m_exception, '']
        writer.write(write_record, [i for i in range(len(write_record))])

        print 'expection is {}'.format(np.mean(expections))
        print 'map is {}'.format(m_map)
        print 'click_num is {}'.format(np.mean(m_clicks))
        print 'view_deep is {}'.format(np.mean(m_views))

    def step_by_simulator(self, session, _features, batch_size):
        rank_size = self.rank_size

        features_mask = _features.get('features_mask')
        features = _features.get('features')

        _feed_dict = {}

        zero_vec = [0.0 for _ in range(699)]
        list_features = [[zero_vec for _ in range(rank_size)] for _ in range(batch_size * rank_size)]

        _feed_dict[self.env_features.get('features')] = list_features
        _feed_dict[self.env_features.get('features_mask')] = np.tile(features_mask, rank_size)
        _feed_dict[self.env_features.get('seq_len')] = np.tile(_features.get('seq_len'), rank_size)

        selected_index = [[] for _ in range(batch_size)]
        selected = [[False for _ in range(rank_size)] for _ in range(batch_size)]
        for round in range(rank_size):
            # 更新待排
            for r_index in range(rank_size):
                for b_index in range(batch_size):
                    list_features[r_index * batch_size + b_index][round] = features[b_index][r_index]
            # run
            _ctr, _pbr = session.run([self.env.ctr_prediction, self.env.pbr_prediction], feed_dict=_feed_dict)

            # 选择
            for b_index in range(batch_size):
                max_score = 0.0
                max_index = 0
                for r_index in range(int(features_mask[b_index])):
                    if selected[b_index][r_index]:
                        continue
                    alpha = 0.6
                    score = alpha * _ctr[r_index * batch_size + b_index][round]
                    score += (1.0 - alpha) * (1.0 - _pbr[r_index * batch_size + b_index][round])
                    if score > max_score:
                        max_score = score
                        max_index = r_index

                if len(selected_index[b_index]) < int(features_mask[b_index]):
                    selected_index[b_index].append(max_index)
                    selected[b_index][max_index] = True
                else:
                    selected_index[b_index].append(len(selected_index[b_index]))

            # update
            for b_index in range(batch_size):
                for r_index in range(rank_size):
                    list_features[r_index * batch_size + b_index][round] = features[b_index][
                        selected_index[b_index][-1]]

        return selected_index

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
