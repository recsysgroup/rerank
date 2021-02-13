# coding: utf-8
import numpy as np
import collections
import tensorflow as tf
import constant as C
import re
from importlib import import_module
from sklearn.metrics import roc_auc_score


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)


def remove_control_chars(s):
    control_chars = ''.join(map(unichr, range(0, 32) + range(127, 160)))
    control_char_re = re.compile('[%s]' % re.escape(control_chars))
    return control_char_re.sub('', s)


def format_data(word):
    word = remove_control_chars(word)
    if word == "" or word == "null" or word == "NULL" or word == "\\N":
        return "unk"
    else:
        return word


def parse_seq_str_op(text, max_seq_len):
    def user_func(text):
        vals = []
        parts = [i for i in text.split(',')]

        for i in range(min(len(parts), max_seq_len)):
            vals.append(parts[i])

        mask = str(len(vals))
        vals = vals + [''] * (max_seq_len - len(vals))

        return vals, mask

    y = tf.py_func(user_func, [text], [tf.string, tf.string])
    y[0].set_shape((max_seq_len))
    y[1].set_shape(())
    return y


def parse_seq_dense_str_op(text, max_seq_len, dim):
    def user_func(text):
        vals = []
        parts = [i for i in text.split(',')]

        for i in range(min(len(parts), max_seq_len)):
            feas = [float(o) for o in parts[i].split(':')]
            vals.append(feas)

        mask = str(len(vals))
        for i in range(max_seq_len - len(vals)):
            vals.append([0 for i in range(dim)])

        return np.array(vals, dtype=np.float32), mask

    y = tf.py_func(user_func, [text], [tf.float32, tf.string])
    y[0].set_shape((max_seq_len, dim))
    y[1].set_shape(())
    return y


def text_dataset_and_decode_builder(table, config, mode='train'):
    columns = []

    for column in config.get(C.CONFIG_COLUMNS):
        exclude = column.get('exclude')
        if exclude and mode in exclude:
            continue
        col_side, col_name = column.get(C.CONFIG_COLUMNS_NAME).split(':')
        col_type = column.get(C.CONFIG_COLUMNS_TYPE)
        columns.append((col_name, col_type, column))

    record_defaults = []
    selected_cols = []
    colname_2_index = {}

    for column in columns:
        record_defaults.append('')
        selected_cols.append(column[0])
        colname_2_index[column[0]] = len(colname_2_index)

    def _decode_record(text):

        line = tf.decode_csv(text, [[""] for _ in range(len(columns))], field_delim=';')

        ret_dict = {}

        for name, type, col in columns:
            if type == C.CONFIG_COLUMNS_TYPE_SINGLE:
                ret_dict[name] = line[colname_2_index.get(name)]
            elif type == C.CONFIG_COLUMNS_TYPE_SEQ:
                _value, _mask = parse_seq_str_op(line[colname_2_index.get(name)], col.get('seq_len'))
                ret_dict[name] = _value
                if col.get('need_mask'):
                    ret_dict[name + '_mask'] = _mask
            elif type == C.CONFIG_COLUMNS_TYPE_SEQ_DENSE:
                _value, _mask = parse_seq_dense_str_op(line[colname_2_index.get(name)], col.get('seq_len'),
                                                       col.get('dim'))
                ret_dict[name] = _value
                if col.get('need_mask'):
                    ret_dict[name + '_mask'] = _mask

        return ret_dict

    tf.logging.info("read table [{0}], record_defaults is {1}, selected_cols is {2}".format(table, str(record_defaults),
                                                                                            str(selected_cols)))
    d = tf.data.TextLineDataset([table])
    return d, _decode_record


def input_fn_builder(table, config):
    def input_fn():
        d, _decode_record = text_dataset_and_decode_builder(table, config)
        d = d.repeat()
        d = d.shuffle(buffer_size=config.get(C.CONFIG_TRAIN_BATCH_SIZE) * 100)
        d = d.map(_decode_record, num_parallel_calls=C.NUM_PARALLEL_CALLS)
        d = d.batch(batch_size=config.get(C.CONFIG_TRAIN_BATCH_SIZE))
        d = d.prefetch(buffer_size=config.get(C.CONFIG_TRAIN_BATCH_SIZE))
        return d

    return input_fn


def input_eval_fn_builder(table, config):
    def input_fn():
        d, _decode_record = text_dataset_and_decode_builder(table, config, mode='eval')
        d = d.repeat()
        d = d.map(_decode_record, num_parallel_calls=C.NUM_PARALLEL_CALLS)
        d = d.batch(batch_size=config.get(C.CONFIG_EVAL_BATCH_SIZE))
        d = d.prefetch(buffer_size=config.get(C.CONFIG_EVAL_BATCH_SIZE))
        return d

    return input_fn


def input_predict_fn_builder(table, config):
    def input_fn():
        d, _decode_record = text_dataset_and_decode_builder(table, config, mode='predict')
        d = d.repeat(1)
        d = d.map(_decode_record, num_parallel_calls=C.NUM_PARALLEL_CALLS)
        d = d.batch(batch_size=config.get(C.CONFIG_PREDICT_BATCH_SIZE))
        d = d.prefetch(buffer_size=config.get(C.CONFIG_PREDICT_BATCH_SIZE))

        iterator = d.make_one_shot_iterator()
        one_element = iterator.get_next()
        return one_element

    return input_fn


def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    with tf.variable_scope(name_or_scope='', reuse=True):
        alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        beta = tf.get_variable('beta' + name, _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                               dtype=tf.float32)
    input_shape = list(_x.get_shape())

    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False, name=name, reuse=tf.AUTO_REUSE)
    # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_p = tf.sigmoid(beta * x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def seq_max_len(config, name):
    columns = config['columns']
    for col in columns:
        col_name = col['name'].split(':')[1]
        if col_name == name and col['type'] in ('seq', 'seq_dense'):
            return col['seq_len']
    raise RuntimeError('{} not found in config'.format(name))


def switch_state(features, name, sample_index, selected_id):
    _ids_len = int(features.get(name + '_mask')[sample_index])
    features.get(name)[sample_index][_ids_len] = selected_id
    features.get(name + '_mask')[sample_index] = str(_ids_len + 1)


def load_policy_network_module(config):
    policy_network_name = config["policy_network"]

    if config.get(C.BIZ_NAME) is not None:
        try:
            biz_name = config.get(C.BIZ_NAME)
            path = '.'.join(['biz', biz_name, policy_network_name])
            tf.logging.info('register policy_network from {}'.format(path))
            return import_module(path)
        except ImportError:
            pass

    path = '.'.join(['policy_network', policy_network_name])
    tf.logging.info('register policy_network from {}'.format(path))
    return import_module(path)


def load_simulator_network_module(config):
    policy_network_name = config["simulator_network"]

    if config.get(C.BIZ_NAME) is not None:
        try:
            biz_name = config.get(C.BIZ_NAME)
            path = '.'.join(['biz', biz_name, policy_network_name])
            tf.logging.info('register policy_network from {}'.format(path))
            return import_module(path)
        except ImportError:
            pass

    path = '.'.join(['policy_network', policy_network_name])
    tf.logging.info('register simulator_network from {}'.format(path))
    return import_module(path)


def load_hooks(config, features, Network, prefix):
    eval_hooks = []
    if config.get(prefix):
        for hook in config[prefix].split(","):
            tf.logging.info("register eval hook: %s" % (hook))
            file_name, class_name = hook.split(".")
            mod = None
            if config.get(C.BIZ_NAME) is not None:
                try:
                    mod = __import__(".".join(["biz", config.get(C.BIZ_NAME), file_name]), fromlist=[class_name])
                except ImportError:
                    pass

            if mod is None:
                mod = __import__(".".join(["hooks", file_name]), fromlist=[class_name])

            hook_class = getattr(mod, class_name)
            policy_network_scope = config.get('policy_network_scope', '')
            with tf.variable_scope(policy_network_scope):
                eval_hooks.append(hook_class(config, features, Network))

    return eval_hooks


def load_trainer_module(config):
    train_name = config["trainer"]
    if config.get(C.BIZ_NAME) is not None:
        try:
            biz_name = config.get(C.BIZ_NAME)
            path = '.'.join(['biz', biz_name, train_name])
            tf.logging.info('register trainer from {}'.format(path))
            return import_module(path)
        except ImportError:
            pass

    tf.logging.info('register predictor from {}'.format('.'.join(['trainers', train_name])))
    return import_module('.'.join(['trainers', train_name]))


def load_evaluator_module(config):
    eval_name = config["evaluator"]
    if config.get(C.BIZ_NAME) is not None:
        try:
            biz_name = config.get(C.BIZ_NAME)
            path = '.'.join(['biz', biz_name, eval_name])
            tf.logging.info('register trainer from {}'.format(path))
            return import_module(path)
        except ImportError:
            pass

    tf.logging.info('register predictor from {}'.format('.'.join(['evals', eval_name])))
    return import_module('.'.join(['evals', eval_name]))


def load_predictor_module(config):
    predictor_name = config["predictor"]
    if config.get(C.BIZ_NAME) is not None:
        try:
            biz_name = config.get(C.BIZ_NAME)
            path = '.'.join(['biz', biz_name, predictor_name])
            tf.logging.info('register predictor from {}'.format(path))
            return import_module(path)
        except ImportError:
            pass

    tf.logging.info('register predictor from {}'.format('.'.join(['predictors', predictor_name])))
    return import_module('.'.join(['predictors', predictor_name]))


def load_eval_hooks(config, features, Network):
    return load_hooks(config, features, Network, 'eval_hooks')


def load_train_hooks(config, features, Network):
    return load_hooks(config, features, Network, 'train_hooks')


def load_sampler_module(config):
    if "sampler" not in config:
        return None
    sampler_name = config["sampler"]
    if config.get(C.BIZ_NAME) is not None:
        try:
            biz_name = config.get(C.BIZ_NAME)
            path = '.'.join(['biz', biz_name, sampler_name])
            tf.logging.info('register sampler from {}'.format(path))
            return import_module(path)
        except ImportError:
            pass

    tf.logging.info('register sampler from {}'.format('.'.join(['sampler', sampler_name])))
    return import_module('.'.join(['sampler', sampler_name]))


class GaucCaler:
    def __init__(self):
        self.rate_dict = {}
        self.label_dict = {}
        pass

    def add(self, key, rate, label):
        if key not in self.rate_dict:
            self.rate_dict[key] = [rate]
            self.label_dict[key] = [label]
        else:
            self.rate_dict[key].append(rate)
            self.label_dict[key].append(label)

    def gauc(self):
        sum_rate_auc = 0.0
        sum_rate = 0.0

        for key, label_list in self.label_dict.items():
            _len = len(label_list)
            _avg = (sum(label_list) + 0.0) / _len
            if _avg == 0 or _avg == 1:
                continue

            sum_rate += _len
            rate_list = self.rate_dict.get(key)
            gauc = roc_auc_score(label_list, rate_list)
            sum_rate_auc += gauc * _len

        return sum_rate_auc / (sum_rate + 1e-8)


if __name__ == '__main__':
    import json

    f = open("./config.json", 'r')
    config = json.load(f, encoding="utf-8")

    input_files = []
    input_files_path = "../data/train/*.tfrecord"
    tf.logging.info(input_files_path)
    for input_pattern in input_files_path.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))
    print
    "*** Input Files ***"
    for input_file in input_files:
        print
        "  %s" % input_file

    input_fn = input_fn_builder(input_files=input_files, config=config, batch_size=1)
    iter_fn = input_fn().make_one_shot_iterator()
    features = iter_fn.get_next()

    with tf.Session() as sess:
        print
        features
        result = sess.run(features)
        for i in result:
            print
            i, np.shape(result[i])
