# -*- coding: utf-8 -*-
import tensorflow as tf
import json
import sys
import constant as C
import os
from utils import get_assignment_map_from_checkpoint
from utils import input_fn_builder, input_eval_fn_builder, input_predict_fn_builder
import utils
from hooks.duplicate_weight_hook import DuplicateWeightHook
from importlib import import_module

flags = tf.flags

FLAGS = flags.FLAGS

# flags for basic train setting
flags.DEFINE_string("task_type", None, "train,export")
flags.DEFINE_string("config", None, "indicate the config file path")
flags.DEFINE_string("record_file_root", None, "indicate the train tenorflow record file path")
flags.DEFINE_string("checkpoint_path", None, "indicate the checkpoint file path")
flags.DEFINE_string("init_checkpoint_path", None, "indicate the last checkpoint file path")
flags.DEFINE_string("pretrain_checkpoint_path", None, "indicate the pretrain checkpoint file path")
flags.DEFINE_string("eval_checkpoint_path", None, "indicate the eval checkpoint file path")
flags.DEFINE_integer("save_checkpoints_steps", 10000, "indicate the how many steps to save checkpoint ones")
flags.DEFINE_integer("batch_size", 32, "indicate the training batch size")
flags.DEFINE_integer("num_train_steps", 200000, "indicate the max train steps")
flags.DEFINE_integer("max_user_eval_steps", 50, "indicate the max user steps")
flags.DEFINE_integer("max_eval_steps", 1, "indicate the max evaluation steps")
flags.DEFINE_integer("stay_time_for_eval", 5, "if content stay time above this threshold treate as positive content")
flags.DEFINE_bool("is_clean_saved_model", "False", "indicate is to del the saved model path")

# flags for distribution
flags.DEFINE_string("train_table", None, "")
flags.DEFINE_string("eval_table", None, "")
flags.DEFINE_string("predict_table", None, "")
flags.DEFINE_string("result_table", None, "")
flags.DEFINE_integer("task_index", None, "Worker task index")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("worker_count", 1, "total work")
flags.DEFINE_string("output_path", None, "")

flags.mark_flag_as_required("config")
flags.mark_flag_as_required("task_type")
flags.mark_flag_as_required("checkpoint_path")

TF_VERSION = tf.__version__


def model_fn_builder(config, init_checkpoint_path):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for Estimator."""

        # policy_network_name = config["policy_network"]
        # policy_network_module = import_module('.'.join(['policy_network', policy_network_name]))
        policy_network_module = utils.load_policy_network_module(config)
        sampler_module = utils.load_sampler_module(config)
        if sampler_module is not None:
            sampler = sampler_module.Sampler(features)
            sampler.sample()

        # build train process
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_module = utils.load_trainer_module(config)

            policy_network_scope = config.get('policy_network_scope', '')
            with tf.variable_scope(policy_network_scope):
                loss = train_module.train_fn(config, features, policy_network_module.PGNetwork)

            # build restore params process
            if init_checkpoint_path is not None and tf.gfile.Exists(init_checkpoint_path):
                t_vars = tf.trainable_variables()
                (assignment_map, initialized_variable_names
                 ) = get_assignment_map_from_checkpoint(t_vars, init_checkpoint_path)

                tf.train.init_from_checkpoint(init_checkpoint_path, assignment_map)

                tf.logging.info("**** Trainable Variables ****")
                for var in t_vars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)

            if FLAGS.pretrain_checkpoint_path is not None and tf.gfile.Exists(FLAGS.pretrain_checkpoint_path):
                t_vars = tf.trainable_variables()
                filter_t_vars = []
                for val in t_vars:
                    if val.name.startswith('pretrain'):
                        filter_t_vars.append(val)
                (assignment_map, initialized_variable_names
                 ) = get_assignment_map_from_checkpoint(filter_t_vars, FLAGS.pretrain_checkpoint_path)

                tf.train.init_from_checkpoint(FLAGS.pretrain_checkpoint_path, assignment_map)

                tf.logging.info("**** Trainable Variables ****")
                for var in filter_t_vars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_PRETRAIN_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)

            global_step = tf.train.get_or_create_global_step()
            # setting learning rate
            learning_rate = config["init_learning_rate"]

            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                FLAGS.num_train_steps,
                end_learning_rate=1e-6,
                power=1.0,
                cycle=False)

            tf.summary.scalar('learning_rate', learning_rate)
            train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

            train_hooks = utils.load_train_hooks(config, features, policy_network_module.PGNetwork)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks
            )

            return output_spec

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_hooks = utils.load_eval_hooks(config, features, policy_network_module.PGNetwork)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=tf.constant(0),
                evaluation_hooks=eval_hooks
            )
        else:
            model = policy_network_module.PGNetwork(config, features, config.get('rank_size'), training=False,
                                                    batch_size=config.get(C.CONFIG_EVAL_BATCH_SIZE))
            predictions = {
                'prediction': model.prediction,
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def export_saved_model(config):
    if tf.gfile.Exists(FLAGS.output_path):
        tf.gfile.DeleteRecursively(FLAGS.output_path)

    features = {}
    input_infos = {}

    for column in config.get(C.CONFIG_COLUMNS):
        fea_side, fea_name = column.get(C.CONFIG_COLUMNS_NAME).split(':')
        fea_type = column.get(C.CONFIG_COLUMNS_TYPE)

        if fea_side in ['user', 'item', 'context', 'list']:
            if fea_type == C.CONFIG_COLUMNS_TYPE_SINGLE:
                features[fea_name] = tf.placeholder(tf.string, shape=[None], name=fea_name)
                input_infos[fea_name] = tf.saved_model.utils.build_tensor_info(features.get(fea_name))
            elif fea_type == C.CONFIG_COLUMNS_TYPE_SEQ:
                seq_len = column.get('seq_len')
                features[fea_name] = tf.placeholder(tf.string, shape=[None, seq_len], name=fea_name)
                input_infos[fea_name] = tf.saved_model.utils.build_tensor_info(features.get(fea_name))
                if column.get('need_mask'):
                    features[fea_name + "_mask"] = tf.placeholder(tf.string, shape=[None],
                                                                  name=fea_name + '_mask')
                    input_infos[fea_name + "_mask"] = tf.saved_model.utils.build_tensor_info(
                        features.get(fea_name + '_mask'))

    policy_network_fn = utils.load_policy_network_module(config)

    extra_input = policy_network_fn.extra_input(config)
    if extra_input is not None:
        for name, val in extra_input:
            features[name] = val
            input_infos[name] = tf.saved_model.utils.build_tensor_info(val)

    policy_network_scope = config.get('policy_network_scope', '')
    with tf.variable_scope(policy_network_scope):
        policy_network = policy_network_fn.PGNetwork(config, features, config.get('rank_size'), training=False)

    prediction = policy_network.prediction

    _outputs = {}

    for name, tensor in prediction.items():
        _outputs[name] = tf.saved_model.utils.build_tensor_info(tf.identity(tensor, name=name))

    signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=input_infos,
            outputs=_outputs,
            method_name="embedding")
    )

    version = 0
    path = FLAGS.output_path
    while tf.gfile.Exists(FLAGS.output_path):
        version += 1
        path = os.path.join(FLAGS.output_path, '_' + str(version))

    print('export path : {0}'.format(path))

    builder = tf.saved_model.builder.SavedModelBuilder(path)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        tf.logging.info("****** checkpoint path ******")
        tf.logging.info(FLAGS.checkpoint_path)
        saver = tf.train.Saver()
        # saver.save(sess, FLAGS.checkpoint_path)
        # exit()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"signature": signature})
    builder.save()

    with tf.gfile.GFile(path + '/done', mode='w') as fdone:
        fdone.write('')
        print("done")


def predict(config, worker_count, task_index):
    worker_device = "/job:localhost/replica:0/task:0/cpu:%d" % (0)
    print("worker_deivce = %s" % worker_device)

    # assign io related variables and ops to local worker device
    with tf.device(worker_device):
        input_fn = input_predict_fn_builder(
            table=FLAGS.predict_table,
            config=config
        )
        features = input_fn()

    # construct the model structure
    policy_network_module = utils.load_policy_network_module(config)

    predictor_module = utils.load_predictor_module(config)
    predictor = predictor_module.Predictor(config, features, policy_network_module.PGNetwork)

    policy_network_scope = config.get('policy_network_scope', '')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=policy_network_scope)
    saver = tf.train.Saver(var_list=g_vars)
    latest_ckpt_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    print 'latest_ckpt_path', latest_ckpt_path

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, latest_ckpt_path)

        writer = tf.python_io.TableWriter(FLAGS.result_table, slice_id=FLAGS.task_index)

        predictor.predict(sess, writer)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("Tensorflow Version: " + TF_VERSION)

    # load config file
    tf.logging.info("***** loding config *****")
    tf.logging.info(FLAGS.config)
    with open(FLAGS.config, 'r') as f:
        config = json.load(f)
        if FLAGS.config.split('/')[1] == 'biz':
            config[C.BIZ_NAME] = FLAGS.config.split('/')[2]

    if FLAGS.task_type == "train":
        sess_config = tf.ConfigProto(allow_soft_placement=True)

        run_config = tf.estimator.RunConfig(
            model_dir=FLAGS.checkpoint_path,
            save_checkpoints_steps=config.get('save_checkpoints_steps'),
            session_config=sess_config,
            log_step_count_steps=10)

        model_fn = model_fn_builder(config, FLAGS.init_checkpoint_path)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params={"batch_size": FLAGS.batch_size},
            config=run_config)

        if FLAGS.worker_count > 1:
            FLAGS.worker_count -= 1
        if FLAGS.task_index > 0:
            FLAGS.task_index -= 1

        train_input_fn = input_fn_builder(
            table=FLAGS.train_table,
            config=config
        )

        tf.logging.info("***** Running training *****")
        tf.logging.info("Batch size = %d", FLAGS.batch_size)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

        # do eval
        eval_input_fn = input_eval_fn_builder(
            table=FLAGS.eval_table,
            config=config)
        tf.logging.info("***** Running evaluation *****")
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=FLAGS.max_eval_steps, start_delay_secs=30,
                                          throttle_secs=30)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



    elif FLAGS.task_type == "export":
        export_saved_model(config)

    elif FLAGS.task_type == 'predict':
        predict(config, FLAGS.worker_count, FLAGS.task_index)


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
