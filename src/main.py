import tensorflow as tf
import json
import constant as C
import os
from utils import get_assignment_map_from_checkpoint
from utils import input_fn_builder, input_eval_fn_builder, input_predict_fn_builder
from importlib import import_module
import time
from utils import get_assignment_map_from_checkpoint
import utils

flags = tf.flags

FLAGS = flags.FLAGS

# flags for basic train setting
flags.DEFINE_string("task_type", None, "train,export")
flags.DEFINE_string("config", None, "indicate the config file path")
flags.DEFINE_string("record_file_root", None, "indicate the train tenorflow record file path")
flags.DEFINE_string("checkpoint_path", None, "indicate the checkpoint file path")
flags.DEFINE_string("init_checkpoint_path", None, "indicate the last checkpoint file path")
flags.DEFINE_string("simulator_checkpoint_path", None, "indicate the simulator checkpoint file path")
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


def trian_and_eval_on_single_worker(config):
    train_input_fn = input_fn_builder(
        table=FLAGS.train_table,
        config=config
    )
    d = train_input_fn()
    iterator = d.make_one_shot_iterator()
    features = iterator.get_next()

    global_step = tf.train.get_or_create_global_step()
    # construct the model structure
    # loss, optimizer = model_fn(features, labels, global_step)
    policy_network_module = utils.load_policy_network_module(config)

    simulator_network_module = utils.load_simulator_network_module(config)

    trainer_module = utils.load_trainer_module(config)

    trainer = trainer_module.Trainer(config, features
                                     , policy_network_module.PGNetwork
                                     , simulator_network_module.PGNetwork
                                     , global_step
                                     , FLAGS.simulator_checkpoint_path
                                     )

    eval_graph = tf.Graph()
    with eval_graph.as_default() as g:
        eval_input_fn = input_eval_fn_builder(
            table=FLAGS.eval_table,
            config=config
        )
        eval_d = eval_input_fn()
        eval_iterator = eval_d.make_one_shot_iterator()
        eval_features = eval_iterator.get_next()
        eval_module = utils.load_evaluator_module(config)
        evalutor = eval_module.Evaluator(config, eval_features
                                                     , policy_network_module.PGNetwork
                                                     , simulator_network_module.PGNetwork
                                                     , FLAGS.simulator_checkpoint_path)
        eval_saver = tf.train.Saver(max_to_keep=10)

    if FLAGS.init_checkpoint_path is not None:
        t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reinforce')
        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(t_vars, FLAGS.init_checkpoint_path)

        tf.train.init_from_checkpoint(FLAGS.init_checkpoint_path, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in t_vars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

    hooks = []
    step = 0
    previous_ckpt_path = ''
    with tf.train.MonitoredTrainingSession(master=''
            , checkpoint_dir=FLAGS.checkpoint_path
            , save_checkpoint_secs=60
            , is_chief=True, hooks=hooks) as mon_sess:
        while True:
            # _, c, g = mon_sess.run([optimizer, loss, global_step])

            trainer.train(mon_sess)

            _global_step = mon_sess.run(global_step)

            print 'step:{}'.format(_global_step)

            # eval
            if _global_step > 0 and _global_step % 100 == 0:
                latest_ckpt_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
                if latest_ckpt_path is None or latest_ckpt_path == previous_ckpt_path:
                    continue
                print 'latest_ckpt_path', latest_ckpt_path
                with tf.Session(graph=eval_graph) as eval_sess:
                    eval_sess.run(tf.global_variables_initializer())
                    eval_sess.run(tf.local_variables_initializer())
                    eval_saver.restore(eval_sess, latest_ckpt_path)
                    evalutor.eval(eval_sess)
                    previous_ckpt_path = latest_ckpt_path

            if _global_step >= FLAGS.num_train_steps:
                break

    print("%d steps finished." % step)


def train_and_eval(config, cluster=None):
    # single worker
    if cluster == None:
        trian_and_eval_on_single_worker(config)
        pass
    else:
        worker_spec = FLAGS.worker_hosts.split(",")
        worker_count = len(worker_spec)

        is_chief = FLAGS.task_index == 0

        # construct the servers
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        # join the ps server
        if FLAGS.job_name == "ps":
            server.join()

        # start the training
        if FLAGS.task_index != 1:
            train(config, worker_count=worker_count, task_index=FLAGS.task_index, cluster=cluster,
                  is_chief=is_chief,
                  target=server.target)
        else:
            eval(config, worker_count=worker_count, task_index=FLAGS.task_index, cluster=cluster, is_chief=is_chief,
                 target=server.target)


def train(config, worker_count, task_index, cluster, is_chief, target):
    worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
    print("worker_deivce = %s" % worker_device)

    # assign io related variables and ops to local worker device
    with tf.device(worker_device):
        train_input_fn = input_fn_builder(
            table=FLAGS.train_table,
            config=config,
            slice_id=FLAGS.task_index,
            slice_count=worker_count
        )
        d = train_input_fn()
        iterator = d.make_one_shot_iterator()
        features = iterator.get_next()

    # assign global variables to ps nodes
    available_worker_device = "/job:worker/task:%d" % (task_index)
    with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # construct the model structure
        # loss, optimizer = model_fn(features, labels, global_step)
        policy_network_module = utils.load_policy_network_module(config)

        simulator_network_module = utils.load_simulator_network_module(config)

        trainer_module = utils.load_trainer_module(config)

        trainReinforce = trainer_module.TrainReinforce(config, features
                                                       , policy_network_module.PGNetwork
                                                       , simulator_network_module.PGNetwork
                                                       , global_step
                                                       , FLAGS.simulator_checkpoint_path
                                                       )

        if FLAGS.init_checkpoint_path is not None:
            t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='reinforce')
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(t_vars, FLAGS.init_checkpoint_path)

            tf.train.init_from_checkpoint(FLAGS.init_checkpoint_path, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in t_vars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

    # hooks = [tf.train.StopAtStepHook(last_step=FLAGS.num_train_steps)]
    hooks = []
    step = 0
    with tf.train.MonitoredTrainingSession(master=target
            , checkpoint_dir=FLAGS.checkpoint_path
            , save_checkpoint_secs=120
            , is_chief=is_chief, hooks=hooks) as mon_sess:
        while True:
            # _, c, g = mon_sess.run([optimizer, loss, global_step])

            trainReinforce.train(mon_sess)

            _global_step = mon_sess.run(global_step)

            if task_index == 0:
                print 'step:{}'.format(_global_step)

            if _global_step >= FLAGS.num_train_steps:
                break

    print("%d steps finished." % step)


def eval(config, worker_count, task_index, cluster, is_chief, target):
    worker_device = "/job:localhost/replica:0/task:0/cpu:%d" % (0)
    print("worker_deivce = %s" % worker_device)

    # assign io related variables and ops to local worker device
    with tf.device(worker_device):
        eval_input_fn = input_eval_fn_builder(
            table=FLAGS.eval_table,
            config=config
        )
        d = eval_input_fn()
        iterator = d.make_one_shot_iterator()
        features = iterator.get_next()

    # construct the model structure
    policy_network_module = utils.load_policy_network_module(config)

    simulator_network_module = utils.load_simulator_network_module(config)

    trainer_module = utils.load_trainer_module(config)

    evalReinforce = trainer_module.EvalReinforce(config, features
                                                 , policy_network_module.PGNetwork
                                                 , simulator_network_module.PGNetwork
                                                 , FLAGS.simulator_checkpoint_path)

    saver = tf.train.Saver(max_to_keep=10)

    previous_ckpt_path = ''

    for index in range(10000):
        time.sleep(60)
        latest_ckpt_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        print 'latest_ckpt_path', latest_ckpt_path
        if latest_ckpt_path is None or latest_ckpt_path == previous_ckpt_path:
            continue

        with tf.Session() as sess:
            saver.restore(sess, latest_ckpt_path)
            evalReinforce.eval(sess)
            previous_ckpt_path = latest_ckpt_path

            step = int(latest_ckpt_path.split('-')[-1])
            if step >= FLAGS.num_train_steps:
                break


def predict(config, worker_count, task_index):
    worker_device = "/job:localhost/replica:0/task:0/cpu:%d" % (0)
    print("worker_deivce = %s" % worker_device)

    # assign io related variables and ops to local worker device
    with tf.device(worker_device):
        input_fn = input_predict_fn_builder(
            table=FLAGS.predict_table,
            config=config,
            slice_count=worker_count,
            slice_id=task_index
        )
        features = input_fn()

        # construct the model structure
    policy_network_module = utils.load_policy_network_module(config)

    simulator_network_module = utils.load_simulator_network_module(config)

    # trainer_module = utils.load_trainer_module(config)
    predictor_module = utils.load_predictor_module(config)

    predictor = predictor_module.Predictor(config, features
                                           , policy_network_module.PGNetwork
                                           , simulator_network_module.PGNetwork
                                           , FLAGS.simulator_checkpoint_path)

    policy_network_scope = config.get('policy_network_scope', 'reinforce')
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=policy_network_scope)
    saver = tf.train.Saver(var_list=g_vars)
    latest_ckpt_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    print 'latest_ckpt_path', latest_ckpt_path

    with tf.Session() as sess:
        simulator_checkpoint_path = FLAGS.simulator_checkpoint_path
        if simulator_checkpoint_path is not None:
            t_vars = tf.trainable_variables()
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(t_vars, simulator_checkpoint_path)

            tf.train.init_from_checkpoint(simulator_checkpoint_path, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in t_vars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env'))
        sess.run(init_op)
        saver.restore(sess, latest_ckpt_path)

        writer = tf.python_io.TableWriter(FLAGS.result_table, slice_id=FLAGS.task_index)

        predictor.predict(sess, writer)


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

    policy_network_scope = config.get('policy_network_scope', 'reinforce')
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

    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    worker_count = len(worker_spec)

    if FLAGS.task_type == 'train':
        # cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
        train_and_eval(config)

    elif FLAGS.task_type == 'predict':
        predict(config, worker_count=worker_count, task_index=FLAGS.task_index)

    elif FLAGS.task_type == "export":
        export_saved_model(config)


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
