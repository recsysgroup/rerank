import tensorflow as tf
import utils
from bert import modeling


def extra_input():
    return [
        ('last_state', tf.placeholder(tf.float32, shape=[None, 64 + 17], name='last_state')),
        ('point_vec', tf.placeholder(tf.float32, shape=[None, 1, 64 + 17], name='point_vec'))
    ]


class PGNetwork:
    def __init__(self, config, features, _rank_size, trainable=True, scope="train", batch_size=None, training=True):
        hidden_size = 128

        list_vec = features['features']  # [B, seq_len, dense_dim]
        reshape_list_vec = tf.reshape(list_vec, [-1, 699])

        seq_len = tf.string_to_number(features['seq_len'], out_type=tf.int32)
        seq_mask = tf.sequence_mask(seq_len,
                                    utils.seq_max_len(config, 'features'))
        attention_mask = modeling.create_attention_mask_from_input_mask(
            list_vec, seq_mask)

        def build_net(_reshape_list_vec, need_pos=True):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                pos_embedding = tf.get_variable(shape=[_rank_size, hidden_size],
                                            dtype=tf.float32,
                                            initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                                         stddev=0.01),
                                            trainable=True,
                                            name="pos_embedding"
                                            )

                hidden_reshape_list_vec = tf.layers.dense(reshape_list_vec, hidden_size, name='dense')

            hidden_list_vec = tf.reshape(hidden_reshape_list_vec, [-1, 20, hidden_size])
            if need_pos:
                hidden_list_vec = hidden_list_vec + pos_embedding

            # Run the stacked transformer.
            # `sequence_output` shape = [batch_size, seq_length, hidden_size].
            outputs = modeling.transformer_model(
                input_tensor=hidden_list_vec,
                hidden_size=hidden_size,
                num_hidden_layers=2,
                num_attention_heads=1,
                intermediate_size=hidden_size * 4,
                do_return_all_layers=False)

            outputs = tf.squeeze(tf.layers.dense(outputs, 1, activation=None), axis=-1)

            return outputs

        with tf.variable_scope('ctr'):
            ctr_outputs = build_net(reshape_list_vec, need_pos=False)

        with tf.variable_scope('pbr'):
            pbr_outputs = build_net(reshape_list_vec, need_pos=True)

        self._ctr_prediction = tf.nn.sigmoid(ctr_outputs)
        self._pbr_prediction = tf.nn.sigmoid(pbr_outputs)

        if training:
            with tf.name_scope('auc'):
                self._auc = {}

                if 'clicks' in features:
                    label = tf.string_to_number(features['clicks'], out_type=tf.float32)
                    bounce = tf.string_to_number(features['bounces'], out_type=tf.float32)
                    seq_len = tf.string_to_number(features['seq_len'], out_type=tf.int32)
                    weights = tf.sequence_mask(seq_len,
                                               utils.seq_max_len(config, 'features'), dtype=tf.int32)
                    self._auc['ctr'] = tf.metrics.auc(label, self._ctr_prediction, weights=weights)
                    self._auc['pbr'] = tf.metrics.auc(bounce, self._pbr_prediction, weights=weights)

            with tf.name_scope('loss'):
                clicks = tf.string_to_number(features['clicks'], out_type=tf.float32)
                bounce = tf.string_to_number(features['bounces'], out_type=tf.float32)
                seq_len = tf.string_to_number(features['seq_len'], out_type=tf.int32)
                weights = tf.sequence_mask(seq_len,
                                           utils.seq_max_len(config, 'features'), dtype=tf.int32)
                ctr_loss = tf.losses.log_loss(clicks, self._ctr_prediction, weights)
                pbr_loss = tf.losses.log_loss(bounce, self._pbr_prediction, weights)
                self._loss = ctr_loss + pbr_loss

    @property
    def ctr_prediction(self):
        return self._ctr_prediction

    @property
    def pbr_prediction(self):
        return self._pbr_prediction

    @property
    def auc(self):
        return self._auc

    @property
    def loss(self):
        return self._loss
