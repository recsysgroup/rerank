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
        rnn_hidden_size = 128
        hidden_size = 128

        go_emb = tf.get_variable(shape=[rnn_hidden_size],
                                 dtype=tf.float32,
                                 initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                              stddev=0.01),
                                 name="go_embedding")

        encoder_gru_fn = tf.keras.layers.GRU(rnn_hidden_size, return_state=True, return_sequences=True)

        cand_vec = features['features']

        cand_len = tf.string_to_number(features['features_mask'], out_type=tf.int32)
        cand_mask = tf.sequence_mask(cand_len,
                                     utils.seq_max_len(config, 'features'))

        encoder_vec, _ = encoder_gru_fn(cand_vec, mask=cand_mask)

        pos_embedding = tf.get_variable(shape=[_rank_size, hidden_size],
                                        dtype=tf.float32,
                                        initializer=tf.initializers.truncated_normal(mean=0.0,
                                                                                     stddev=0.01),
                                        trainable=True,
                                        name="pos_embedding")

        hidden_cand_vec = tf.layers.dense(cand_vec, hidden_size)
        hidden_cand_vec = hidden_cand_vec + pos_embedding

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        encoder_vec = modeling.transformer_model(
            input_tensor=hidden_cand_vec,
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=1,
            intermediate_size=hidden_size * 4,
            do_return_all_layers=False)

        self._point_vec = encoder_vec

        decoder_gru_fn = tf.keras.layers.GRU(rnn_hidden_size, return_state=True)

        list_vec = features['selected_vec']  # [B, seq_len, dense_dim]

        list_vec = tf.layers.dense(list_vec, hidden_size, activation=tf.nn.relu, name='dense1', reuse=tf.AUTO_REUSE)
        list_vec = tf.layers.dense(list_vec, hidden_size, activation=None, name='dense2', reuse=tf.AUTO_REUSE)

        seq_len = tf.string_to_number(features['selected_vec_mask'], out_type=tf.int32)
        seq_mask = tf.sequence_mask(seq_len,
                                    utils.seq_max_len(config, 'features'))

        selected_len = tf.string_to_number(features['selected_vec_mask'], out_type=tf.float32)
        not_first = tf.minimum(selected_len, 1.0)

        # outputs[B, seq_len, embedding_size]
        _, state = decoder_gru_fn(list_vec, mask=seq_mask)

        go_vec = tf.matmul(tf.expand_dims(1 - not_first, axis=1), tf.expand_dims(go_emb, axis=0))
        query_vec = go_vec + tf.expand_dims(not_first, axis=1) * state

        def output_fn(_query_vec, _point_vec):
            atten_vec = tf.expand_dims(_query_vec, axis=1) * _point_vec
            logits = tf.reduce_sum(atten_vec, axis=-1)

            rank_mask = features["rank_mask"]
            # mask logist
            neg_mask = rank_mask - tf.ones(shape=[1, _rank_size], dtype=tf.float32)
            neg_mask = neg_mask * 1000
            action_distribution = tf.nn.softmax(logits + neg_mask)

            return action_distribution

        self._action_distribution = output_fn(query_vec, encoder_vec)

        if 'point_vec' in features and 'last_state' in features and 'last_vec' in features:
            inc_seq_len = tf.minimum(seq_len, 1)
            inc_seq_mask = tf.sequence_mask(inc_seq_len, 1)
            last_vec = features['last_vec']
            last_vec = tf.layers.dense(last_vec, hidden_size, activation=tf.nn.relu, name='dense1', reuse=tf.AUTO_REUSE)
            last_vec = tf.layers.dense(last_vec, hidden_size, activation=None, name='dense2', reuse=tf.AUTO_REUSE)
            _, next_state = decoder_gru_fn(tf.expand_dims(last_vec, axis=1),
                                           initial_state=features['last_state'], mask=inc_seq_mask)
            inc_query_vec = go_vec + tf.expand_dims(not_first, axis=1) * next_state
            self._inc_action_distribution = output_fn(inc_query_vec, features['point_vec'])
            self._next_state = next_state

    @property
    def action_distribution(self):
        return self._action_distribution

    @property
    def inc_action_distribution(self):
        return self._inc_action_distribution

    @property
    def next_state(self):
        return self._next_state

    @property
    def point_vec(self):
        return self._point_vec

