import tensorflow as tf
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from projection import get_project_funtion
from decoder import dynamic_decoder

class model(object):
    def __init__(self,
                 embed,  # 词嵌入
                 vocabulary,
                 vocabulary_count,
                 num_layers,  # encoder和decoder的层数
                 num_units,  # encoder和decoder的隐藏状态维度
                 learning_rate,
                 max_gradient_norm,
                 max_len):

        self.post_string = tf.placeholder(dtype=tf.string, shape=(None, None), name="post_string")  # post字符串，batch_size*length
        self.response_string = tf.placeholder(dtype=tf.string, shape=(None, None), name="response_string")  # response字符串，batch_size*length
        self.label_string = tf.placeholder(dtype=tf.string, shape=(None, None), name="label_string")
        self.post_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="post_len")  # post长度
        self.response_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="reponse_len")  # response长度
        self.embed = tf.get_variable(dtype=tf.float32, initializer=embed, name="embed")  # 词嵌入，作为变量训练
        self.vocabulary = tf.constant(vocabulary, dtype=tf.string)  # 词汇表

        self.batch_size = tf.shape(self.post_string)[0]
        self.encoder_len = tf.shape(self.post_string)[1]
        self.decoder_len = tf.shape(self.response_string)[1]

        self.mask = tf.cumsum(tf.one_hot(self.response_len-1, self.decoder_len), axis=1, reverse=True)

        # 将字符转化成id表示的表
        self.string_to_id = MutableHashTable(key_dtype=tf.string,
                                             value_dtype=tf.int64,
                                             default_value=1,
                                             shared_name="string_to_id",
                                             name="string_to_id",
                                             checkpoint=True)
        # 将id转化成字符串表示的表
        self.id_to_string = MutableHashTable(key_dtype=tf.int64,
                                             value_dtype=tf.string,
                                             default_value="_NDW",
                                             shared_name="id_to_string",
                                             name="id_to_string",
                                             checkpoint=True)

        # 将post和response转化成id表示
        self.post_id = self.string_to_id.lookup(self.post_string)  # batch_size*length
        self.response_id = self.string_to_id.lookup(self.response_string)  # batch_size*length
        self.label_id = self.string_to_id.lookup(self.label_string)  #

        # 将post和response转化成嵌入表示
        self.post_embed = tf.nn.embedding_lookup(embed, self.post_id)  # batch_size*length*embed_size
        self.response_embed = tf.nn.embedding_lookup(embed, self.response_id)  # batch_size*length*embed_size

        # encoder和decoder的层数和维度
        encoder_cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        decoder_cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])

        projection_fn, loss_fn, inference_fn = get_project_funtion(vocabulary_count)

        with tf.variable_scope("encoder"):
            self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                              self.post_embed,
                                                              self.post_len,
                                                              dtype=tf.float32)
            # self.encoder_output_shape = tf.shape(self.encoder_output)  # [batch_size encoder_len num_units]
            # self.encoder_state_shape = tf.shape(self.encoder_state)  # [num_layers 2 batch_size num_units]

        with tf.variable_scope("decoder"):
            self.decoder_output, self.decoder_state, self.loop_state = dynamic_decoder(decoder_cell,
                                                                                       encoder_state=self.encoder_state,
                                                                                       input=self.response_embed,
                                                                                       response_len=self.response_len)
            # self.decoder_output_shape = tf.shape(self.decoder_output)  # [batch_size decoder_len num_units]
            # self.decoder_state_shape = tf.shape(self.decoder_state)  # [num_layers 2 batch_size num_units]
            # self.softmaxed_probability = projection_function(self.decoder_output)  # 词汇表softmaxed后的概率 [batch_size decoder_len vovabulary_count]
            # self.maximum_likelihood_id = tf.argmax(self.softmaxed_probability, axis=2)  # [batch_size decoder_len]
            # self.output_string = self.id_to_string.lookup(self.maximum_likelihood_id)
            self.loss, self.avg_loss = loss_fn(self.decoder_output, self.label_id, self.mask)


        with tf.variable_scope("decoder", reuse=True):
            self.inference_output, self.inference_state, self.inference_loop_state = dynamic_decoder(decoder_cell,
                                                                                                      encoder_state=self.encoder_state,
                                                                                                      projection_function=projection_fn,
                                                                                                      embed=self.embed,
                                                                                                      max_len=max_len)
            self.inference_maximum_likelihood_id = inference_fn(self.inference_output)
            self.inference_string = self.id_to_string.lookup(self.inference_maximum_likelihood_id)  # [batch_size decoder_len]

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.params = tf.global_variables()
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=3)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))



















