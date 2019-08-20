import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from dynamic_decoder import dynamic_rnn_decoder
from output_projection import output_projection_layer
from attention_decoder import * 
from tensorflow.contrib.session_bundle import exporter

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
NONE_ID = 0
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class Model(object):
    def __init__(self,
            num_symbols,  # 词汇表size
            num_embed_units,  # 词嵌入size
            num_units,  # RNN 每层单元数
            num_layers,  # RNN 层数
            embed,  # 词嵌入
            entity_embed=None,  #
            num_entities=0,  #
            num_trans_units=100,  #
            learning_rate=0.0001,
            learning_rate_decay_factor=0.95,  #
            max_gradient_norm=5.0,  #
            num_samples=500,  # 样本个数，sampled softmax
            max_length=60,
            mem_use=True,
            output_alignments=True,
            use_lstm=False):
        
        self.posts = tf.placeholder(tf.string, (None, None), 'enc_inps')  # batch_size * encoder_len
        self.posts_length = tf.placeholder(tf.int32, (None), 'enc_lens')  # batch_size
        self.responses = tf.placeholder(tf.string, (None, None), 'dec_inps')  # batch_size * decoder_len
        self.responses_length = tf.placeholder(tf.int32, (None), 'dec_lens')  # batch_size
        self.entities = tf.placeholder(tf.string, (None, None, None), 'entities')  # batch_size * triple_num * triple_len
        self.entity_masks = tf.placeholder(tf.string, (None, None), 'entity_masks')  # 没用到
        self.triples = tf.placeholder(tf.string, (None, None, None, 3), 'triples')  # batch_size * triple_num * triple_len * 3
        self.posts_triple = tf.placeholder(tf.int32, (None, None, 1), 'enc_triples')  # batch_size * encoder_len
        self.responses_triple = tf.placeholder(tf.string, (None, None, 3), 'dec_triples')  # batch_size * decoder_len * 3
        self.match_triples = tf.placeholder(tf.int32, (None, None, None), 'match_triples')  # batch_size * decoder_len * triple_num

        # 获得 encoder_batch_size ,编码器的 encoder_len
        encoder_batch_size, encoder_len = tf.unstack(tf.shape(self.posts))
        # 获得 triple_num
        # 每个 post 包含的知识图个数（补齐过的）
        triple_num = tf.shape(self.triples)[1]
        # 获得 triple_len
        # 每个知识图包含的关联实体个数（补齐过的）
        triple_len = tf.shape(self.triples)[2]

        # 使用的知识三元组
        one_hot_triples = tf.one_hot(self.match_triples, triple_len)  # batch_size * decoder_len * triple_num * triple_len
        # 用 1 标注了哪个时间步产生的回复用了知识三元组
        use_triples = tf.reduce_sum(one_hot_triples, axis=[2, 3])  # batch_size * decoder_len

        # 词汇映射到 index 的 hash table
        self.symbol2index = MutableHashTable(
                key_dtype=tf.string,  # key张量的类型
                value_dtype=tf.int64,  # value张量的类型
                default_value=UNK_ID,  # 缺少key的默认值
                shared_name="in_table",  # If non-empty, this table will be shared under the given name across multiple sessions
                name="in_table",  # 操作名
                checkpoint=True)  # if True, the contents of the table are saved to and restored from checkpoints. If shared_name is empty for a checkpointed table, it is shared using the table node name.

        # index 映射到词汇的 hash table
        self.index2symbol = MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value='_UNK',
                shared_name="out_table",
                name="out_table",
                checkpoint=True)

        # 实体映射到 index 的 hash table
        self.entity2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=NONE_ID,
                shared_name="entity_in_table",
                name="entity_in_table",
                checkpoint=True)

        # index 映射到实体的 hash table
        self.index2entity = MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value='_NONE',
                shared_name="entity_out_table",
                name="entity_out_table",
                checkpoint=True)

        # 将 post 的 string 映射成词汇 id
        self.posts_word_id = self.symbol2index.lookup(self.posts)  # batch_size * encoder_len
        # 将 post 的 string 映射成实体 id
        self.posts_entity_id = self.entity2index.lookup(self.posts)  # batch_size * encoder_len

        # 将 response 的 string 映射成词汇 id
        self.responses_target = self.symbol2index.lookup(self.responses)  # batch_size * decoder_len
        # 获得解码器的 batch_size，decoder_len
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        #  去掉 responses_target 的最后一列，给第一列加上 GO_ID
        self.responses_word_id = tf.concat([tf.ones([batch_size, 1], dtype=tf.int64)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)  # batch_size * decoder_len

        # 得到 response 的 mask
        # 首先将回复的长度 one_hot 编码
        # 然后横着从右向左累计求和，形成一个如果该位置在长度范围内，则为1，否则则为0的矩阵，最后一步 reshape 应该没有必要
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1, 
            decoder_len), reverse=True, axis=1), [-1, decoder_len])  # batch_size * decoder_len

        # 初始化 词嵌入 和 实体嵌入，传入了参数就直接赋值，没有的话就随机初始化
        if embed is None:
            self.embed = tf.get_variable('word_embed', [num_symbols, num_embed_units], tf.float32)
        else:
            self.embed = tf.get_variable('word_embed', dtype=tf.float32, initializer=embed)
        if entity_embed is None:
            self.entity_trans = tf.get_variable('entity_embed', [num_entities, num_trans_units], tf.float32, trainable=False)
        else:
            self.entity_trans = tf.get_variable('entity_embed', dtype=tf.float32, initializer=entity_embed, trainable=False)

        # 添加一个全连接层，输入是实体的嵌入，该层的 size=num_trans_units，激活函数是tanh
        # 为什么还要用全连接层连一下？？？？？？
        self.entity_trans_transformed = tf.layers.dense(self.entity_trans, num_trans_units, activation=tf.tanh, name='trans_transformation')
        # 7 * num_trans_units 的全零初始化的数组
        padding_entity = tf.get_variable('entity_padding_embed', [7, num_trans_units], dtype=tf.float32, initializer=tf.zeros_initializer())

        # 把 padding_entity 添加到 entity_trans_transformed 的最前，补了有什么用？？？？？？？？？？？？？
        self.entity_embed = tf.concat([padding_entity, self.entity_trans_transformed], axis=0)

        # tf.nn.embedding_lookup 以后维度会+1，所以通过reshape来取消这个多出来的维度
        triples_embedding = tf.reshape(tf.nn.embedding_lookup(self.entity_embed, self.entity2index.lookup(self.triples)), [encoder_batch_size, triple_num, -1, 3 * num_trans_units])
        entities_word_embedding = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.entities)), [encoder_batch_size, -1, num_embed_units])  # [batch_size,triple_num*triple_len,num_embed_units]

        # 把 head，relation，tail分割开来
        head, relation, tail = tf.split(triples_embedding, [num_trans_units] * 3, axis=3)

        # 静态图注意力机制
        with tf.variable_scope('graph_attention'):
            # 将头和尾连接起来
            head_tail = tf.concat([head, tail], axis=3)  # batch_size * triple_num * triple_len * 200

            # tanh(dot(W, head_tail))
            head_tail_transformed = tf.layers.dense(head_tail, num_trans_units, activation=tf.tanh, name='head_tail_transform')  # batch_size * triple_num * triple_len * 100

            # dot(W, relation)
            relation_transformed = tf.layers.dense(relation, num_trans_units, name='relation_transform')  # batch_size * triple_num * triple_len * 100

            # 两个向量先元素乘，再求和，等于两个向量的内积
            # dot(traspose(dot(W, relation)), tanh(dot(W, head_tail)))
            e_weight = tf.reduce_sum(relation_transformed * head_tail_transformed, axis=3)  # batch_size * triple_num * triple_len

            # 图中每个三元组的 alpha 权值
            alpha_weight = tf.nn.softmax(e_weight)  # batch_size * triple_num * triple_len

            # tf.expand_dims 使 alpha_weight 维度+1 batch_size * triple_num * triple_len * 1
            # 对第2个维度求和,由此产生每个图 100 维的图向量表示
            graph_embed = tf.reduce_sum(tf.expand_dims(alpha_weight, 3) * head_tail, axis=2)  # batch_size * triple_num * 100

        """
        [0, 1, 2... encoder_batch_size] 转化成 encoder_batch_size * 1 * 1 的矩阵 [[[0]], [[1]], [[2]],...]
        tf.tile 将矩阵的第 1 维进行扩展 encoder_batch_size * encoder_len * 1 [[[0],[0]...]],...]
        与 posts_triple 在第 2 维度上进行拼接，形成 indices 矩阵
        indices 矩阵：
        [
         [[0 0], [0 0], [0 0], [0 0], [0 1], [0 0], [0 2], [0 0],...encoder_len],
         [[1 0], [1 0], [1 0], [1 0], [1 1], [1 0], [1 2], [1 0],...encoder_len],
         [[2 0], [2 0], [2 0], [2 0], [2 1], [2 0], [2 2], [2 0],...encoder_len]
         ,...batch_size
        ]
        tf.gather_nd 将 graph_embed 中根据上面矩阵提供的索引检索图向量，再回填至 indices 矩阵
        encoder_batch_size * encoder_len * 100
        """
        graph_embed_input = tf.gather_nd(graph_embed, tf.concat([tf.tile(tf.reshape(tf.range(encoder_batch_size, dtype=tf.int32), [-1, 1, 1]), [1, encoder_len, 1]), self.posts_triple], axis=2))

        # 将 responses_triple 转化成实体嵌入 batch_size * decoder_len * 300
        triple_embed_input = tf.reshape(tf.nn.embedding_lookup(self.entity_embed, self.entity2index.lookup(self.responses_triple)), [batch_size, decoder_len, 3 * num_trans_units])

        # 将 posts_word_id 转化成词嵌入
        post_word_input = tf.nn.embedding_lookup(self.embed, self.posts_word_id)  # batch_size * encoder_len * 300

        # 将 responses_word_id 转化成词嵌入
        response_word_input = tf.nn.embedding_lookup(self.embed, self.responses_word_id)  # batch_size * decoder_len * 300

        # post_word_input, graph_embed_input 在第二个维度上拼接
        self.encoder_input = tf.concat([post_word_input, graph_embed_input], axis=2)  # batch_size * encoder_len * 400
        # response_word_input, triple_embed_input 在第二个维度上拼接
        self.decoder_input = tf.concat([response_word_input, triple_embed_input], axis=2)  # batch_size * decoder_len * 600

        # 构造 deep RNN
        encoder_cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        decoder_cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        
        # rnn encoder
        encoder_output, encoder_state = dynamic_rnn(encoder_cell,   self.encoder_input,
                self.posts_length, dtype=tf.float32, scope="encoder")

        # 由于词汇表维度过大，所以输出的维度不可能和词汇表一样。通过 projection 函数，可以实现从低维向高维的映射
        # 返回：输出函数，选择器函数，计算序列损失，采样序列损失，总体损失的函数
        output_fn, selector_fn, sequence_loss, sampled_sequence_loss, total_loss = output_projection_layer(num_units, 
                num_symbols, num_samples)

        
        # 用于训练的 decoder
        with tf.variable_scope('decoder'):
            # 得到注意力函数
            # 准备注意力
            # attention_keys_init: 注意力的 keys
            # attention_values_init: 注意力的 values
            # attention_score_fn_init: 计算注意力上下文的函数
            # attention_construct_fn_init: 计算所有上下文拼接的函数
            attention_keys_init, attention_values_init, attention_score_fn_init, attention_construct_fn_init \
                    = prepare_attention(encoder_output, 'bahdanau', num_units, imem=(graph_embed, triples_embedding), output_alignments=output_alignments and mem_use)#'luong', num_units)

            # 返回训练时解码器每一个时间步对输入的处理函数
            decoder_fn_train = attention_decoder_fn_train(
                    encoder_state, attention_keys_init, attention_values_init,
                    attention_score_fn_init, attention_construct_fn_init, output_alignments=output_alignments and mem_use, max_length=tf.reduce_max(self.responses_length))

            # 输出，最终状态，alignments 的 TensorArray
            self.decoder_output, _, alignments_ta = dynamic_rnn_decoder(decoder_cell, decoder_fn_train,
                    self.decoder_input, self.responses_length, scope="decoder_rnn")


            if output_alignments:

                self.decoder_loss, self.ppx_loss, self.sentence_ppx = total_loss(self.decoder_output, self.responses_target, self.decoder_mask, self.alignments, triples_embedding, use_triples, one_hot_triples)
                self.sentence_ppx = tf.identity(self.sentence_ppx, name='ppx_loss')  # 将 sentence_ppx 转化成一步操作
            else:
                self.decoder_loss = sequence_loss(self.decoder_output, 
                        self.responses_target, self.decoder_mask)

        # 用于推导的 decoder
        with tf.variable_scope('decoder', reuse=True):
            # 得到注意力函数
            attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                    = prepare_attention(encoder_output, 'bahdanau', num_units, reuse=True, imem=(graph_embed, triples_embedding), output_alignments=output_alignments and mem_use)#'luong', num_units)
            decoder_fn_inference = attention_decoder_fn_inference(
                    output_fn, encoder_state, attention_keys, attention_values, 
                    attention_score_fn, attention_construct_fn, self.embed, GO_ID, 
                    EOS_ID, max_length, num_symbols, imem=(entities_word_embedding, tf.reshape(triples_embedding, [encoder_batch_size, -1, 3*num_trans_units])), selector_fn=selector_fn)
            # imem: ([batch_size,triple_num*triple_len,num_embed_units],[encoder_batch_size, triple_num*triple_len, 3*num_trans_units]) 实体次嵌入和三元组嵌入的元组
                
            self.decoder_distribution, _, output_ids_ta = dynamic_rnn_decoder(decoder_cell,
                    decoder_fn_inference, scope="decoder_rnn")

            output_len = tf.shape(self.decoder_distribution)[1]  # decoder_len
            output_ids = tf.transpose(output_ids_ta.gather(tf.range(output_len)))  # [batch_size, decoder_len]

            # 对 output 的值域行裁剪
            word_ids = tf.cast(tf.clip_by_value(output_ids, 0, num_symbols), tf.int64)  # [batch_size, decoder_len]

            # 计算的是采用的实体词在 entities 的位置
            # 1、tf.shape(entities_word_embedding)[1] = triple_num*triple_len
            # 2、tf.range(encoder_batch_size): [batch_size]
            # 3、tf.reshape(tf.range(encoder_batch_size) * tf.shape(entities_word_embedding)[1], [-1, 1]): [batch_size, 1] 实体词在 entities 中的偏移量
            # 4、tf.clip_by_value(-output_ids, 0, num_symbols): [batch_size, decoder_len] 实体词的相对位置
            # 5、entity_ids: [batch_size * decoder_len] 加上偏移量之后在 entities 中的实际位置
            entity_ids = tf.reshape(tf.clip_by_value(-output_ids, 0, num_symbols) + tf.reshape(tf.range(encoder_batch_size) * tf.shape(entities_word_embedding)[1], [-1, 1]), [-1])

            # 计算的是所用的实体词
            # 1、entities: [batch_size, triple_num, triple_len]
            # 2、tf.reshape(self.entities, [-1]): [batch_size * triple_num * triple_len]
            # 3、tf.gather: [batch_size*decoder_len]
            # 4、entities: [batch_size, output_len]
            entities = tf.reshape(tf.gather(tf.reshape(self.entities, [-1]), entity_ids), [-1, output_len])

            words = self.index2symbol.lookup(word_ids)  # 将 id 转化为实际的词
            # output_ids > 0 为 bool 张量，True 的位置用 words 中该位置的词替换
            self.generation = tf.where(output_ids > 0, words, entities)
            self.generation = tf.identity(self.generation, name='generation')

        # 初始化训练过程
        self.learning_rate = tf.Variable(float(learning_rate), 
                trainable=False, dtype=tf.float32)

        # ？？？
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)

        # 更新参数的次数
        self.global_step = tf.Variable(0, trainable=False)

        # 要训练的参数
        self.params = tf.global_variables()

        # 选择优化算法
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.lr = opt._lr

        # 根据 decoder_loss 计算 params 梯度
        gradients = tf.gradients(self.decoder_loss, self.params)
        # 梯度裁剪
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        tf.summary.scalar('decoder_loss', self.decoder_loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)

        self.merged_summary_op = tf.summary.merge_all()
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        self.saver_epoch = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000, pad_step_number=True)

    # 打印参数
    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    #
    def step_decoder(self, session, data, forward_only=False, summary=False):
        input_feed = {self.posts: data['posts'],
                self.posts_length: data['posts_length'],
                self.responses: data['responses'],
                self.responses_length: data['responses_length'],
                self.triples: data['triples'],
                self.posts_triple: data['posts_triple'],
                self.responses_triple: data['responses_triple'],
                self.match_triples: data['match_triples']}

        if forward_only:
            output_feed = [self.sentence_ppx]
        else:
            output_feed = [self.sentence_ppx, self.gradient_norm, self.update]

        if summary:
            output_feed.append(self.merged_summary_op)

        return session.run(output_feed, input_feed)
