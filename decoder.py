import tensorflow as tf
from model_config import START_WORD_ID, END_WORD_ID
"""
动态RNN复写函数模板 
def loop_fn(time,  # 初始为0，第time个时间步之前的处理，标量
            cell_output,  # 上一个时间步的输出
            cell_state,  # RNN长时记忆
            loop_state):  # 存放一些循环信息，例如是否已完成或者注意力系数
    return (next_done,  # 是否已完成，boolean
            next_cell_input,  # 下个时间步的输入，一般是输入加上注意力的拼接
            next_cell_state,  # RNN长时记忆，在第一个时间步时初始化，之后都不用去管它
            emit_output,  # 模型的输出
            next_loop_state)  # 存放一些循环的信息
"""

def dynamic_decoder(cell,  # RNNCell
                    encoder_state,  # encoder最后输出状态[num_layers 2 batch_size num_units]
                    input=None,  # 训练时，给这个赋response [batch_size decoder_len embedding_size]
                    response_len=None,  # 回复的长度列表 [batch_size]
                    projection_function=None,  # 将decoder输出映射到词汇表的函数
                    embed=None,  # [vovabulary_count embedding_size]
                    max_len=None):  # 推导时的最大长度

    with tf.name_scope("dynamic_decoder"):
        if input is not None:  # 训练时
            dtype = input.dtype  # tf.float32
            decoder_len = tf.shape(input)[1]  # decoder_len
            input = tf.transpose(input, perm=[1, 0, 2])  # [decoder_len batch_size embedding_size]

            input_tensorarray = tf.TensorArray(dtype=dtype, size=decoder_len, clear_after_read=False)
            input_tensorarray = input_tensorarray.unstack(input)  # decoder_len*[batch_size embedding_size]

######### 动态RNN复写函数 ##############################################################################################
        def loop_fn(time,  # 初始为0，第time个时间步之前的处理
                    cell_output,  # 上一个时间步的输出
                    cell_state,  # RNN长时记忆
                    loop_state):  # 存放一些循环信息，例如是否已完成或者注意力系数
            # 在训练的模式下 ###########################################################################################
            if input is not None:
                if cell_state is None:  # 第0个时间步之前的处理
                    emit_output = None  # 第0个时间步之前是没有输出的
                    cell_state = encoder_state  # encoder赋值encoder的最后状态
                    next_cell_state = cell_state  # 将cell状态一直传递下去
                    next_cell_input = input_tensorarray.read(0)  # 读取第0个时间步的输入
                    next_loop_state = loop_state  # 将循环状态信息一直传递下去，如果有必要可以从里面存取一些信息
                    next_done = time >= response_len  # 如果是第response_len个时间步之前的处理，说明已经解码完成了
                    # 这里可以再加入一些初始化信息
                else:  # 之后的时间步的处理
                    emit_output = cell_output  # 这里的输出并没有做任何加工
                    next_cell_state = cell_state  # 将cell状态一直传递下去
                    next_cell_input = tf.cond(
                        tf.equal(time, decoder_len),
                        lambda: tf.zeros_like(input_tensorarray.read(0), dtype=dtype),
                        lambda: input_tensorarray.read(time),
                    )
                    next_done = time >= response_len  # 如果是第response_len个时间步之前的处理，说明已经解码完成了
                    next_loop_state = loop_state  # 将循环状态信息一直传递下去，如果有必要可以从里面存取一些信息
            # 在推导的模式下 ###########################################################################################
            else:
                if cell_state is None:  # 第0个时间步之前的处理
                    emit_output = None  # 第0个时间步之前是没有输出的
                    cell_state = encoder_state  # encoder赋值encoder的最后状态
                    next_cell_state = cell_state  # 将cell状态一直传递下去
                    batch_size = tf.shape(cell_state)[2]
                    next_cell_input_id = tf.ones([batch_size], dtype=tf.int32) * START_WORD_ID  # 第一步的输入为起始词 batch_size
                    next_cell_input = tf.gather(embed, next_cell_input_id)  # 第一步的输入 [batch_size embedding_size]
                    next_loop_state = loop_state  # 将循环状态信息一直传递下去，如果有必要可以从里面存取一些信息
                    next_done = tf.zeros([batch_size], dtype=tf.bool)  # 如果是第response_len个时间步之前的处理，说明已经解码完成了
                    # 这里可以再加入一些初始化信息
                else:  # 之后的时间步的处理
                    emit_output = cell_output  # 这里的输出并没有做任何加工
                    next_cell_state = cell_state  # 将cell状态一直传递下去
                    batch_size = tf.shape(cell_state)[2]
                    softmaxed_probability = projection_function(emit_output)  # 词汇表softmaxed后的概率 [batch_size vovabulary_count]
                    maximum_likelihood_id = tf.argmax(softmaxed_probability, axis=1)  # [batch_size]
                    done = tf.equal(maximum_likelihood_id, END_WORD_ID)
                    next_cell_input = tf.gather(embed, maximum_likelihood_id)  # [batch_size, embedding_size]
                    next_done = tf.cond(
                        tf.equal(time, max_len),
                        lambda: tf.ones([batch_size], dtype=tf.bool),
                        lambda: done,
                    )
                    next_loop_state = loop_state

            return (next_done, next_cell_input, next_cell_state, emit_output, next_loop_state)
########################################################################################################################

        output_tensorarray, final_state, final_loop_state = tf.nn.raw_rnn(cell, loop_fn, scope="decoder_rnn")

        output = output_tensorarray.stack()  # [decoder_len batch_size num_units]
        output = tf.transpose(output, perm=[1, 0, 2])  # [batch_size decoder_len num_units]

        return output, final_state, final_loop_state