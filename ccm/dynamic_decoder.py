# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Seq2seq layer operations for use in neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs

__all__ = ["dynamic_rnn_decoder"]

def dynamic_rnn_decoder(cell,  # 多层的 RNNCell
                        decoder_fn,  # 对每个时间步输出进行处理成输入的函数
                        inputs=None,  # 训练时，传入该参数，为 response 的嵌入向量 [batch_size, decoder_len, 600（300为词嵌入，100*3为3个实体嵌入）]
                        sequence_length=None,  # 训练时，传入该参数，为 response 的长度向量
                        parallel_iterations=None,  # 没用到这个参数
                        swap_memory=False,  # 没用到这个参数
                        time_major=False,  # 表示输入的数据集是否是 time-major 的，实验中为 False
                        scope=None,  # ="decoder_rnn"
                        name=None):  # 没用到这个参数
    """ seq2seq 模型的 RNN 动态解码器.

    dynamic_rnn_decoder 类似于 tf.python.ops.rnn.dynamic_rnn，因为解码器没有假设序列长度和输入的 batch size

    dynamic_rnn_decoder 有两种模式：训练和推导。并且，希望用户为每种模式创建分别的函数

    在训练和推导模式，cell 和 decoder_fn 都是被需要的。其中 cell 为每个时间步用的 RNN，
    decoder_fn 允许为 early stopping， state， next input， context 建模

    当训练时，要求用户提供 inputs。在每个时间步上，所提供 input 的一个切片被传给 decoder_fn，这修改并返回下个时间步的 input。

    sequence_length 在训练时为了展开而被需要，例如，当 input is not None。在测试时，当 input is None，sequence_length 就用不着了。

    在推导时，inputs 被期望为 None，并且 input 从 decoder_fn 中被单独的推导。

    Args:
        cell: RNNCell 的一个实例
        decoder_fn:
            一个需要 time， cell state， cell input，cell output 和 context state 的函数。
            他返回一个 early stopping 向量，cell state， next input， cell output 和 context state。
        inputs: 用于解码的输入，嵌入的形式
            If `time_major == False` (default), this must be a `Tensor` of shape:
                `[batch_size, max_time, ...]`.
            If `time_major == True`, this must be a `Tensor` of shape:
                `[max_time, batch_size, ...]`.
            The input to `cell` at each time step will be a `Tensor` with dimensions
                `[batch_size, ...]`.

        sequence_length:
            （可选） 一个 size 为 batch_size 的 int32/int64 向量。
            如果 inputs is not None 并且 sequence_length is None，
            它从 inputs 中被推导出来作为最大可能的序列长度
        parallel_iterations: (Default: 32).    平行运行中的迭代数量。
            这些操作没有任何的时间的依赖并且能够平行运行。
            这个参数为了空间折损了时间。
            值 >> 1 使用更多的内存但是花费更少的时间，
            然而较小的参数使用更少的内存但是计算的时间更久。
        swap_memory: 透明的交换前向传播产生的张量但是需要来自 GPU 到 CPU 的反向传播
            这允许训练可能不适用于单个 GPU 的 RNNs，只存在非常小的（或没有）性能损失。
        time_major: The shape format of the `inputs` and `outputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            使用 time_major = True 是更有效率的，因为它避免了开始和结束时 RNN 计算的转换
            但是大多数 TensorFlow 数据是 batch-major 的，所以这个函数默认接受和发出
            batch-major 形式的输入和输出。
        scope: VariableScope for the `raw_rnn`;
            defaults to None.
        name: NameScope for the decoder;
            defaults to "dynamic_rnn_decoder"

    Returns:
        一个元组 (outputs, final_state, final_context_state) 其中:

            outputs: RNN 输出张量
                If time_major == False (default), this will be a `Tensor` shaped:
                    `[batch_size, max_time, cell.output_size]`.
                If time_major == True, this will be a `Tensor` shaped:
                    `[max_time, batch_size, cell.output_size]`.
            final_state: The final state and will be shaped
                `[batch_size, cell.state_size]`.

            final_context_state:
                上下文状态通过 decoder_fn 的最终调用被返回。如果上下文状态在图运行之后
                保持保持间隔数据，这就是有用的。
                例如，一种使推导输出多样化的方法是使用一个随机的解码器函数，在这种情况下，
                我们想存储解码的输出，而不仅仅是 RNN 的输出。这能够通过在 context_state
                中维护一个 TensorArray 实现，并且存储每个迭代解码的输出。
    Raises:
        ValueError: if inputs is not None and has less than three dimensions.
    """
    with ops.name_scope(name, "dynamic_rnn_decoder",
                                            [cell, decoder_fn, inputs, sequence_length,
                                             parallel_iterations, swap_memory, time_major, scope]):
        if inputs is not None:
            # 将输入转化成张量
            inputs = ops.convert_to_tensor(inputs)
            # 测试输入的维度,不能小于 2
            if inputs.get_shape().ndims is not None and (
                    inputs.get_shape().ndims < 2):
                raise ValueError("Inputs must have at least two dimensions")

            # 如果不是 time_major，就要做一个转置
            if not time_major:
                # [batch, seq, features] -> [seq, batch, features]
                inputs = array_ops.transpose(inputs, perm=[1, 0, 2])  # decoder_len * batch_size * 600

            dtype = inputs.dtype
            input_depth = int(inputs.get_shape()[2])  # 600
            batch_depth = inputs.get_shape()[1].value  # batch_size
            max_time = inputs.get_shape()[0].value  # decoder_len
            if max_time is None:
                max_time = array_ops.shape(inputs)[0]

            # 将解码器的输入设置成一个 tensor 数组
            # 数组长度为 decoder_len，数组的每个元素是个 batch_size * 600 的张量
            inputs_ta = tensor_array_ops.TensorArray(dtype, size=max_time)
            inputs_ta = inputs_ta.unstack(inputs)

        def loop_fn(time, cell_output, cell_state, loop_state):
            """loop_fn 是一个函数，这个函数在 rnn 的相邻时间步之间被调用。
            　　
            函数的总体调用过程为：
            1. 初始时刻，先调用一次loop_fn，获取第一个时间步的cell的输入，loop_fn 中进行读取初始时刻的输入。
            2. 进行cell自环　(output, cell_state) = cell(next_input, state)
            3. 在 t 时刻 RNN 计算结束时，cell 有一组输出 cell_output 和状态 cell_state，都是 tensor；
            4. 到 t+1 时刻开始进行计算之前，loop_fn 被调用，调用的形式为
                loop_fn( t, cell_output, cell_state, loop_state)，而被期待的输出为：(finished, next_input, initial_state, emit_output, loop_state)；
            5. RNN 采用 loop_fn 返回的 next_input 作为输入，initial_state 作为状态，计算得到新的输出。
            在每次执行（output， cell_state） =  cell(next_input, state)后，执行 loop_fn() 进行数据的准备和处理。
            emit_structure 即上文的 emit_output 将会按照时间存入 emit_ta 中。
            loop_state  记录 rnn loop 的变量的状态。用作记录状态
            tf.where 是用来实现dynamic的。

            time: 第 time 个时间步之前的处理，起始为 0
            cell_output: 上一个时间步的输出
            cell_state: RNNCells 的长时记忆
            loop_state: 保存了上个时间步执行后是否已经结束，如果输出 alignments，还保存了存有 alignments 的 TensorArray
            return:
            """

            # 解码之前第一次调用
            if cell_state is None:
                # cell_state is None 时，cell_output 应该为 None
                if cell_output is not None:
                    raise ValueError("Expected cell_output to be None when cell_state "
                                                     "is None, but saw: %s" % cell_output)
                # cell_state is None 时，loop_state 应该为 None
                if loop_state is not None:
                    raise ValueError("Expected loop_state to be None when cell_state "
                                                     "is None, but saw: %s" % loop_state)
                context_state = None

            # 后续的调用
            else:
                if isinstance(loop_state, tuple):
                    (done, context_state) = loop_state
                else:
                    done = loop_state
                    context_state = None

            # 训练
            # 训练时 input is not None
            # 获得 next_cell_input
            if inputs is not None:
                # 第一个时间步之前的处理
                if cell_state is None:
                    next_cell_input = inputs_ta.read(0)  # 其实第一列都是 GO_ID

                # 之后的 cell 之间的处理
                else:

                    if batch_depth is not None:
                        batch_size = batch_depth
                    else:
                        batch_size = array_ops.shape(done)[0]  # done 是对循环是否结束的标注，

                    # 如果 time == max_time, 则 next_cell_input = batch_size * 600 的全 1 矩阵
                    # 否则，next_cell_input 从数据中读下一时间步的数据
                    next_cell_input = control_flow_ops.cond(
                            math_ops.equal(time, max_time),
                            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtype),
                            lambda: inputs_ta.read(time))

                # emit_output = attention
                (next_done, next_cell_state, next_cell_input, emit_output,
                 next_context_state) = decoder_fn(time, cell_state, next_cell_input, cell_output, context_state)
            # 推导
            else:
                # next_cell_input 通过 decoder_fn 获得
                (next_done, next_cell_state, next_cell_input, emit_output,
                 next_context_state) = decoder_fn(time, cell_state, None, cell_output,
                                                                                    context_state)
            # 检查是否已经结束
            if next_done is None:  # 当训练时，next_done 返回的是 None
                next_done = time >= sequence_length  # 当 time >= sequence_length 时，next_done = True

            # 构建 next_loop_state
            if next_context_state is None:  # 如果不输出 alignments
                next_loop_state = next_done
            else:
                next_loop_state = (next_done, next_context_state)

            return (next_done, next_cell_input, next_cell_state,
                            emit_output, next_loop_state)

        # Run raw_rnn function
        outputs_ta, final_state, final_loop_state = rnn.raw_rnn(
                cell, loop_fn, parallel_iterations=parallel_iterations,
                swap_memory=swap_memory, scope=scope)
        outputs = outputs_ta.stack()

        # 如果要输出 alignments，就获取 final_context_state
        if isinstance(final_loop_state, tuple):
            final_context_state = final_loop_state[1]
        else:
            final_context_state = None

        # 如果不是 time_major，就转置回去
        if not time_major:
            # [seq, batch, features] -> [batch, seq, features]
            outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
        return outputs, final_state, final_context_state
