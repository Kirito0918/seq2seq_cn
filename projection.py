import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers

def get_project_funtion(vocabulary_count):  # 返回一个能将隐藏状态映射函数
    def project_fn(input):
        output = layers.linear(input, vocabulary_count, scope="projection_layer")
        softmaxed_probability = tf.nn.softmax(output)  # batch_size*decoder_len*vocabulary_count
        return softmaxed_probability

    def loss_fn(decoder_output, label_id, mask):
        with tf.variable_scope("decoder_rnn"):
            softmaxed_probability = tf.nn.softmax(
                layers.linear(decoder_output, vocabulary_count, scope="projection_layer"))
            logits = tf.reshape(softmaxed_probability, [-1, vocabulary_count])  # [batch_size*decoder_len vovabulary_count]
            labels = tf.reshape(label_id, [-1])  # [batch_size*decoder_len]
            label_mask = tf.reshape(mask, [-1])  # [batch_size*decoder_len]
            local_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [batch_size*decoder_len]
            total_size = tf.reduce_sum(label_mask)
            total_size += 1e-12
            loss = tf.reduce_sum(local_loss*label_mask)
            avg_loss = loss / total_size
            return loss, avg_loss

    def inference_fn(inference_output):
        with tf.variable_scope("decoder_rnn"):
            inference_softmaxed_probability = tf.nn.softmax(
                layers.linear(inference_output, vocabulary_count, scope="projection_layer"))  # 词汇表softmaxed后的概率 [batch_size decoder_len vovabulary_count]
            inference_maximum_likelihood_id = tf.argmax(inference_softmaxed_probability, axis=2)
            return inference_maximum_likelihood_id

    return project_fn, loss_fn, inference_fn
