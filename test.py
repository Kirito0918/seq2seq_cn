import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# a = ["0"] * 100
# print(a)
#
# with open("./data/vocabulary", "w") as file:
#     file.writelines("a\n")
#     file.writelines("b\n")

# list = [0] * 200
# print([str(item)for item in list])
#
# a = "a   a"
# print(a.split(" "))

# print(2**32)

# ValueError: Shape must be rank 0 but is rank 1 for 'cond/Switch' (op: 'Switch') with input shapes: [4], [4].必须是标量
# time = tf.constant([7, 7, 7, 7], dtype=tf.int32)
# max_time = tf.constant([5, 7, 10, 12], dtype=tf.int32)
# result = tf.cond(tf.equal(time, max_time),
#                  lambda: tf.zeros([4, 10], dtype=tf.int32),
#                  lambda: tf.ones([4, 10], dtype=tf.int32))
# with tf.Session() as sess:
#     print(result.eval())

# time = tf.constant(7, dtype=tf.int32)
# max_time = tf.constant([5, 8, 10, 12], dtype=tf.int32)
# result = time >= max_time
# with tf.Session() as sess:
#     print(result.eval())

# probability = tf.constant([0.1, 0.3, 0.4, 0.2], dtype=tf.float32)
# argmax = tf.argmax(probability, axis=0)
# with tf.Session() as sess:
#     print(argmax.eval())

reference = [["我", "叫", "包笑明"]]
candidate = ["我", "叫", "包笑明"]
smooth = SmoothingFunction()
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
print(score)