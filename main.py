from model_config import DATASET_LENGTH, DEVSET_LENGTH, TESTSET_LENGTH
from model_config import PAD_WORD, START_WORD, END_WORD, BATCH_SIZE, VOCABULARY_COUNT
from model_config import NUM_LAYERS, NUM_UNITS, LEARNING_RATE, MAX_GRADIENT_NORM, MAX_LENGTH
from model_config import TRAIN
import numpy as np
from model import model
import tensorflow as tf
import copy
# from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

def read_dataset():  # 读取训练集
    post_string = []
    total_length = DATASET_LENGTH + DEVSET_LENGTH + TESTSET_LENGTH
    with open("./data/stc_weibo_train_post", "r", encoding="utf-8") as posts:
    # with open("./data/post_test", "r", encoding="utf-8") as posts:
        line_count = 0
        for line in posts:
            line = line[: -1].split(" ")
            post_string.append(line)
            line_count += 1
            if line_count % total_length == 0:
                print("posts读取完成，读取数量：", line_count)
                break
    # print("第200条post：", post_string[199])
    response_string = []
    with open("./data/stc_weibo_train_response", "r", encoding="utf-8") as responses:
        line_count = 0
        for line in responses:
            line = line[: -1].split(" ")
            response_string.append(line)
            line_count += 1
            if line_count % total_length == 0:
                print("responses读取完成，读取数量：", line_count)
                break
    # print("第200条response：", response_string[199])
    train_post_string = post_string[: DATASET_LENGTH]
    train_response_string = response_string[: DATASET_LENGTH]
    print("训练集大小：", len(train_post_string))
    dev_post_string = post_string[DATASET_LENGTH: DATASET_LENGTH + DEVSET_LENGTH]
    dev_response_string = response_string[DATASET_LENGTH: DATASET_LENGTH + DEVSET_LENGTH]
    print("验证集大小：", len(dev_post_string))
    test_post_string = post_string[DATASET_LENGTH + DEVSET_LENGTH:]
    test_response_string = response_string[DATASET_LENGTH + DEVSET_LENGTH:]
    print("测试集大小：", len(test_post_string))
    return train_post_string, train_response_string, dev_post_string, dev_response_string, test_post_string, test_response_string

def read_embed():  # 读取词嵌入
    embed = []
    with open("./data/embed.txt", "r", encoding="utf-8") as embeds:
        for line in embeds:
            line = line[: -1].split(" ")
            line = [float(item) for item in line]
            embed.append(line)
    for item in embed:
        if len(item) != 200:
            print("词向量维度错误")
    print("词向量数量：", len(embed))
    embed = np.array(embed, dtype=np.float32)
    # print("第100个词向量：", embed[99])
    return embed

def read_vocabulary():
    vocabulary = []
    with open("./data/vocabulary.txt", "r", encoding="utf-8") as vocabularies:
        for line in vocabularies:
            line = line[: -1]
            vocabulary.append(line)
    print("词汇表大小：", len(vocabulary))
    # print("第100个词汇：", vocabulary[99])
    return vocabulary

def get_batch_data(post, response):  # 得到一批数据
    post_len = [len(item) for item in post]
    response_len = [len(item)+1 for item in response]
    encoder_len = max(post_len)
    # print("post的最大长度：", encoder_len)
    decoder_len = max(response_len)
    # print("response的最大长度：", decoder_len)
    for index in range(len(post)):  # 补齐post的长度
        post[index] = post[index] + [PAD_WORD]*(encoder_len-len(post[index]))
    label = copy.deepcopy(response)
    for index in range(len(response)):  # 补齐response的长度
        response[index] = [START_WORD] + response[index] + [PAD_WORD]*(decoder_len-1-len(response[index]))
    for index in range(len(label)):  # 补齐label的长度
        label[index] = label[index] + [END_WORD] + [PAD_WORD]*(decoder_len-1-len(label[index]))
    post = np.array(post)
    response = np.array(response)
    post_len = np.array(post_len, dtype=np.int32)
    response_len = np.array(response_len, dtype=np.int32)
    batch_data = {"post": post,
                  "response": response,
                  "label": label,
                  "post_len": post_len,
                  "response_len": response_len}
    return batch_data

if __name__ == '__main__':
    train_post, train_response, dev_post, dev_response, test_post, test_response = read_dataset()  # 读取选定数据集长度的post和response，列表
    embed = read_embed()  # 读取词嵌入，np数组
    vocabulary = read_vocabulary()  # 读取词汇表，列表
    vocabulary_index = list(range(len(vocabulary)))
    num_train_set = len(train_post)
    num_dev_set = len(dev_post)
    num_test_set = len(test_post)
    smooth = SmoothingFunction()
    seq2seq = model(embed=embed,
                    vocabulary=vocabulary,
                    vocabulary_count=VOCABULARY_COUNT,
                    num_layers=NUM_LAYERS,
                    num_units=NUM_UNITS,
                    learning_rate=LEARNING_RATE,
                    max_gradient_norm=MAX_GRADIENT_NORM,
                    max_len=MAX_LENGTH)

    with tf.Session() as sess:
        if tf.train.get_checkpoint_state("./train/"):
            print("从记录中回复模型参数！")
            seq2seq.saver.restore(sess, tf.train.latest_checkpoint("./train/"))
            seq2seq.print_parameters()
        else:
            print("重新创建模型参数！")
            tf.global_variables_initializer().run()
            sess.run(seq2seq.string_to_id.insert(tf.constant(vocabulary, dtype=tf.string), tf.constant(vocabulary_index, dtype=tf.int64)))
            sess.run(seq2seq.id_to_string.insert(tf.constant(vocabulary_index, dtype=tf.int64), tf.constant(vocabulary, dtype=tf.string)))
        if TRAIN:
            while(True):
                start = 0
                end = BATCH_SIZE
                while(start < num_train_set):
                    if end > num_train_set:
                        end = num_train_set
                    data = get_batch_data(train_post[start: end], train_response[start: end])  # 获得一批格式化过的数据
                    feed_data = {seq2seq.post_string: data["post"],
                                 seq2seq.response_string: data["response"],
                                 seq2seq.label_string: data["label"],
                                 seq2seq.post_len: data["post_len"],
                                 seq2seq.response_len: data["response_len"]}
                    batch_size = end - start
                    _, loss, avg_loss = sess.run([seq2seq.update, seq2seq.loss, seq2seq.avg_loss], feed_dict=feed_data)
                    print("start=", start, "end=", end-1, "每条数据的平均损失：", loss/batch_size, "，每个单词平均损失：", avg_loss)
                    start = end
                    end += BATCH_SIZE
                print("结束一轮的训练，记录模型参数。")
                seq2seq.saver.save(sess, "./train/", global_step=seq2seq.global_step)
                total_loss = 0.0
                st = 0
                ed = BATCH_SIZE
                bleu_1_gram_scores = []
                while (st < num_dev_set):
                    if ed > num_dev_set:
                        ed = num_dev_set
                    dev_data = get_batch_data(dev_post[st: ed], dev_response[st: ed])
                    bs = ed - st
                    feed_data = {seq2seq.post_string: dev_data["post"],
                                 seq2seq.response_string: dev_data["response"],
                                 seq2seq.label_string: dev_data["label"],
                                 seq2seq.post_len: dev_data["post_len"],
                                 seq2seq.response_len: dev_data["response_len"]}
                    loss = sess.run(seq2seq.loss, feed_dict=feed_data)
                    total_loss += loss
                    inference_string = sess.run(seq2seq.inference_string, feed_dict=feed_data)
                    inference_string = [[str(word, encoding="utf-8") for word in response] for response in
                                        inference_string.tolist()]
                    inference_response = []
                    for response in inference_string:
                        try:
                            sentense = response[: response.index(END_WORD)]
                        except Exception as e:
                            sentense = response
                        inference_response.append(sentense)
                    for id in range(bs):
                        reference = [inference_response[id]]
                        candidate = dev_data["response"][id]
                        bleu_1_gram_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
                        bleu_1_gram_scores.append(bleu_1_gram_score)
                    st = ed
                    ed += BATCH_SIZE
                print("验证集上每条数据的平均损失：", total_loss/num_dev_set)
                print("验证集上平均每条数据的1-gram bleu分数：", sum(bleu_1_gram_scores)/len(bleu_1_gram_scores))
        else:
            total_loss = 0.0
            st = 0
            ed = BATCH_SIZE
            bleu_1_gram_scores = []
            generated_response = []
            while (st < num_test_set):
                if ed > num_test_set:
                    ed = num_test_set
                test_data = get_batch_data(test_post[st: ed], test_response[st: ed])
                bs = ed - st
                feed_data = {seq2seq.post_string: test_data["post"],
                             seq2seq.response_string: test_data["response"],
                             seq2seq.label_string: test_data["label"],
                             seq2seq.post_len: test_data["post_len"],
                             seq2seq.response_len: test_data["response_len"]}
                loss = sess.run(seq2seq.loss, feed_dict=feed_data)
                total_loss += loss
                inference_string = sess.run(seq2seq.inference_string, feed_dict=feed_data)
                inference_string = [[str(word, encoding="utf-8") for word in response] for response in
                                    inference_string.tolist()]
                inference_response = []
                for response in inference_string:
                    try:
                        sentense = response[: response.index(END_WORD)]
                    except Exception as e:
                        sentense = response
                    inference_response.append(sentense)
                    generated_response.append(sentense)
                for id in range(bs):
                    reference = [inference_response[id]]
                    candidate = test_data["response"][id]
                    bleu_1_gram_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0),
                                                      smoothing_function=smooth.method1)
                    bleu_1_gram_scores.append(bleu_1_gram_score)
                st = ed
                ed += BATCH_SIZE
            with open("./data/post_test", "w", encoding="utf-8") as file:
                for index in range(TESTSET_LENGTH):
                    file.write("post:" + " ".join(test_post[index]) +
                               "\nrespose:" + " ".join(test_response[index]) +
                               "\ngenerated_respose:" + " ".join(generated_response[index]) + "\n")
            print("测试集上每条数据的平均损失：", total_loss / num_dev_set)
            print("测试集上平均每条数据的1-gram bleu分数：", sum(bleu_1_gram_scores) / len(bleu_1_gram_scores))




