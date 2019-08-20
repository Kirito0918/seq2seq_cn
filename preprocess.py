from collections import defaultdict
from model_config import PAD_WORD, NOT_DEFINED_WORD, START_WORD, END_WORD
from model_config import DATASET_LENGTH, DEVSET_LENGTH, TESTSET_LENGTH
from model_config import VOCABULARY_COUNT, VOCABULARY_SIZE


#########################################  载入post和response数据集  ###################################################
total_lenth = DATASET_LENGTH + DEVSET_LENGTH + TESTSET_LENGTH
posts_string = []  # post列表
with open("./data/stc_weibo_train_post", "r", encoding="utf-8") as posts:
    line_count = 0
    for line in posts:
        line = line[: -1].split(" ")
        posts_string.append(line)
        line_count += 1
        if line_count % total_lenth == 0:
            print("posts读取完成，读取数量：", line_count)
            break

responses_string = []  # response列表
with open("./data/stc_weibo_train_response", "r", encoding="utf-8") as responses:
    line_count = 0
    for line in responses:
        line = line[: -1].split(" ")
        responses_string.append(line)
        line_count += 1
        if line_count % total_lenth == 0:
            print("responses读取完成，读取数量：", line_count)
            break
########################################################################################################################

##########################################  载入预训练的词向量  ########################################################
vocabulary = {}  # 载入腾讯的词汇表
with open("./data/Tencent_AILab_ChineseEmbedding.txt", "r", encoding="utf-8") as embeds:
    line_count = 0
    for line in embeds:
        if line_count == 0:
            line_count += 1
            continue
        line = line[: -1].split(" ")
        word = line[0]
        embed = line[1:]
        vocabulary[word] = embed
        line_count += 1
        if line_count % VOCABULARY_SIZE == 0:  # 载入词向量个数，8824330词，200维
            print("读取词向量完成，读取", line_count, "行")
            break
########################################################################################################################

#######################################  统计数据集中每个单词的词频，降序排序  #########################################
word_frequency = defaultdict(int)  # post和response用到的词汇和词频
for post in posts_string:
    for word in post:
        word_frequency[word] += 1
for response in responses_string:
    for word in response:
        word_frequency[word] += 1
word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

used_vocabulary = []  # post和response用到的词汇
for word, frequency in word_frequency:
    used_vocabulary.append(word)
used_vocabulary = [PAD_WORD] + [NOT_DEFINED_WORD] + [START_WORD] + [END_WORD] + used_vocabulary
print("posts和responses使用的词汇数量：", len(used_vocabulary))
########################################################################################################################

#######################################  截取需要个数的词汇表和词向量  #################################################
with open("./data/vocabulary.txt", "w", encoding='utf-8') as file:
    word_count = 0
    for word in used_vocabulary:
        file.writelines(word + "\n")
        word_count += 1
        if word_count % VOCABULARY_COUNT == 0:
            print("选取词汇表大小：", VOCABULARY_COUNT)
            break

with open("./data/embed.txt", "w", encoding='utf-8') as file:
    word_count = 0
    for word in used_vocabulary:
        if word in vocabulary.keys():
            file.writelines(" ".join(vocabulary[word]) + "\n")
            word_count += 1
        else:
            file.writelines(" ".join(['0'] * 200) + "\n")
            word_count += 1
        if word_count % VOCABULARY_COUNT == 0:
            break
########################################################################################################################

