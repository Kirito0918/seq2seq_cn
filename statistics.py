# 数据集大小4435959
# 词嵌入大小5731606，第一行的内容为空

# with open('./data/stc_weibo_train_post', 'r', encoding='utf8') as f:
#     count = 0
#     for idx, line in enumerate(f):
#         count += 1
#         line = line.strip()
#         if idx == 0:
#             print(line)
#             print(line.split())
#     print(count)  # 4435959
#
# with open('./data/stc_weibo_train_response', 'r', encoding='utf8') as f:
#     count = 0
#     for idx, line in enumerate(f):
#         count += 1
#         line = line.strip()
#         if idx == 0:
#             print(line)
#             print(line.split())
#     print(count)  # 4435959

with open('./data/Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf8') as f:
    count = 0
    for idx, line in enumerate(f):
        count += 1
        line = line.strip()
        if idx == 0 or idx == 1:
            print(idx)
            print(line)
    print(count)  # 5731607

