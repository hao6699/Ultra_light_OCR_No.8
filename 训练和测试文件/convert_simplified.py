from zhconv import convert

# 训练集标签繁体转简体
with open('LabelTrain_simplified.txt', 'w') as f_w:
    with open('LabelTrain.txt') as f:
        for line in f.readlines():
            img_name, label = line.strip().split('\t')
            simplified_label = convert(label, 'zh-cn')
            content = img_name + '\t' + simplified_label
            f_w.write(content)
            f_w.write('\n')

# 验证集集标签繁体转简体
with open('test_label_simplified.txt', 'w') as f_w:
    with open('test_label.txt') as f:
        for line in f.readlines():
            img_name, label = line.strip().split('\t')
            simplified_label = convert(label, 'zh-cn')
            content = img_name + '\t' + simplified_label
            f_w.write(content)
            f_w.write('\n')