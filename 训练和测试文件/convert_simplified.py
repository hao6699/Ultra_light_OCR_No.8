from zhconv import convert


# 字典文件繁体转简体
with open('alphabet_simplified.txt', 'w') as f_w:
    simplified_alphabets = set()
    with open('alphabet.txt') as f:
        for line in f.readlines():
            alpha = line.strip()
            simplified_alpha = convert(alpha, 'zh-cn')
            simplified_alphabets.add(simplified_alpha)
    for simplified_alpha in sorted(list(simplified_alphabets)):
        f_w.write(simplified_alpha)
        f_w.write('\n')

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