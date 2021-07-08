"""
带ctc预测的标签生成
"""
import joblib

train_true_dict = {}
with open('LabelTrain.txt') as f:
    for line in f.readlines():
        img_name, label = line.strip().split('\t')
        train_true_dict[img_name] = label

data = joblib.load('../output/rec/ctc_pred.dump')
with open('LabelTrain_ctcpred.txt', 'w') as f_train_w:
    for img_path, value in data.items():
        img_name = img_path.split('/')[-1]
        prob = value[0][1]
        pred_label = value[0][0]
        word_images = []
        pred_results = value[1]
        pred_results = pred_results.tolist()
        if train_true_dict[img_name] == pred_label:
            content = img_name + '\t' + pred_label + '\t' + str(pred_results)
        else:
            content = img_name + '\t' + pred_label + '\t' + 'None'
        f_train_w.write(content)
        f_train_w.write('\n')
