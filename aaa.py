# 训练语句
# python tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_tinyresnet_train_v2.0.yml

# 模型转换语句
# python tools/export_model.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model=./output/rec_chinese_lite_v2.0/iter_epoch_60  Global.save_inference_dir=./inference/rec_inference/

# 模型推断语句1
# python tools/infer/predict_rec.py  --rec_model_dir=./inference/rec_inference/  --image_dir=/newjixieyingpan/百度轻量级文字识别/A榜测试数据集/A榜测试数据集/TestAImages

# 模型推断语句2
# python tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_tinyresnet_train_v2.0.yml -o Global.infer_img="/newjixieyingpan/百度轻量级文字识别/A榜测试数据集/A榜测试数据集/TestAImages" Global.pretrained_model="./output/rec_chinese_tinyresnet_v2.0/best_accuracy"

# with open('/newjixieyingpan/百度轻量级文字识别/B榜测试数据集/B榜测试数据集/test_label.txt', 'w') as f_w:
#     with open('./output/rec/predicts_chinese_tinyresnet_v2.0.txt') as f:
#         for line in f.readlines():
#             path, text, prob = line.strip().split('\t')
#             content = path[55:] + '\t' + text.replace(' ', '')
#             f_w.write(content)
#             f_w.write('\n')
import joblib
a = joblib.load('/newjixieyingpan/百度轻量级文字识别/训练数据集/eval.dump')
for k, v in a.items():
    print(k, v)
    break