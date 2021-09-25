########################################################################################################################
# Developed by Wendy Li, Steve Zhang, Philips Healthcare
# If using this script for your feature extraction project, welcome to contact us and add us as a co-authors
# The script is only suitable for a simple binary classification task, for more complicated tasks you should revise code
# email:Wendy.LI@philips.com steve.zhang_1@philips.com
########################################################################################################################
from config import *
import pandas as pd
print('HELLOWORLD')
image_dir = 'E:\PycharmProjects\image'
mask_dir = 'E:\PycharmProjects\mask'
# 得到文件夹下所有的子文件夹路径的列表
# def get_folders(param):
#     folders = []
#     for filepath, dirs, names in os.walk(param):
#         for name in names:
#             img_dir = os.path.join(param, name)
#             print(image_dir)
#             folders.append(img_dir)


# 遍历文件夹，按照文件首字母abcd排序
for filepath, dirs, names in os.walk(image_dir):
    imgPaths = []
    for name in names:
        img_path = os.path.join(image_dir, name)
        # print(img_path)
        imgPaths.append(img_path)
    sortedImageDir = sorted(imgPaths)

for path in sortedImageDir:
    # print(path)
    real_name = path.split("/")[-1].split("_Image.nii.gz")[0]
    print(real_name)


# 遍历文件夹，完成特征提取
result_all = get_features(sortedImageDir,mask_dir)
result_all = pd.DataFrame(result_all)

# result_all.to_csv('result_all.csv')

# # 加上标签列，完善特征DataFrame
# label_negative = [0]
# label_negative = [val for val in label_negative for i in range(27)]
# label_positive = [1]
# label_positive = [val for val in label_positive for i in range(48)]
# label_all = label_negative + label_positive
# result_all.insert(0, 'label', label_all)
label_dir = "/root/qiteam/LiYing/Philips/example_BJ/labe_prostate.csv"
label = pd.read_csv(label_dir,usecols=[1])
# print(label)
#
result_all.insert(0,'label',label)
result_all.to_csv('./result/result_all.csv')
#
# 将excel各行随机打乱
result_all = shuffle(result_all)

# 生成完整的组学特征表格，把不必要的信息进行删除（pyradiomics提取的特征向量中包含diagnostics字段可删除）
cols = [i for i in result_all.columns.tolist() if not i.startswith('diagnostics')]
result_all = result_all[cols]

# Train test split on dataframe
msk = np.random.rand(len(result_all)) <= 0.8
train_all = result_all[msk]
test_all = result_all[~msk]
train_all.to_csv('./result/train_features.csv')
test_all.to_csv('./result/test_features.csv')

# Make a prediction based on a designed ML pipeline
train_file = r'./result/train_features.csv'
test_file =  r'./result/test_features.csv'
df_train = pd.read_csv(train_file, index_col=0, header=0)
df_test = pd.read_csv(test_file, index_col=0, header=0)
label_train = df_train['label']
feature_name =df_train.columns.tolist()[1:]
print('feature_name', feature_name)
feature_train = df_train[feature_name]
label_test = df_test['label']
feature_test = df_test[feature_name]

print('feature_train', feature_train.shape)
print('feature_test', feature_test.shape)

feature_select_and_predict(feature_train, label_train, feature_test, label_test, select_sklearn=True, method_sk='f_classif', feature_num=20,
                             method_classification='AE')



