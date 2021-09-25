import itertools
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2
from sklearn import preprocessing
from feature_selector import FeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
from sklearn.utils import shuffle
import glob

# 读取图像
def read_file(file):
    img = sitk.GetArrayFromImage(sitk.ReadImage(file))
    if img.shape != (512,512):
        print('This file size is wrong! -----------')
        print(file)
        img = np.resize(img,(512,512))
    return img

# 得到文件夹下所有的子文件路径的列表
def get_file(path):
    files = []
    if not os.path.exists(path):
        return -1
    for filepath, dirs, names in os.walk(path):
        for filename in names:
                files.append(os.path.join(filepath, filename))
    return files

# 得到文件夹下所有的子文件夹路径的列表
# def get_folders(path):
#     folders = []
#     if not os.path.exists(path):
#         return -1
#     for filepath, dirs, names in os.walk(path):
#         # print(filepath)
#         if not(filepath.endswith('data')) and not(filepath.endswith('positive')) and not(filepath.endswith('negative'))\
#                 and not(filepath.endswith('M_P')):
#          folders.append(filepath)
#     return folders

# 遍历标签为1和0文件夹的所有dcm及roi文件进行特征提取批处理
def get_features(sortedImageDir,maskDir):
    result_all = []
    for path in sortedImageDir:
        # print(path)
        real_name = path.split("/")[-1].split("_Image.nii.gz")[0]
        print(real_name)
        mask_name = real_name + "_Resize_Mask.nii.gz"
        mask_path = os.path.join(maskDir, mask_name)
        print(mask_path)
        print(path)
        result = extractor.execute(path, mask_path)
        result_all.append(result)
    print(len(result_all))
    return result_all

# 标准化 -mean/std
def standardscaler_feature(train_feature):
    columns_name = train_feature.columns
    scaler = preprocessing.StandardScaler().fit(train_feature)
    return columns_name, scaler

# 主成分分析
def pca_anlysis(feature_train, feature_test, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(feature_train)
    features_pca_train = pca.fit_transform(feature_train)
    features_pca_test = pca.fit_transform(feature_test)
    return features_pca_train, features_pca_test

# 特征筛选
def select_feature(feature, label):
    fs = FeatureSelector(data=feature, labels=label)
    # Missing value 找缺失值
    fs.identify_missing(missing_threshold=0.3)
    # Single_unique_value 找单一值
    fs.identify_single_unique()
    # Collinear value 贡献性特征 去掉相关性低的特征
    fs.identify_collinear(correlation_threshold=0.98, one_hot=False)
    invalid_feature_d = fs.ops  # 字典
    invalid_feature = []
    for k, v in invalid_feature_d.items():
        for feature in v:
            if feature not in invalid_feature:
                invalid_feature.append(feature)
    return invalid_feature


# method: f_clssif 基于方差的feature selection/chi2/RFE
# f_clssif: Compute the ANOVA F-value for the provided sample.
def select_feature_sklearn(X_train, label, feature_name, method, feature_num):
    if method == 'f_classif' or method == 'chi2':
        if method == 'f_classif':
            fs = SelectKBest(f_classif, k=feature_num)
        else:
            fs = SelectKBest(chi2, k=feature_num)

        fs.fit(X_train, label)
        feature_n = fs.get_support(True)
        # print('feature_n', feature_n)
        feature_select = [feature_name[i] for i in range(len(feature_name)) if i in feature_n]
        print('feature_select', feature_select)
    else:
        fs = RFE(DecisionTreeClassifier(), n_features_to_select=feature_num)
        fs.fit(X_train, label)
        feature_n = fs.support_
        feature_select = np.where(feature_n == True)
        feature = feature_select[0].tolist()
        print('feature', feature)
        feature_se = []
        for j in range(len((feature_name).tolist())):
            if j in feature:
                feature_se.append(((feature_name).tolist())[j])
        print('feature_select', feature_se)
    return feature_n

# Classifier: DecisionTreeClassifier/SVC/RandomForestClassifier/LogisticRegression
def classify_analysis(X_train, y_train, method='SVC'):
    clf = []
    print('method', method)
    if method == 'DecisionTreeClassifier':
        clf = DecisionTreeClassifier()
    if method == 'SVC':
        clf = SVC(kernel="linear", probability=True)
    if method == 'RandomForestClassifier':
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    if method == 'LogisticRegression':
        clf = LogisticRegression()
    if method == 'AE':
        clf = MLPClassifier(random_state=1, max_iter=500)
    if method == 'LDA':
        clf = LDA()
    if method == 'Adaboost':
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    if method == 'GP':
        kernel1 = 1.0 * RBF(1.0)
        clf = GaussianProcessClassifier(kernel=kernel1,random_state=0)
    clf.fit(X_train, y_train)
    return clf

def clf_result(clf, feature, label):
    print('accuracy_score')
    print(accuracy_score(label, clf.predict(feature)))
    print(classification_report(label, clf.predict(feature)))

# 画散点图
def plot_dot(features_1, features_2, label):
    plt.figure()  # 新建一张图进行绘制
    plt.scatter(features_1, features_2, c=label, edgecolor='k')  # 绘制两个主成分组成坐标的散点图
    plt.show()

# Measurement of TP, FP, TN, FN
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

# Evaluation metrics measurements.
def evaluation(TP, FP, TN, FN):
    # Sensitivity, hit rate, recall, or true positive rate
    Sensitivity = TP / (TP + FN)
    # Specificity or true negative rate
    Specificity = TN / (TN + FP)
    # Overall accuracy
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    # Precision or positive predictive value
    Precision = TP / (TP + FP)
    # F1-score
    F1_score = (2*Precision*Sensitivity)/(Precision + Sensitivity)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    return Sensitivity,Specificity,Accuracy,Precision,F1_score,NPV,FPR,FNR,FDR

# ROC curve drawing
def ROC_CURVE(true_test,network_test,true_train,network_train):
    fpr_test, tpr_test, _ = roc_curve(true_test, network_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    fpr_train, tpr_train, _ = roc_curve(true_train, network_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    plt.figure()
    plt.cla()
    lw = 2
    plt.plot(fpr_test, tpr_test, color='darkorange',
             lw=lw, label='ROC curve of test set (area = %0.4f)' % roc_auc_test)
    plt.plot(fpr_train, tpr_train, color='b',
             lw=lw, label='ROC curve of train set (area = %0.4f)' % roc_auc_train)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("./result/roc.png")
    plt.show()

# confusion matrix
def plot_confusion_matrix(confusion_mat):
    plt.cla()
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = ['test_label_0', 'test_label_1']
    plt.xticks([0,1], tick_marks)
    plt.yticks([0,1], tick_marks)
    thresh = confusion_mat.max() / 2.
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, confusion_mat[i, j],
             horizontalalignment="center",
             color="white" if confusion_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./result/confusion_mat.png')
    plt.show()

def feature_select_and_predict(X_train, y_train, X_test, y_test, select_sklearn=False, method_sk='f_classif', feature_num=10,
                           method_classification='DecisionTreeClassifier'):
    # 特征筛选
    print('-------------Preliminary feature screening---------------')
    invalid_feature = select_feature(X_train, y_train)
    X_train = X_train.drop(invalid_feature, axis=1)
    X_test = X_test.drop(invalid_feature, axis=1)
    print('X_train', X_train.shape)
    print('X_test', X_test.shape)

    # 标准化
    columns_name, scaler = standardscaler_feature(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if select_sklearn == True:
        print('--------------Further feature screening---------------')
        feature_n = select_feature_sklearn(X_train, y_train,
                                                          columns_name, method=method_sk, feature_num=feature_num)
        X_train = X_train[:, feature_n]
        X_test = X_test[:, feature_n]
        print('X_train', X_train.shape)
        print('X_test', X_test.shape)
    clf = classify_analysis(X_train, y_train, method=method_classification)
    print("--------Train set result-----------")
    clf_result(clf, X_train, y_train)
    print("--------Test set result-----------")
    clf_result(clf, X_test, y_test)

    print('----------------end---------------------')
    # Calculate the prediction result of train and test set.
    predict_test = clf.predict(X_test)
    predict_train = clf.predict(X_train)

    # Plot confusion matrix and ROC_Curve
    con_matrix = confusion_matrix(y_pred=predict_test, y_true=y_test)

    # ROC curve drawing
    ROC_CURVE(y_test, predict_test, y_train, predict_train)
    # confusion matrix
    plot_confusion_matrix(con_matrix)


# 特征提取器的设置
param_path = "E:\PycharmProjects\pyradiomics\pyradiomics\pyradiomics-master\examples\exampleSettings\exampleMR_5mm.yaml"
extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(param_path,)
extractor.enableAllFeatures()
extractor.enableAllImageTypes()
