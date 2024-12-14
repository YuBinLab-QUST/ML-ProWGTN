import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale,StandardScaler
from sklearn.metrics import average_precision_score, f1_score, zero_one_loss,\
    hamming_loss,accuracy_score,multilabel_confusion_matrix,jaccard_score,recall_score
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error
from tensorflow.keras.layers import Conv1D, Dense,AveragePooling1D,MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

#导入数据
# data=pd.read_csv('D-po_AAC.csv',header=None)      #0.4845
# label=pd.read_csv('polabel(519).csv',header=None)
# data=pd.read_csv('D-po_CTDC.csv')     #0.6332
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_CTDD.csv')     #0.314672
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_CTDT.csv')     #0.602317
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_CTriad.csv')   #0.619691
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_DDE.csv')      #0.642857
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_DPC.csv')      #0.667954
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_GTPC.csv')     #0.627413
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_PAAC.csv')     #0.716216
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_EBGW.csv')       #0.386100
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_TPC.csv')       #0.760618
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_EDT11.csv')       #TPC得到0.723938
# data=pd.read_csv('D-po_EDT1.csv')          #RPT得到0.689189
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_RPT.csv')         #,使用RPT得到的0.687259
# # data=pd.read_csv('D-po_RPT11.csv')     #使用TPC的方法得到的0.731660
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('D-po_NMBroto.csv')       #0.598456
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-GO.csv')            #维度为912的精度0.598456
# label=pd.read_csv('polabel.csv')

# data=pd.read_csv('D-po_PSSM.csv')     #0.720077
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('psepppssmR.csv')     #0.698842
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('psepppssm.csv')     #0.675676
# label=pd.read_csv('polabel.csv')

# data=pd.read_csv('1-po-MLSI-10.csv')     #0.65444
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-MLSI.csv')     #0.704633
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-MLSI-6.csv')     #0.673745
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-MLSI-10.csv')     #0.68339
# label=pd.read_csv('polabel.csv')

# data=pd.read_csv('po-6-GPDDCT.csv')     #0.795367
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-5-GDDCT.csv')     #0.762548
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-5-PDDCT.csv')     #0.704633
# label=pd.read_csv('polabel.csv')

# data=pd.read_csv('feature_select-6-276.csv')     # 0.703276
# data=pd.read_csv('feature_select-5-GDDCT-274.csv')     #0.601156
# data=pd.read_csv('feature_select-5-PDDCT-610.csv')      #0.678227
# label=pd.read_csv('polabel.csv',header=None)

# data=pd.read_csv('po-wMLDA-40(519).csv',header=None)    #0.94027
# label=pd.read_csv('polabel(519).csv',header=None)
# data=pd.read_csv('po-wMLDA-40(520).csv')   #0.947977
# label=pd.read_csv('polabel(520).csv')

# data=pd.read_csv('po-ALL_9_with_weights.csv')   #维度太大0.00000
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-wMLDAb-9.csv')      #0.982625
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-50-GINI_347.csv')      #0.984556
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('po-ALL_6_with_weights.csv')      #accuracy_score = 0.129344
# data=pd.read_csv('po-weights(6)-wMLDAb(40).csv')      #0.990347
# data=pd.read_csv('output.csv')  #accuracy_score = 0.079151
# data=pd.read_csv('output-3479-40.csv')  #accuracy_score = 0.079151
# data=pd.read_csv('output-3479-40.csv')  #accuracy_score = 0.079151
# label=pd.read_csv('polabel.csv')  #accuracy_score = 0.000000


# data=pd.read_csv('1(519+200).csv',header=None)   #accuracy_score = 0.991655
#
# data=pd.read_csv('po-all-GINI_300.csv')      #
# data=pd.read_csv('po-all-GINI_300.csv',header=None)    #0.712909
# label=pd.read_csv('polabel.csv')
#
# data=pd.read_csv('D-po_ProtBert.csv')         #0.803089
# label=pd.read_csv('polabel.csv')
# data=pd.read_csv('ProtBert-gan(719).csv')       #0.869263    0.851182   0.872045
# label=pd.read_csv('polabel(719).csv',header=None)   #header=None不要将第一行视为列名，并将自动为每一列分配数字索引。
# data=pd.read_csv('feature_select.csv')        #0.793834
# label=pd.read_csv('polabel.csv',header=None)
# data=pd.read_csv('gan_features.csv')            #1.000000
# label=pd.read_csv('polabel(200).csv')
# data=pd.read_csv('gan_features(00200).csv')            #1.000000
# label=pd.read_csv('polabel(200).csv')
# data=pd.read_csv('gan_features(20).csv')         #1.000000
# data=pd.read_csv('gan_features().csv')         #1.000000
# data=pd.read_csv('gan_features(1024-40).csv')         #1.000000
# label=pd.read_csv('增强label(20).csv')
# data=pd.read_csv('gan_features(1000-50).csv')         #1.000000
# label=pd.read_csv('polabel(1000-50).csv')
# data=pd.read_csv('gan_features(0001-50).csv')         #1.000000
# label=pd.read_csv('polabel(0001-50).csv')

# data=pd.read_csv('D-ne_CT.csv')            #accuracy_score = 0.556434
# label=pd.read_csv('nelabel.csv')

# data=pd.read_csv('po-MRMD_500.csv')  #accuracy_score = 0.861004
# data=pd.read_csv('po-GINI_500.csv')  #accuracy_score = 0.791506
# data=pd.read_csv('po-GINI_50.csv')  #accuracy_score = 0.791506
# data=pd.read_csv('po-XGB-500.csv')  #accuracy_score =0.874517
# data=pd.read_csv('po-RandomForest-40.csv')  #accuracy_score =0.878378
# data=pd.read_csv('encoded_features.csv')  #accuracy_score =0.961390
# data=pd.read_csv('encoded_features(损失函数11).csv')  #accuracy_score =0.903475
# data=pd.read_csv('po-weights(3)-wMLDAb(200).csv')  #accuracy_score =0.874517
# data=pd.read_csv('po-weights(3)-wMLDAb(40).csv')  #accuracy_score = 0.984556
# data=pd.read_csv('po-weights(3)-PCA(10).csv')  #accuracy_score = 0.496139
# data=pd.read_csv('po-weights(3)-CCA(10).csv')  #accuracy_score = 0.978764

# data=pd.read_csv('po-encoded_features-10.csv')  #accuracy_score = 0.955598
# data=pd.read_csv('po-encoded_features-20.csv')  #accuracy_score = 0.976834
# data=pd.read_csv('po-encoded_features-30.csv')  #accuracy_score = 0.951737
# data=pd.read_csv('po-encoded_features-40.csv')  #accuracy_score = 0.965251
# data=pd.read_csv('po-encoded_features-50.csv')  #accuracy_score = 0.967181
# data=pd.read_csv('po-encoded_features-60.csv')  #accuracy_score = 0.978764
# data=pd.read_csv('po-encoded_features-70.csv')  #accuracy_score = 0.953668
# data=pd.read_csv('po-encoded_features-80.csv')  #accuracy_score = 0.957529
# data=pd.read_csv('po-encoded_features-90.csv')  #accuracy_score = 0.978764

# data=pd.read_csv('po-encoded_features-60（DE归一化T）.csv')  #accuracy_score = 0.978764

# data=pd.read_csv('po-GINI-90.csv')  #accuracy_score =0.59
# data=pd.read_csv('po-MRMD-10.csv')  #accuracy_score = 0.922780
# data=pd.read_csv('po-XGB-30.csv')  #accuracy_score = 0.920849
# label=pd.read_csv('polabel(519).csv')

data = pd.read_csv('po(1000).csv',header=None)   #
label = pd.read_csv('polabel(1000)',header=None)


print(data.shape)
print(label.shape)

X = np.array(data)   
y = np.array(label)

def get_shuffle(dataset, label):
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset, label
X, y = get_shuffle(X, y)

n=np.shape(X)[1]
label_y=[]
for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]==1:
                label_y.append(j+1)
                break
# print(label_y)
def to_class(pred_y):
    '''
    输入：
        pred_y(ndarray):预测值(各分量为概率)
    作用：
        本函数将输入的pred_y转换成class
    输出:
        pred_y(ndarray):预测值(各分量为0，1)
    '''
    for i in range(len(pred_y)):
        pred_y[i][pred_y[i]>=0.5]=1
        pred_y[i][pred_y[i]<0.5]=0
    return pred_y
from keras.layers import Conv1D
##定义深度学习模型
def build_model(input_dim,num_class):
    # model = models.Sequential()
    model = tf.keras.models.Sequential()   #精度更好
    model.add(Conv1D(filters = 10, kernel_size = 5, padding = 'same', activation= 'relu'))
    model.add(AveragePooling1D(pool_size=2,strides=2,padding="SAME"))
    model.add(Conv1D(filters = 10, kernel_size =  5, padding = 'same', activation= 'relu'))
    model.add(AveragePooling1D(pool_size=2,strides=2,padding="SAME"))

    model.add(Flatten())
    model.add(Dense(int(input_dim), activation = 'relu'))
    model.add(Dense(num_class, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    # loss_function = torch.nn.BCELoss().to(device)
    # Binary Cross - Entropy Loss
    return model

[sample_num,input_dim]=np.shape(X)
# print(X.shape)  #input_dim=348
num_iter=0
##进行交叉验证,并训练
num_class=4
yscore=np.ones((1,num_class))*0.5
yclass=np.ones((1,num_class))*0.5
ytest=np.ones((1,num_class))*0.5

skf=StratifiedKFold(n_splits=5,random_state = 0,shuffle = True)
for train_index,test_index in skf.split(X,label_y):
    train_X,test_X=X[train_index],X[test_index]
    train_Y,test_Y=y[train_index],y[test_index]
    train_X = np.reshape(train_X,(-1,1,input_dim))
    test_X = np.reshape(test_X,(-1,1,input_dim))
    print('--------- 当前是第{0}个iteration ---------'.format(num_iter))

    protein_model =build_model(input_dim,num_class)

    # # 添加回调函数，保存最佳模型参数
    # checkpoint_dir = "E:/多标签蛋白质亚细胞定位/代码/亚细胞定位预测/3多标签分类器/CNN/Model/"
    # os.makedirs(checkpoint_dir, exist_ok=True)
    # checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
    # checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    hist=protein_model.fit(x=train_X,
                    y=train_Y,
                   verbose=0,epochs=5,batch_size=10)     #epochs=80,batch_size=16可以更改
    y_score=protein_model.predict(test_X)

    # 保存模型参数,构建的模型中第一句需要修改  model = keras.models.Sequential()
    # models.save('E:/多标签蛋白质亚细胞定位/代码/亚细胞定位预测/3多标签分类器/CNN/Model/model.h5')

    # print('11:',test_Y.shape)    #(143, 4)
    # print('22:',y_score.shape)   #(143, 4)
    # print('33:',ytest.shape)     #(1, 4)
    # print('44:',yscore.shape)    #(1, 4)

    ###########################################
    # result2=yscore
    # data_csv = pd.DataFrame(data=result2)
    # data_csv.to_csv('CNN-yscore.csv')
    ########################################
    # y_class=to_class(y_score)
    hist=[]
    protein_model=[]
    num_iter+=1

    yscore=np.vstack((yscore,y_score))
    # yclass=np.vstack((yclass,y_class))
    ytest=np.vstack((ytest,test_Y))

def evaluate(y_gt, y_pred, threshold_value=0.5):
    print("thresh = {:.6f}".format(threshold_value))
    y_pred_bin = y_pred >= threshold_value
    OAA=accuracy_score(y_gt, y_pred_bin)
    print("accuracy_score = {:.6f}".format(OAA))
    mAP = average_precision_score(y_gt, y_pred)
    print("mAP = {:.2f}%".format(mAP * 100))
    score_f1_macro = f1_score(y_gt, y_pred_bin, average="macro")
    print("Macro_f1_socre = {:.6f}".format(score_f1_macro))
    score_f1_micro = f1_score(y_gt, y_pred_bin, average="micro")
    print("Micro_f1_socre = {:.6f}".format(score_f1_micro))
    score_F1_weighted = f1_score(y_gt, y_pred_bin, average="weighted")
    print("score_F1_weighted = {:.6f}".format(score_F1_weighted))
    Mconfusion_matri=multilabel_confusion_matrix(y_gt, y_pred_bin)
    print("混淆矩阵",Mconfusion_matri)
    h_loss = hamming_loss(y_gt, y_pred_bin)
    print("Hamming_Loss = {:.6f}".format(h_loss))
    R_loss = label_ranking_loss(y_gt, y_pred_bin)
    print("ranking_loss= {:.6f}".format(R_loss))
    z_o_loss = zero_one_loss(y_gt, y_pred_bin)
    print("zero_one_loss = {:.6f}".format(z_o_loss))
    CV=coverage_error(y_gt, y_pred_bin)
    print("coverage_error= {:.6f}".format(CV))

    jaccard = jaccard_score(y_gt, y_pred_bin, average='samples')
    print("Jaccard Similarity:", jaccard)
    recall = recall_score(y_gt, y_pred_bin, average='samples')
    print("Recall:", recall)

    sepscore_rnn=[]
    sepscore_rnn.append([OAA,mAP, score_f1_macro,score_f1_micro,score_F1_weighted, h_loss, R_loss, z_o_loss,CV])
    result=sepscore_rnn
    data_csv = pd.DataFrame(data=result)
    data_csv.to_csv('CNN-po.csv')
    data=pd.read_csv(r'CNN-po.csv', header=None, names=['OAA', 'AP', 'f1_macro', 'f1_micro', 'F1', 'h_loss', 'R_loss', 'z_o_loss', 'CV'])
    data.to_csv('CNN-po.csv',index=False)
    result1=ytest
    result2=yscore
    result3=yclass

    # 真实标签、预测得分和预测标签数据
    data_csv = pd.DataFrame(data=result2)
    data_csv.to_csv('CNN-yscore-1000--1.csv',header=None)
    data_csv = pd.DataFrame(data=result1)
    data_csv.to_csv('CNN-ytrue-1000--1.csv',header=None)
    data_csv = pd.DataFrame(data=result3)
    data_csv.to_csv('CNN-ypred-1000--1.csv',header=None)

import torch
print(ytest[1:,:])

evaluate (ytest[1:,:],yscore[1:,:], threshold_value=0.5)



#保存模型参数,构建的模型中需要修改成model = keras.models.Sequential()
# models=build_model(input_dim, num_class)
# models.save('E:/多标签蛋白质亚细胞定位/代码/亚细胞定位预测/3多标签分类器/CNN/Model/model.h5')
# models=build_model(input_dim, num_class)
# model_state_dict = models.state_dict()
# torch. save(model_state_dict,'E:/多标签蛋白质亚细胞定位/代码/亚细胞定位预测/3多标签分类器/CNN/Model'++"model.pth.pkl")

# #画roc曲线
# import numpy as np
# from itertools import cycle
# import matplotlib.pyplot as plt
# from scipy import interp
#
# out_dim=num_class
# ##获取 X和 y
# y_test1= np.array(ytest[1:,:])
# y_score1= np.array(yscore[1:,:])
# n_classes = out_dim
# # print(y_test1)
# # Compute ROC curve and ROC area for each class
# # 创建空字典来存储每个类别的假阳性率、真阳性率和ROC曲线下面积。
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# # 循环遍历每个类别，计算该类别的ROC曲线和AUC值
# for i in range(n_classes):
#     # 计算第i个类别的假阳性率（FPR）、真阳性率（TPR）和阈值
#     fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_score1[:, i])
#     # 计算第i个类别的ROC曲线下面积（AUC）
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# # 将计算得到的micro-average的FPR和TPR分别赋值给fpr["micro"]和tpr["micro"]
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test1.ravel(), y_score1.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  #将计算得到的AUC值赋值给roc_auc["micro"]
#
# plt.figure()   # 创建一个新的图形窗口。
# lw = 2   #设置线条的宽度。
#
# # First aggregate all false positive rates
# # 将所有类别的假阳性率合并为一个唯一的数组。
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)   #创建一个与所有假阳性率相同大小的数组，用于存储平均真阳性率。
#
# for i in range(n_classes):
#     # 使用interp函数在所有的假阳性率（all_fpr）上插值计算平均真阳性率（mean_tpr），并将其累加到mean_tpr中/
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
# # Finally average it and compute AUC
# # 得到在不同假阳性率下的平均真阳性率。
# mean_tpr /= n_classes
# # 使用auc函数计算得到macro-average的ROC曲线下面积（AUC）。最终，将计算得到的macro-average的FPR、TPR和AUC分别存储在fpr["macro"]、tpr["macro"]和roc_auc["macro"]中
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# # Plot all ROC curves
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (AUC = {0:0.4f})'
#                ''.format(roc_auc["micro"]),
#          color='orangered', linewidth=2)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (AUC = {0:0.4f})'
#                ''.format(roc_auc["macro"]),
#          color='blue', linewidth=2)
#
# #创建一个颜色循环迭代器，用于绘制不同类别的ROC曲线。
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#
#
# "用来绘制一条对角线的直线，即y=x线"
# "[0, 1]：表示绘制直线的x坐标和y坐标范围，即从(0,0)到(1,1)。"
# "k：表示直线的颜色为黑色（'k'代表黑色）。"
# "lw=lw：设置直线的宽度为之前定义的lw变量的值。"
# plt.plot([0, 1], [0, 1], 'k', lw=lw)
#
# # 设置x轴和y轴的范围
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# # 分别设置x轴和y轴的标签
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # 设置图形的标题。
# plt.title('Some extension of Receiver operating characteristic to multi-label')
# # 添加图例，显示micro-average和macro-average的AUC值。
# plt.legend(loc="lower right")
# #plt.savefig("图.png", dpi=3000, bbox_inches = 'tight')
# # 展示绘制好的ROC曲线图形。
# plt.show()




######################################################################################################
# import numpy as np
# import pandas as pd
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import scale, StandardScaler
# from sklearn.metrics import average_precision_score, f1_score, zero_one_loss, hamming_loss, accuracy_score, \
#     multilabel_confusion_matrix
# from tensorflow.keras import layers
# from tensorflow.keras import models
# from tensorflow.keras import optimizers
# from sklearn.metrics import label_ranking_loss
# from sklearn.metrics import coverage_error
# from tensorflow.keras.layers import Conv1D, Dense, AveragePooling1D, MaxPooling1D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten
#
# # 导入数据
# data = pd.read_csv('po-100-2-50-4-2.csv')
# label = pd.read_csv('polabel-100-2-50-4-2.csv')
# print(data.shape)
# print(label.shape)
# num_class = 4
# X = np.array(data)
# y = np.array(label)
# n = np.shape(X)[1]
# label_y = []
# for i in range(y.shape[0]):
#     for j in range(y.shape[1]):
#         if y[i, j] == 1:
#             label_y.append(j + 1)
#             break
#
# def to_class(pred_y):
#     '''
#     输入：
#         pred_y(ndarray):预测值(各分量为概率)
#     作用：
#         本函数将输入的pred_y转换成class
#     输出:
#         pred_y(ndarray):预测值(各分量为0，1)
#     '''
#     for i in range(len(pred_y)):
#         pred_y[i][pred_y[i] >= 0.5] = 1
#         pred_y[i][pred_y[i] < 0.5] = 0
#     return pred_y
#
# def build_model(input_dim, num_class):
#     model = models.Sequential()
#     model.add(Conv1D(filters=10, kernel_size=5, padding='same', activation='relu'))
#     model.add(AveragePooling1D(pool_size=2, strides=2, padding="SAME"))
#     model.add(Conv1D(filters=10, kernel_size=5, padding='same', activation='relu'))
#     model.add(AveragePooling1D(pool_size=2, strides=2, padding="SAME"))
#     model.add(Flatten())
#     model.add(Dense(int(input_dim), activation='relu'))
#     model.add(Dense(num_class, activation='softmax', name="Dense_2"))
#     model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
#     return model
#
# [sample_num, input_dim] = np.shape(X)
#
# num_iter = 0
# ##进行交叉验证,并训练
# yscore = np.ones((1, num_class)) * 0.5
# yclass = np.ones((1, num_class)) * 0.5
# ytest = np.ones((1, num_class)) * 0.5
# skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
# for train_index, test_index in skf.split(X, label_y):
#     train_X, test_X = X[train_index], X[test_index]
#     train_Y, test_Y = y[train_index], y[test_index]
#     train_X = np.reshape(train_X, (-1, 1, input_dim))
#     test_X = np.reshape(test_X, (-1, 1, input_dim))
#
#     print('--------- 当前是第{0}个iteration ---------'.format(num_iter))
#     print(train_X.shape)
#     protein_model = build_model(input_dim, num_class)
#     hist = protein_model.fit(x=train_X,
#                              y=train_Y,
#                              verbose=0, epochs=80, batch_size=16)
#     y_score = protein_model.predict(test_X)
#     print(y_score.shape)
#     result2 = yscore
#     data_csv = pd.DataFrame(data=result2)
#     data_csv.to_csv('CNN-yscore.csv')
#     # y_class=to_class(y_score)
#     hist = []
#     protein_model = []
#     num_iter += 1
#     yscore = np.vstack((yscore, y_score))
#     # yclass=np.vstack((yclass,y_class))
#     ytest = np.vstack((ytest, test_Y))
#
# def evaluate(y_gt, y_pred, threshold_value=0.5):
#     print("thresh = {:.6f}".format(threshold_value))
#     y_pred_bin = y_pred >= threshold_value
#
#     OAA = accuracy_score(y_gt, y_pred_bin)
#     print("accuracy_score = {:.6f}".format(OAA))
#     mAP = average_precision_score(y_gt, y_pred)
#     print("mAP = {:.2f}%".format(mAP * 100))
#     score_f1_macro = f1_score(y_gt, y_pred_bin, average="macro")
#     print("Macro_f1_socre = {:.6f}".format(score_f1_macro))
#     score_f1_micro = f1_score(y_gt, y_pred_bin, average="micro")
#     print("Micro_f1_socre = {:.6f}".format(score_f1_micro))
#     score_F1_weighted = f1_score(y_gt, y_pred_bin, average="weighted")
#     print("score_F1_weighted = {:.6f}".format(score_F1_weighted))
#     Mconfusion_matri = multilabel_confusion_matrix(y_gt, y_pred_bin)
#     print("混淆矩阵", Mconfusion_matri)
#     h_loss = hamming_loss(y_gt, y_pred_bin)
#     print("Hamming_Loss = {:.6f}".format(h_loss))
#     R_loss = label_ranking_loss(y_gt, y_pred_bin)
#     print("ranking_loss= {:.6f}".format(R_loss))
#     z_o_loss = zero_one_loss(y_gt, y_pred_bin)
#     print("zero_one_loss = {:.6f}".format(z_o_loss))
#     CV = coverage_error(y_gt, y_pred_bin)
#     print("coverage_error= {:.6f}".format(CV))
#
#     sepscore_rnn = []
#     sepscore_rnn.append([OAA, mAP, score_f1_macro, score_f1_micro, score_F1_weighted, h_loss, R_loss, z_o_loss, CV])
#     result = sepscore_rnn
#     data_csv = pd.DataFrame(data=result)
#     data_csv.to_csv('CNN-po.csv')
#     data = pd.read_csv(r'CNN-po.csv', header=None,
#                        names=['OAA', 'AP', 'f1_macro', 'f1_micro', 'F1', 'h_loss', 'R_loss', 'z_o_loss', 'CV'])
#     data.to_csv('CNN-po.csv', index=False)
#     result1 = ytest
#     result2 = yscore
#     result3 = yclass
#
#     data_csv = pd.DataFrame(data=result2)
#     data_csv.to_csv('CNN-yscore.csv')
#     # data_csv = pd.DataFrame(data=result1)
#     # data_csv.to_csv('CNN-ytrue.csv')
#     # data_csv = pd.DataFrame(data=result3)
#     # data_csv.to_csv('CNN-ypred.csv')
#
# evaluate(ytest[1:, :], yscore[1:, :], threshold_value=0.5)
#
# # 画roc曲线
# import numpy as np
# from itertools import cycle
# import matplotlib.pyplot as plt
# from scipy import interp
#
# out_dim = num_class
# ##获取 X和 y
# y_score1 = np.array(yscore[1:, :])
# y_test1 = np.array(ytest[1:, :])
# n_classes = out_dim
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_score1[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test1.ravel(), y_score1.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# plt.figure()
# lw = 2
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# # Finally average it and compute AUC
# mean_tpr /= n_classes
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# # Plot all ROC curves
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (AUC = {0:0.4f})'
#                ''.format(roc_auc["micro"]),
#          color='orangered', linewidth=2)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (AUC = {0:0.4f})'
#                ''.format(roc_auc["macro"]),
#          color='blue', linewidth=2)
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#
# plt.plot([0, 1], [0, 1], 'k', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-label')
# plt.legend(loc="lower right")
# # plt.savefig("图.png", dpi=3000, bbox_inches = 'tight')
# plt.show()