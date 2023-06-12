import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#读取文件
data= pd.read_excel("C:\\Users\\22389\\Desktop\\diabetes_prediction_dataset.xlsx")


#x因变量 y自变量
x=data.drop(['diabetes'],axis=1)
y=data['diabetes']
feat_labels = x.columns[0:]
target_labels = y

#训练集和测试集的划分
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4)

#数据的标准化RobustScaler方法
from sklearn.preprocessing import RobustScaler
cols=x_train.columns
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train=pd.DataFrame(x_train,columns=[cols])
x_test=pd.DataFrame(x_test,columns=[cols])

#计算10个决策树拟合的精确度
model=RandomForestClassifier(n_estimators=10,
                             random_state=0)
model.fit(x_train,y_train)
#预测概率
p=model.predict_proba(x_test)
y_pred=model.predict(x_test)
print("10个决策树的模型准确率得分 ：{0:0.4f}".format(accuracy_score(y_test, y_pred)))

#对特征重要性进行提取和排序
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print(importances)
for i in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (i, 30, feat_labels[indices[i]], importances[indices[i]]))

#作重要性排序图
plt.title('Feature importance')
plt.bar(range(x_train.shape[1]), importances[indices], align='center')
plt.xticks(range(x_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()
#绘制ROC曲线
fpr0, tpr0, thresholds0 = roc_curve(y_test, p[:,0], pos_label = 0)
fpr1, tpr1, thresholds1 = roc_curve(y_test, p[:,1], pos_label = 1)
plt.plot(fpr0,tpr0)
plt.plot(fpr1,tpr1)
plt.xlabel("fpr",fontsize=20)
plt.ylabel("tpr",fontsize=20)
plt.show()

#auc指标
print('AUC指标 :{0:0.4f}'.format(roc_auc_score(y_true=y_test,y_score=y_pred)))

from sklearn import tree
import graphviz

# 以第三棵决策树为例，输出决策森林。需要将下载该模块程序，再将其添加到环境变量，再重新安装该代码包，这个图比较大，可能没什么看头
# tree_3 = model.estimators_[2]
#
# dot_data = tree.export_graphviz(tree_3,
#                                 filled = True,
#                                 rounded = True,
#                                 special_characters = True)
#
# graph = graphviz.Source(dot_data)
# graph.render('tree_3')

'''计算100个决策数的模型准确率得分，相比10棵树的精度，没有明显提升'''
rfc_100=RandomForestClassifier(n_estimators=100,
                               random_state=0)
rfc_100.fit(x_train,y_train)
y_pred_100 = rfc_100.predict(x_test)
print('100个决策树的模型准确率得分 ：{0:0.4f}'.format(accuracy_score(y_test,y_pred_100)))
y_pred=model.predict(x_test)
y_pred_train = model.predict(x_train)

print("测试集模型精度:{0:0.4f}".format(accuracy_score(y_test,y_pred)))
print("训练集模型精度:{0:0.4f}".format(accuracy_score(y_train,y_pred_train)))

#混淆矩阵及精度得分
cm = confusion_matrix(y_test,y_pred)
print('混淆矩阵\n\n',cm)
print('\n没病预测没病=',cm[0,0])
print('\n有病预测有病=',cm[1,1])
print('\n没变预测有病=',cm[0,1])
print('\n有病预测没病=',cm[1,0])

#混淆矩阵热力图
sns.heatmap(cm,annot=True,fmt='d',cmap='rocket')
plt.show()
#输出模型评价指数
print(classification_report(y_test,y_pred))






