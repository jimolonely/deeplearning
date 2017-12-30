
# 准备数据

```python
import pandas as pd

train_data = pd.read_csv('training.csv')
# 指出类别到数字的转换
labels = set(train_data['class'])
i = 0
label_to_digit = {}
for label in labels:
    label_to_digit[label] = i
    i += 1
print(label_to_digit)
# {'water': 0, 'forest': 1, 'orchard': 2, 'farm': 3, 'grass': 4, 'impervious': 5}

# 替换
for k,v in label_to_digit.items():
    train_data = train_data.replace(k,v)
train_data.head()

train_data.to_csv('train_data.csv')
d = pd.read_csv('train_data.csv')
d.head()

# 同样处理测试数据
# 不过要保证分类标号一样
test_data = pd.read_csv('testing.csv')
test_data.head()
# 替换
for k,v in label_to_digit.items():
    test_data = test_data.replace(k,v)
test_data.head()
test_data.to_csv('test_data.csv')
```

# 初始尝试训练
在数据未经任何处理的情况下看看可以达到什么效果,然后发现NB算法容错效果最好.
```python
# 逻辑回归
logreg = linear_model.LogisticRegression(C=0.000001)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc) # 0.48

# 高斯朴素贝叶斯
gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc) # 0.687

# 随机森林
clf = RandomForestClassifier(n_estimators=10)
y_pred = clf.fit(X_train,y_train).predict(X_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc) # 0.63
```

可以看到,很多分类算法效果都不好,所以需要查查原因.
# 查看数据分布
经过分析,数据的分布是非常不均匀的,所以导致很多算法都预测出错,只管看看数据分布:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAIN_DATA_FILE,
    target_dtype=np.int,
    features_dtype=np.float32,
    target_column=0
)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TEST_DATA_FILE,
    target_dtype=np.int,
    features_dtype=np.float32,
    target_column=0
)
# print(training_set)
X, y = training_set.data, training_set.target
print(X)
print(y)
X_test, y_test = test_set.data, test_set.target
print(X_test)
print(y_test)

train_count = []
test_count = []
xs = []
for i in range(6):
    xs.append(i)
    train_count.append(list(y).count(i))
    test_count.append(list(y_test).count(i))
print(train_count)
print(test_count)

# 画图
plt.figure(1, figsize=(9, 9))
plt.subplot(221)
plt.bar(xs, train_count)
plt.subplot(222)
plt.pie(train_count, autopct='%.2f', labels=xs)
plt.subplot(223)
plt.bar(xs, test_count)
plt.subplot(224)
plt.pie(test_count, autopct='%.2f', labels=xs)
plt.show()
```
![data](./image/000.png)

# 如何处理样本不均衡问题?
常用的方法有以下几种:

1. 欠采样(Under-Sampling)
```
通过消除占多数的类的样本来平衡类分布；直到多数类和少数类的实例实现平衡.
但可能会丢失重要数据
```
2. 过采样(Over-Sampling)
```
通过复制少数类来增加其中的实例数量，从而可增加样本中少数类的代表性.
这种方法不会带来信息损失。表现优于欠采样.但可能过拟合.
```
3. 基于聚类的过采样（Cluster-Based Over Sampling）
```
在这种情况下，K-均值聚类算法独立地被用于少数和多数类实例。
这是为了识别数据集中的聚类。随后，每一个聚类都被过采样以至于相同类的所有聚类有着同样
的实例数量，且所有的类有着相同的大小.
这种聚类技术有助于克服类之间不平衡的挑战,但和过采样一样,可能过拟合.
```
4. 合成少数类过采样技术（SMOTE）

论文地址: [https://www.jair.org/media/953/live-953-2037-jair.pdf](https://www.jair.org/media/953/live-953-2037-jair.pdf)
```
synthetic minority over-sampling technique(SMOTE)
基本思想是:从少数类中把一个数据子集作为一个实例取走，接着创建相似的新合成的实例。
这些合成的实例接着被添加进原来的数据集。新数据集被用作样本以训练分类模型。
通过随机采样生成的合成样本而非实例的副本，可以缓解过拟合的问题。不会损失有价值信息。
但当生成合成性实例时，SMOTE 并不会把来自其他类的相邻实例考虑进来。
这导致了类重叠的增加，并会引入额外的噪音。
```
上面都是通过重采样来处理不平衡,结合本题,如果使用过采样,我们需要对另外5个类都增加数据.
下面还有一些集成技术.

[http://scikit-learn.org/stable/modules/ensemble.html](http://scikit-learn.org/stable/modules/ensemble.html)

1. Bagging
```
生成多个个不同替换的引导训练样本，并分别训练每个自举算法上的算法，然后再聚合预测.
每个样本与原数据集有相同的分布,所以,只有当原分类器表现良好时才能使最后结果更好.
常用的: Bagging methods, Forests of randomized trees
```
2. Boosting
```
和Bagging相反,Boosting是将多个分类器串起来减少误差.
 AdaBoost, Gradient Tree Boosting
```

下面就是使用上述方法进行的训练结果.

# 训练数据

## 1.使用Bagging

我采用的方式是将1类数据随机分成几份(4,5,6),然后分别和另外几类进行训练,以此达到比较均衡的数据
分布,最后再进行投票选出结果,此选择过程是选择预测结果出现最多的类别.

```python
data, data_test = read_train_test_data()

np.random.shuffle(data)
X_test, y_test = split_data_label(data_test)

X_1, y_1, X_n1, y_n1 = split_class_1(data)
pred_record = []
y_pred = None
al = int(len(X_1) / 6)
gnb = GaussianNB()
for i in range(6):
    xt1 = X_1[i * al:(i + 1) * al, :]
    yt1 = y_1[i * al:(i + 1) * al]
    X_temp = np.concatenate((xt1, X_n1), axis=0)
    y_temp = np.concatenate((yt1, y_n1), axis=0)
    # print(temp.shape)
    if i == 0:
        y_pred = gnb.fit(X_temp, y_temp).predict(X_test)
    else:
        y_p = gnb.fit(X_temp, y_temp).predict(X_test)
        y_pred = np.concatenate((y_pred, y_p))

        # acc = metrics.accuracy_score(y_test, y_pred)
        # print(acc)
y_pred = y_pred.reshape(6, -1)
print(y_pred.shape)

col_len = y_pred.shape[1]
# col_len = 5
y_pred_final = []
for i in range(col_len):
    choose = Counter(list(y_pred[:, i])).most_common(1)[0][0]
    # print(choose)
    y_pred_final.append(choose)

acc = metrics.accuracy_score(y_test, y_pred_final)
print(acc)

# 看看预测值和测试值有哪些差别
cnt = {}
test_cnt = {}
for i in range(y_test.shape[0]):
    test_cnt[y_test[i]] = test_cnt.get(y_test[i], 0) + 1
    if y_pred_final[i] != y_test[i]:
        cnt[y_test[i]] = cnt.get(y_test[i], 0) + 1

print(cnt)
key = list(cnt.keys())
val = list(cnt.values())

# 计算每种类别预测错误的百分比
err_percent = []
for k in test_cnt.keys():
    err_percent.append(cnt.get(k, 0) / test_cnt.get(k))
print(err_percent)
plt.figure(1, figsize=(20, 5))
plt.subplot(121)
plt.bar(key, val)
plt.xlabel('class')
plt.ylabel('error cnt')
plt.subplot(122)
# plt.pie(err_percent, labels=list(test_cnt.keys()), autopct='%.2f')
plt.bar(list(test_cnt.keys()), err_percent)
plt.ylabel('percent')
plt.xlabel('class')
plt.show()
```
经过多次求得平均结果:
```
0.707
```

准确度依然保持70%左右,所以来看看到底是哪些预测错了:

下图展示了每种类别预测错误的个数,右边是百分比,经过多次实验,趋势基本相同

![002](./image/002.png)

可以看到,类别2超过50%都错了,其次是第4类,为什么呢?

根据实际情况,1类是森林,4类是草地,而森林和草地看起来相似,而且森林里是有很多草地的,
所以卫星对这2类分不清楚.

## 2.使用重采样技术

[技术文档](http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html#module-imblearn.combine)

### 2.1 SMOTE
```python
# 看看采样前类别分布
print(sorted(Counter(y_train).items()))

# 重采样
X_resampled, y_resampled = SMOTE().fit_sample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))

cls = GaussianNB()

y_pred = cls.fit(X_resampled, y_resampled).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
```
结果:可看到所有数据都是一样的大小.
```
[(0.0, 205), (1.0, 7431), (2.0, 53), (3.0, 1441), (4.0, 446), (5.0, 969)]
[(0.0, 7431), (1.0, 7431), (2.0, 7431), (3.0, 7431), (4.0, 7431), (5.0, 7431)]
0.716666666667
```
### 2.2 降采样
```python
X_resampled, y_resampled = ClusterCentroids().fit_sample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))

cls = GaussianNB()

y_pred = cls.fit(X_resampled, y_resampled).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
```
结果:降采样就差了很多
```
[(0.0, 53), (1.0, 53), (2.0, 53), (3.0, 53), (4.0, 53), (5.0, 53)]
0.62
```
### 2.3 结合过采样和降采样
```python
# 结合降采样和过采样
from imblearn.combine import SMOTEENN

X_resampled, y_resampled = SMOTEENN().fit_sample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))

cls = GaussianNB()

y_pred = cls.fit(X_resampled, y_resampled).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
```
结果:可看到采样后的结果还是偏向与过采样.
```
[(0.0, 7429), (1.0, 7200), (2.0, 7431), (3.0, 7391), (4.0, 7428), (5.0, 7414)]
0.716666666667
```
这是目前能达到的最好结果了.

## 3.分别训练
既然1类别太多,为何不单独提出来训练,先只考虑其中不易混淆的几个类别,然后将易混淆的分类单独
训练.

下面是分别剔除1类和剔除1,4类的训练.

### 3.1 保留5个类
下面分别用上面的3个方法再次进行训练.

```python
# 提取出5个类别,把1(森林去掉)
x_temp = np.array([d_train[i] for i in range(d_train.shape[0]) if d_train[i][-1] != 1])
x_train, y_train = split_data_label(x_temp)
x_temp = np.array([list(d_test[i]) for i in range(d_test.shape[0]) if d_test[i][-1] != 1])
x_test, y_test = split_data_label(x_temp)

# 原贝叶斯方法
cls = GaussianNB()

y_pred = cls.fit(x_train, y_train).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))

# 集成

bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier())

y_pred = bbc.fit(x_train, y_train).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))

# 结合降采样和过采样

X_resampled, y_resampled = SMOTE().fit_sample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))

cls = GaussianNB()

y_pred = cls.fit(X_resampled, y_resampled).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))

```
结果: 同时打印了混淆矩阵.可看出还是集成最好.
```
0.779279279279
[[37  0  2  4  3]
 [ 0 32 12  3  0]
 [ 0  2 46  3  2]
 [ 1  1  7 24  3]
 [ 0  0  2  4 34]]
0.810810810811
[[38  0  2  4  2]
 [ 0 39  5  3  0]
 [ 0  4 44  4  1]
 [ 1  2  8 25  0]
 [ 0  0  3  3 34]]
[(0.0, 1441), (2.0, 1441), (3.0, 1441), (4.0, 1441), (5.0, 1441)]
0.765765765766
[[37  0  3  3  3]
 [ 0 37  7  3  0]
 [ 0  3 43  6  1]
 [ 0  2  6 25  3]
 [ 5  0  1  6 28]]
```
### 3.2 提取4个类
将1类和4类去掉.
```python
# 提取出4个类别,把1(森林)和4(草地)去掉
x_temp = np.array([d_train[i] for i in range(d_train.shape[0]) if d_train[i][-1] != 1])
x_temp = np.array([x_temp[i] for i in range(x_temp.shape[0]) if x_temp[i][-1] != 4])
x_train, y_train = split_data_label(x_temp)
x_temp = np.array([d_test[i] for i in range(d_test.shape[0]) if d_test[i][-1] != 1])
x_temp = np.array([x_temp[i] for i in range(x_temp.shape[0]) if x_temp[i][-1] != 4])
x_test, y_test = split_data_label(x_temp)

cls = GaussianNB()

y_pred = cls.fit(x_train, y_train).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))

# 集成

bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier())

y_pred = bbc.fit(x_train, y_train).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))

# 结合降采样和过采样

X_resampled, y_resampled = SMOTE().fit_sample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))

cls = GaussianNB()

y_pred = cls.fit(X_resampled, y_resampled).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))
```
结果:可看到准确率又提高了
```
0.827956989247
[[37  0  6  3]
 [ 0 33 14  0]
 [ 0  2 49  2]
 [ 0  0  5 35]]
0.865591397849
[[38  0  4  4]
 [ 1 41  5  0]
 [ 0  3 47  3]
 [ 1  0  4 35]]
[(0.0, 1441), (2.0, 1441), (3.0, 1441), (5.0, 1441)]
0.811827956989
[[37  0  6  3]
 [ 0 36 11  0]
 [ 0  3 48  2]
 [ 5  0  5 30]]
```
### 3.3 将1和4类单独训练
不能忘了提出来的类
```python
# 提取出剩余2个类别,1(森林)和4(草地)
x_temp = np.array([d_train[i] for i in range(d_train.shape[0]) if d_train[i][-1] == 1 or d_train[i][-1] == 4])
x_train, y_train = split_data_label(x_temp)
x_temp = np.array([d_test[i] for i in range(d_test.shape[0]) if d_test[i][-1] == 1 or d_test[i][-1] == 4])
x_test, y_test = split_data_label(x_temp)

cls = GaussianNB()

y_pred = cls.fit(x_train, y_train).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))

# 集成

bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier())

y_pred = bbc.fit(x_train, y_train).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))

# 结合降采样和过采样

X_resampled, y_resampled = SMOTE().fit_sample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))

cls = GaussianNB()

y_pred = cls.fit(X_resampled, y_resampled).predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
print(metrics.confusion_matrix(y_test, y_pred))
```
结果:可看到这样的结果比混合训练要好.
```
0.798245614035
[[69  9]
 [14 22]]
0.736842105263
[[56 22]
 [ 8 28]]
[(1.0, 7431), (4.0, 7431)]
0.780701754386
[[65 13]
 [12 24]]
```

## 4.结果对比



# 参考文献
[http://scikit-learn.org/stable/modules/naive_bayes.html](http://scikit-learn.org/stable/modules/naive_bayes.html)

[https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py](https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py)


