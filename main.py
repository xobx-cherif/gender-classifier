import numpy as np

#importing SKlearn models
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC

#importing accuracy score
from sklearn.metrics import accuracy_score

#descisin tree calassifier
treeClf = tree.DecisionTreeClassifier()
#kNN classifier
knnClf = KNeighborsClassifier(n_neighbors=3)
#SVM classifier
svmClf = svm.SVC()
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
#Training data set
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#fiting the models

treeClf = treeClf.fit(X, Y)
knnClf = knnClf.fit(X, Y)
svmClf = svmClf.fit(X, Y)

prediction = []
prediction.append(treeClf.predict([[190, 70, 43]]))
prediction.append(knnClf.predict([[190, 70, 43]]))
prediction.append(svmClf.predict([[190, 70, 43]]))

print(prediction)

#comparing the accuracy of the 3 classifiers


treeClfPred = treeClf.predict(X)
treeAcc = accuracy_score(Y, treeClfPred) * 100
print('Accuracy for descision tree: {}'.format(treeAcc))


knnPred = knnClf.predict(X)
knnAcc = accuracy_score(Y, knnPred) * 100
print('Accuracy for KNN: {}'.format(knnAcc))

svmPred = svmClf.predict(X)
svmAcc = accuracy_score(Y, svmPred) * 100
print('Accuracy for SVM: {}'.format(svmAcc))

# The best classifier from descisin tree, svm, KNN
index = np.argmax([treeAcc, knnAcc, svmAcc])
classifiers = {0: 'Descision tree', 1: 'KNN', 2: 'SVM'}
print('Best gender classifier is {}'.format(classifiers[index]))
