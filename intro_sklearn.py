from sklearn import preprocessing
# change raw feature vectors in suitable format
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, Y_test = train_test_split(X,y, test_size=0.33)

from sklearn import svm
#build a classifier using Support Vector classification algorithm
clf = svm.SVC(gamma=0.0001, C=100.)
# pass the training set and clf learns to classify unknown cases 
clf.fit(X.train, y.train)
# run predictions
clf.predict(X_test)

#evaluate your model accuracy
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, yhat, labells=[1,0))
