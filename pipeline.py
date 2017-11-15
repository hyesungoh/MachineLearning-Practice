from sklearn import datasets
iris = datasets.load_iris()

# f(x) = y
X = iris.data
Y = iris.target

# 데이터를 train과 test로 반반씩 (.5) 나눔
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= .5)

# 의사결정 트리
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# KNN 알고리즘 사용
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, Y_train)

# 예측 결과
predictions = my_classifier.predict(X_test)
# print(predictions)

# 예측 결과 정답의 확률
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))