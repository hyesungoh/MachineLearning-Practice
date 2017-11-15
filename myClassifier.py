#import random

from scipy.spatial import distance

# 두 점 사이의 거리 반환 함수 (3차원(특징 3개) 이상도 가능)
def euc(train_point, test_point) :
    return distance.euclidean(train_point, test_point)

# 최근접 이웃 알고리즘 작성
class scrappyKNN() :
    # 데이터 장착
    def fit(self, X_train, Y_train) :
        self.X_train = X_train
        self.Y_train = Y_train

    # 예측
    def predict(self, X_test) :
        predictions = []

        for row in X_test :
            label = self.closest(row)
            predictions.append(label)

        # 랜덤으로 예측
        # for row in X_test :
            # label = random.choice(self.Y_train)
            # predictions.append(label)

        return predictions

    # 제일 가까운 label의 값으로 예측
    def closest(self, row) :
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)) :
            dist = euc(row, X_train[i])
            if (dist < best_dist) :
                best_dist = dist
                best_index = i

        return self.Y_train[best_index]

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

# KNN 알고리즘
# from sklearn.neighbors import KNeighborsClassifier

my_classifier = scrappyKNN()

my_classifier.fit(X_train, Y_train)

# 예측 결과
predictions = my_classifier.predict(X_test)
# print(predictions)

# 예측 결과 정답의 확률
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))