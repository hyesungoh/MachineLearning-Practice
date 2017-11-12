import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# 꽃의 분류 값
# print(iris.feature_names)

# 꽃의 종류 값
# print(iris.target_names)

# 0번째 꽃의 분류 값
# print(iris.data[0])

# 0번째 꽃의 종류 값
# print(iris.target[0])

# 150개 꽃들의 정보
# for i in range(len(iris.target)) :
#     print("Ex %s : label %s : features %s" %(i, iris.target[i], iris.data[i]))

test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print ("정답 : ", test_target)
print ("결정 트리의 값 : ", clf.predict(test_data))


# 시각화
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")