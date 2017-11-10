from sklearn import tree

# smooth = 1, bumpy = 0
features = [[140, 1],
            [130, 1],
            [150, 0],
            [170, 0]]

# apple = 0, orange = 1
labels = [0,
          0,
          1,
          1]

# 비어있는 규칙 박스
clf = tree.DecisionTreeClassifier()

# 지도 학습 알고리즘 생성
# fit = 데이터에서 패턴을 발견하다
clf = clf.fit(features, labels)

print (clf.predict([[150, 0]]))