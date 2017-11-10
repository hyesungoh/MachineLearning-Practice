from sklearn import tree

# features[0] = 마력, features[1] = 좌석 수
features = [[350, 2],
            [400, 2],
            [380, 2],
            [200, 8],
            [150, 6],
            [250, 10]
            ]

# 1 = 스포츠카, 2 = 미니벤
labels = [1,
          1,
          1,
          2,
          2,
          2]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print("400마력, 시트 2개 : ", clf.predict([[400, 2]]))
print("400마력, 시트 8개 : ", clf.predict([[400, 8]]))
print("200마력, 시트 2개 : ", clf.predict([[200, 2]]))
print("200마력, 시트 6개 : ", clf.predict([[200, 8]]))
