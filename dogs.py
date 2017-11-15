import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

# 랜덤 +- 4인치
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# 그래프 시각화
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

# 유용하지만 완벽한 정보가 아닐 경우 > 많은 특징이 필요한 이유
# 필요없는 특징을 추가할 경우 분류기준의 정확성을 떨어트림
# 연관성이 없는 특징을 사용해야 할 것
# 특징 분류를 절대로 운에 맡기지 말 것