# SVM을 이용한 이미지 분류
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

faces = fetch_lfw_people(min_faces_per_person=60, color=False, resize=0.5)
# min_faces_per_person : 한 사람당 설정 개수 이상의 사진이 있는 자료만 사용
# print(faces)
print(faces.DESCR)
print(faces.data)
print(faces.data.shape)
print(faces.target)
print(faces.target.shape)
print(faces.target_names)
print(faces.images.shape)

# # 이미지 시각화
# plt.imshow(faces.images[1], cmap='bone')
# plt.title('{}'.format(faces.target_names[faces.target[1]]))
# plt.show()

# fig, ax = plt.subplots(3, 5)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap='bone')
#     axi.set(xticks=[], yticks=[], xlabel='{}'.format(faces.target_names[faces.target[i]]))
# plt.show()

# 설명력 95% 되는 최소 개수를 얻기
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(faces.data)
print(pca.n_components_)    # 184


# 주성분 분석으로 이미지 차원을 축소시켜 분류작업 진행
n = 150     # 축소 시킬 차원수
m_pca = PCA(n_components=n, whiten=True, random_state=0)
# whiten=True 주성분의 scale이 작아지도록 조정
x_low = m_pca.fit_transform(faces.data)

print('x_low : ', x_low, ' ', x_low.shape)
fig, ax = plt.subplots(3, 5, figsize=(10,8))
for i, axi in enumerate(ax.flat):
    axi.imshow(m_pca.components_[i].reshape(faces.images[0].shape), cmap='bone')
    axi.axis('off')
    axi.set_title(f'PCA {i+1}')
plt.suptitle('Eigenfaces(주성분 얼굴)', fontsize=12)
plt.tight_layout()
# plt.show()      
# SVM 알고리즘은 얼굴의 특징 패턴으로 분류 작업을 함

print('누적 설명력 : ',m_pca.explained_variance_ratio_.sum())

# 원본 vs 복원 이미지 비교
x_reconst = m_pca.inverse_transform(x_low)
fig, ax = plt.subplots(2,5,figsize=(10,4))
for i in range(5):
    # 원본
    ax[0,i].imshow(faces.images[i], cmap='bone')
    ax[0,i].set_title('원본')
    ax[0,i].axis('off')

    # 복원
    ax[1,i].imshow(
        x_reconst[i].reshape(faces.images[0].shape), cmap='bone'
    )
    ax[1,i].set_title('복원본')
    ax[1,i].axis('off')
plt.suptitle('PCA 복원 비교', fontsize = 12)
plt.tight_layout()
plt.show()

# 분류 모델 생성
svcmodel = SVC(C=1.0, random_state=1)
mymodel = make_pipeline(m_pca, svcmodel)    # PCA와 분류기를 하나의 파이프 라인으로 묶어 순차적으로 실행
print('mymodel : ', mymodel)
# mymodel :  Pipeline(steps=[('pca', PCA(n_components=100, random_state=0, whiten=True)), ('svc', SVC(random_state=1))])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=1, stratify=faces.target)
# stratify 불균형 자료 완화
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

mymodel.fit(x_train, y_train)
pred = mymodel.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10])

# 정확도
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confmat = confusion_matrix(y_test, pred)
print('confusion_matrix :\n',confmat)
print('정확도 : ', accuracy_score(y_test, pred))
print('classification_report :\n', classification_report(y_test, pred, target_names=faces.target_names))

# 오차 행렬 시각화
import seaborn as sns
plt.figure()
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues', xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('예측')
plt.ylabel('실제')
plt.title('confusion_matrix')
plt.show()

# PCA 누적 분산 그래프
import numpy as np
plt.plot(np.cumsum(m_pca.explained_variance_ratio_))
plt.xlabel('주성분 개수')
plt.ylabel('누적 설명력')
plt.title('PCA 설명력')
plt.grid(True)
plt.show()


# 새로운 이미지로 분류해보기

# 실습 1) 기존 데이터 테스트
test_img = faces.data[0].reshape(1,-1)  # (1,2914)
print('test_img : ', test_img)
test_pred = mymodel.predict(test_img)
print('실습1 예측 결과 : ', faces.target_names[test_pred[0]])
print('실습1 실제값 : ', faces.target_names[faces.target[0]])
# 실습1 예측 결과 :  Colin Powell
# 실습1 실제값 :  Colin Powell

# 실습 2) 신규 데이터 테스트
# 이미지 읽기 -> 흑백변환 -> 크기 맞추기(62 X 47) -> 1차원으로 변환 -> predict
from PIL import Image
img = Image.open('Colin Powell.jpg')
img.convert('L')        # 흑백으로 변환
img.resize((47,62))     # width, height 크기 맞추기
# numpy 이미지는 꼴이 (height, width) 지만, PIL 이미지 꼴은 (width, height)
# 이미지는 라이브러리 마다 축 순서가 다른 경우가 많음
img_np = np.array(img)      # numpy 변환
# print(img_np)       # 0-255 숫자 데이터로 구성됨 -> 정규화 필요
img_np /= 255.0         # 정규화
img_flat = img_np.reshape(1,-1) # 1차원으로 변환
new_pred = mymodel.predict(img_flat)
print('실습2 예측 결과 : ', faces.target_names[new_pred[0]])
print('실습2 실제값 : ', faces.target_names[faces.target[0]])

# 시각화 + 예측
plt.imshow(img_np, cmap='bone')
plt.title(f'예측 : {faces.target_name[new_pred[0]]}')
plt.axis('off')
plt.show() 
# 보다 정확한 분류를 위해서는 밝기/위치 정렬 등의 작업 필요.

