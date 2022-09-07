# python_ML_basic
# 머신러닝

<br/>

### 딥러닝
- 인공신경망 기반의 모델
- 비정형 데이터로부터 특징 추출 및 판단까지 기계가 수행

<br/>
<br/>
<br/>

### 머신러닝
- 명시적으로 규칙을 프로그래밍하지 않고 데이터로부터 의사결정을 위한 패턴을 기계가 스스로 학습
- 전통적 방식의 기계학습 
> 명시적 규칙 : if/else문
> 
> 머신러닝 알고리즘은 데이터로부터 규칙을 생성
> 
> 과적합을 방지하는 것이 중요한 과제, 일반화하는 것이 중요함
- 학습 알고리즘(머신러닝 알고리즘)을 이용해 데이터에 숨겨진 규칙을 기계 스스로 습득하게 하고, 그 결과를 이용해 새로운 것을 예측하는 기술


<br/>

#### 1) 머신러닝 사용 할 경우 
- 데이터를 구할 수 있어야 함, 데이터가 가장 중요
- 데이터에 규칙이 있다고 판단될 때
- 기존의 명시적 규칙으로 해결 방법을 찾을 수 없는 복잡한 문제의 경우

<br/>

#### 2) 데이터 형식
1. 정형 데이터 - 머신러닝에서 주로 사용
- 데이터 프레임, 스프레드시트, csv 등의 표 형식

2. 반정형 데이터
- xml, html 등

3. 비정형 데이터
- 이미지, 동영상, 음성, 텍스트 등


<br/>

#### 3) 머신러닝 용어
- feature : 데이터 프레임에서 column, feature의 수 = 차원의 수
- instance : 데이터 row 하나
- 독립변수 : 종속변수를 제외한 feature들
- 종속변수 : 회귀에서는 타깃값 / 분류에서는 레이블 or 클래스

<br/>

#### 4) 데이터의 종류
- **수치 데이터(양적 데이터)**
- > 연속형 데이터(continuous) : 키, 몸무게, 시간...
- > 이산형 데이터(discreate) : 불량품 수, 판매 수량...

- **범주형 데이터(질적 데이터)**
- > 순위형 데이터(ordinal) : 등급(1등급,2등급..) - 우위가 있음
- > 명목형 데이터(nominal) : 성별(남/여), 지역(서울/부산/인천)


<br/>

#### 5) 지도학습 / 비지도학습
- **지도학습** : 학습 알고리즘에 주입하는 훈련 데이터에 타깃값(레이블)이라는 원하는 답이 포함되어 있는 경우
- 타깃값을 만드는 시간과 비용이 많이 든다.
> 분류(Classification) : Y변수가 범주형 일 때, 범주 예측
> 
> 회귀(Regression) : Y변수가 연속형 일 때, 수치 예측
  
- **비지도학습** : 훈련 데이터에 타깃값이 없는 학습 방법
> 군집(Clustering) : 타깃변수가 없고 특성이 비슷한 데이터들로 묶임
> 
> 분류는 정답이 있기 때문에 데이터를 나누는 선을 그리는 것을 목표로 하지만 군집은 그룹으로 묶는 것을 목표로 함


<br/>
<br/>
<br/>




### 머신러닝 학습
#### 데이터 나누기
- 학습 완료 모델의 일반화 오차를 추정하기 위해 사용
- 새로운 외부 샘플에 대해 어느 정도 오차가 있는지 추정
- 학습 데이터 Trian Data
- 테스트 데이터 Test Data
- 검증 데이터 Validation Data : 학습 중인 현재 모델의 성능을 평가하기 위해 사용 / 학습 데이터의 일부


<br/>
<br/>

#### 교차 검증
![교차검증](https://postfiles.pstatic.net/MjAxOTA3MjVfMTYw/MDAxNTY0MDYxOTQxODg2.2SJCkdADPvofL7LceWnSthfefB3UvnQ2_YoRp5F2vFog.4EZrViOF41rKfovPOJJMyv7W2HKTEvfDyg92pwIIIJ4g.PNG.ckdgus1433/image.png?type=w773)
- 적합한 모델을 선택하기 위한 방법
- 학습 세트를 여러 서브셋으로 나누고 모델을 각 서브셋의 조합으로 훈련, 검증
- 학습이 끝나면 선택된 모델을 전체 학습 세트로 학습
- 어느 검증 데이터에 과적합되는 것 방지


<br/>

#### 인코딩
- 라벨 인코딩 : 문자로 표현된 범주형 데이터를 숫자로 변환
- 원핫 인코딩 : 숫자가 크면 영향력이 커질 수 있기 때문에 feature를 늘리고 0/1로 된 데이터를 넣어줌


<br/>

#### 특성 스케일링
- 데이터의 범위를 비슷하게 맞춰주는 작업

1) 표준화
   - 평균이 0, 분산이 1이 되도록 변환

2) Min-Max 스케일링
   - 최댓값이 1, 최솟값이 0, 나머지는 0~1 사이의 값으로 변환


<br/>

#### 불균형 데이터
- 샘플링 기법
- > 오버 샘플링 : 소수의 클래스의 샘플을 늘림
- > 언더 샘플링 : 다수 클래스의 샘플을 줄임


<br/>
<br/>
<br/>


### 머신러닝 성능 평가
#### Cunfusion Matrix
![cm](https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/w1200-h630-p-k-no-nu/confusionMatrxiUpdated.jpg)
- TP : 양성으로 예측, 정답도 양성
- FP : 양성으로 예측, 정답은 음성
- TN : 음성으로 예측, 정답도 음성
- FN : 음성으로 예측, 정답은 양성

- 정밀도(Precision)와 재현율(Sensitivity)은 Trade Off 관계

<br/>

#### F1 Score
- 정밀도와 재현율이 한 쪽으로 편향되지 않도록 추가 보완하는 지표
![f1](https://images.velog.io/images/jadon/post/f06f1d40-605d-4f13-b6ce-35c220c82968/image.png)

- 한 쪽이 아무리 높아도 나머지 한 쪽이 낮으면 f1 score는 높은 점수를 받을 수 없음

<br/>

#### 분류 모델 평가(ROC)
Threshold와 정확도
- 로지스틱 회귀는 모델이 각 클래스에 속할 확률을 출력
- 출력된 확률값에서 threshold값을 비교하여 결정
- Threshold의 값을 변화시키면서 TPR과 FPR의 값을 보며 성능 평가
![roc](https://miro.medium.com/max/832/0*-dBz1HonBn39H0xc.png)


<br/>


#### 회귀 모델 평가
- Error : 실제값과 예측값의 차이
- 오차를 최소가 되도록 모델을 최적화하는 것이 회귀 알고리즘
- 평균제곱오차(MSE)
> MSE : Mean Squared Error
> 
> RMSE : Root Mean Squared Error
> 
> MAE : Mean Absolute Error

<br/>

#### 결정계수(R Square)
- 독립변수가 종속변수를 어느정도 설명하는지 나타내는 지표

<br/>

### Hyper Parameter 튜닝
- grid search : 모든 범위를 격자별로 최적의 값을 찾음
- random search : 랜덤한 범위에서 최적의 값을 찾음
- bayesian optimization : 하이퍼파라미터의 범위를 지정하고, 랜덤하게 R번 탐색하고 B번 만큼 최적의 값을 찾음

<br/>
<br/>
<br/>

# 머신러닝 모델

## 분류 알고리즘
<br/>

### 1. K-NN
- 과거의 데이터를 기억하고 새로운 데이터에 대해서 가장 가까운 유사도(거리)를 가진 기존 데이터에 따라서 분류
- 비선형 모델
- Y가 범주형일 때 사용
- Y가 연속형인 경우 가장 가까운 데이터의 평균으로 새로운 데이터를 예측(평균이 가장 정확) / 분류 알고리즘으로 회귀 문제도 풀 수 있다.
- k가 1인 경우는 과대적합 발생
- k가 클 수록 결정경계가 완만해지지만 과소적합 발생

<br/>

#### hyper parameter
- K값
> k값 정하기 : 학습 error와 테스트 error가 최소가 되는 k값을 찾아야 함
- 거리 측정

<br/>

> 거리 계산(두 점 사이 거리)
> - 유클리드 거리 : 일반적으로 사용하는 피타고라스의 거리와 같은 거리 측정법
> - 맨하탄 거리 : 삼각형에서 ㄱ에 해당하는 거리를 더하는 측정법
> - Norm(놈) : 벡터의 길이, 크기를 측정하는 방법


<br/>
<br/>
<br/>

### 2. Support Vector Machine(SVM)
![svm](https://miro.medium.com/max/609/0*lyr5-f7HRu34OLvd.png)


- 머신러닝에서 가장 인기있는 모델
- 복잡한 분류 문제에 적합
- 선형, 비선형 분류 모두 사용 가능
- 두 개의 클래스를 분류하는 수 많은 직선 중 가장 적절한 선을 찾아냄
- margin을 최대로 하는 선을 찾는 것이 목표


![서포트벡터](https://blog.kakaocdn.net/dn/JyfbT/btqEqtpxbch/flfwGbM7mgv1kP1kkn4nQK/img.png)

**소프트 마진 분류**
- 소프트 마진 : 어느정도의 에러를 고려함

<br/>

#### 하이퍼 파라미터
- C : 마진 사이에 오류를 얼마나 허용할 것인지 규제, 클수록 하드마진, 작을 수록 소프트마진에 가까움
- gamma : 결정경계를 얼마나 유연하게 그릴지 결정 클수록 오버피팅 발생 가능성이 높아짐(커널이 rbf)
- degree : 다항식 커널의 차수 결정(커널이 poly일 때 사용)
- kernel


![커널 트릭](https://www.sallys.space/image/svm/2.png)
### 비선형 SVM분류 : 커널 트릭
- 2차원으로 분류가 불가능할 때 커널을 이용해 다차원으로 바꿔서 분류


<br/>
<br/>
<br/>

### 3. 의사 결정 나무(Decision Tree)
- 불순도 : gini계수나 엔트로피로 계산
- 불순도를 낮추는 방향으로 알고리즘 생성
- 정보 획득량 : 하나의 질문에 의해 감소되는 불순도의 정도

<br/>

- 규제 : 과적합을 막기 위해 규제 추가
> - Max_depth : 깊이 제한
> - Min_sample_split : 분기를 하기위한 최소한의 샘플수
> - Min_sample_leaf : 리프 노드가 가지고 있어야할 최소한의 샘플수
> - Max_leaf_nodes : 리프 노드의 최대수


#### feature importance
- 트리를 사용하는 알고리즘에서 알 수 있음


<br/>
<br/>
<br/>


### 4. 나이브베이즈
![베이즈정리](https://hleecaster.com/wp-content/uploads/2020/01/nbc01.png)
- 베이즈 정리 : 사건 B가 발생함으로써 사건 A의 확률이 어떻게 변하는지 표현한 정리
- 나이브베이즈 : 각 사건을 독립 사건으로 가정하고 베이즈 정리를 적용

<br/>
<br/>
<br/>


### 5. 앙상블(Ensemble)
약한 모델 여러 개를 조합하여 더 정확한 예측에 도움을 주는 방식
- 보팅 : 각 알고리즘 중 성능 좋은 한 가지 선택하는 방식
- 배깅 : 같은 알고리즘으로 다양한 옵션의 모델을 동시에 여러 개 만드는 방식 / 의사 결정 나무로 배깅을 하는 게 **랜덤 포레스트**
- 부스팅 : 모델을 생성할 때 이전 모델의 결과를 확인하고 만드는 방식, 차례대로 모델 생성 / XG부스트의 방식
> - 아다부스트 : 앞의 모델이 예측하지 못한 데이터의 가중치를 높여서 다음 모델 학습
> 
> - 그래디언트 부스팅 : 경사하강법 사용, 앞의 분류기에서의 오차를 줄이는 방향으로 학습

<br/>

#### 그래디언트 부스트 라이브러리
- XGBoost
- LightGBM : 트리의 양 쪽의 균형을 고려하지 않아 속도가 빠름, 나머지는 균형 고려
- CatBoost : 범주형 데이터에 사용
- 그래디언트 부스팅은 결정 나무 기반의 모델
  

<br/>
<br/>
<br/>

## 선형 회귀 알고리즘



- 독립변수와 종속변수를 선형적인 관계로 가정하고 데이터를 가장 잘 나타내는 선형식을 찾음
- 학습에 종속변수가 필요한 지도학습
- 데이터를 가장 잘 나타낸 직선 : 가설(예측)과 실제값과의 차이(오차)가 최소가 되는 직선

<br/>

#### 오차

1. 잔차 제곱합(RSS) 
   
   ![rss](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRjJfRq7btntaS_hQSZBWj6_tIqN-s0B0U-A&usqp=CAU)
- 오차 제곱의 합

<br/>

2. 평균 제곱 오차(MSE)
   ![mse](https://t2.daumcdn.net/thumb/R720x0.fpng/?fname=http://t1.daumcdn.net/brunch/service/user/bhYD/image/ucsieuHiSax6uYbuNeOLCvyivp4.png)
- 오차 제곱의 평균


<br/>

#### 손실함수
   - 비용 함수라고 함
   - 정답과 예측의 차이(Loss)를 계산하는 함수
   - 회귀 알고리즘은 손실함수가 최소가 되는 파라미터를 찾는 게 목표




#### 최소 제곱법(OLS)
   - RSS의 미분값이 0이 되는 점을 B값으로 하여 B값들을 구함

<br/>

#### 경사 하강법(Gradient Descent)
- 목적 : 손실함수가 0이 되는 점을 찾는 것
- 여러 종류의 문제에서 최적의 해법을 찾을 수 있는 최적화 알고리즘
- 비용함수를 최소화하기 위해 반복하여 파라미터 조정
- 기울기를 따라 반복해서 내려감


![경사하강법](https://mblogthumb-phinf.pstatic.net/MjAyMDAzMTZfMjk3/MDAxNTg0MzAxMTIyNjU2.WKCjPeQRON1vq88FQTp78CRJy_kDSN_WQULTNbaGOOcg.kNy6inVKXc5242XKO77MeArOeaxec2UJ6Y1SRh7Qf8wg.PNG.jevida/031520_1938_Gradient2.png?type=w800)

1. 임의 B값에서 시작
2. 그 점에서 경사를 구하기 위해 미분
3. 기울기의 반대방향으로 현재의 B에서 기울기의 크기에 비례한 만큼 빼줌(* 기울기가 양수 - 왼쪽으로 B값 이동)
> B값이 작아지는 정도 : 학습률(Learning Rate)

<br/>

#### 경사 하강법 종류
- 배치 경사 하강법 : 전체 학습 데이터를 하나의 배치로 묶어 학습시키는 경사 하강법(배치 크기 : n)
- 확률적 경사 하강법 : 하나의 데이터를 이용하여 경사 하강법을 1회 진행(배치 크기 : 1)
- 미니 배치 경사 하강법(MSGD) : 전체 데이터를 배치 크기개씩 나눠 배치로 학습(배치 크기 : 사용자 지정)




![경사하강법](https://user-images.githubusercontent.com/86597163/180908214-4b880f4c-9fa0-4d6c-a203-5dbf63bfb544.png)




<br/>
<br/>




#### 규제
- 고차 다항 회귀를 적용하면 과대 적합을 일으키기 쉬움
- 과대 적합을 일으키는 이유
  > 모델 자체가 복잡
  >
  > 학습하기에 데이터가 적다
- 모델의 규제를 통해 다항식의 차수를 줄이는 것으로 과대적합을 줄일 수 있음

1. 릿지 규제
2. 라소 규제


<br/>
<br/>

### 1. 로지스틱 회귀
- 회귀를 사용하는 분류에 사용
- 이진 분류(스팸 필터, 시험 합격 여부...)

![로지스틱](https://t1.daumcdn.net/cfile/tistory/99F325485C7B76BC2B)


- 시그모이드 함수
  > 로지스틱 함수라고도 함
  >
  > ![image](https://user-images.githubusercontent.com/86597163/181146896-c024c5ec-9e15-4ccb-8f87-a83295a50ca2.png)

- cost function 종류
  > cross entropy : 두 확률분포 사이의 오차 측정(binary)


#### 다중 분류
- 이진분류와 cost function이 다름
- softmax 사용
  > 기존 시그모이드 함수를 사용하면 각 입력값들의 결과 합이 1을 초과하는 문제 발생
  >
  > 다중 클래스 분류일 때 softmax 사용(결과 합 : 1)



<br/>
<br/>

## 비지도 학습
### Clustering(군집화)
### 1. K-Means Clustering
![image](https://user-images.githubusercontent.com/86597163/181167226-47baa614-d001-445d-813d-45b09889c706.png)
![image](https://user-images.githubusercontent.com/86597163/181167260-067631fc-06f3-41d8-ba1e-645d469c1086.png)
![image](https://user-images.githubusercontent.com/86597163/181167420-437ec7cb-e2fa-4a2f-ac2a-fefc42efeda8.png)
![image](https://user-images.githubusercontent.com/86597163/181167460-d47e92de-8fe9-4a48-928a-a2a1da287373.png)

[k-means clustering](https://bskyvision.com/entry/k-means-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0%EB%A7%81)


#### SSE
- 적당한 k를 선택하는 법
- 관측치와 중심들 사이의 거리
- 군집이 얼만큼 뭉쳐있는지 나타내는 지표


### 2. 차원 축소
- 차원 : 공간에서 데이터의 위치를 나타내기 위해 필요한 축(axis)의 수
- 차원의 저주 : 차원이 증가하면 정보의 밀도 감소, 정보의 감소로 과적합 문제 발생
- 차원문제의 해결 방법 : 차원 축소

#### 투영(Projection)
- 차원 축소 알고리즘의 주요 접근법 중 하나
- 주성분 분석(PCA)
  > - 차원 축소 알고리즘
  >
  > - 분산이 최대로 되는 축 선택, 축에 대해 투영
