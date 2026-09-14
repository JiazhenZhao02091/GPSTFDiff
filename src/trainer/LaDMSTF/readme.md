## 训练方式

### 在总的时间步中训练一个大的模型

### 分为三段时间步，每个时间步训练不同的去噪网络，同时使用的训练数据集也不相同

每张 image_x0 可以按照Laparoscopic分解为 x1 x2 x3, 
随便取吗？因为U-Net不care输入的形状？


## 采样过程

针对初始噪声epsilon，需要按照拉普拉斯金字塔分解为epsilon1 epsilon2 epsilon3



## 训练流程：
fine_02 == x
            --> x_1
            --> x_2
            --> x_3


Train 01
train D3 == x_3             time interval [0, oo]
    --> random sample
        --> get t --> get alpha_t,sigma_t --> epsilon_t
        epsilon_t --> D3 --> epsilon^ or sample
        --> loss(epsilon^, epsilon_t)

train D2 == (x_2 U x_3)              time interval [0, t_2]  
    --> random sample                  对于D2训练的时候，只使用x_2和x_3，合成x_t的时候，可以直接把x_1忽略掉，噪声也直接取标准高斯分布即可，因为分解后和分解前，并没有衰弱因子的变化
        --> get t --> get alpha_t,sigma_t --> epsilon_t   == (alpha, sigma)
        epsilon_t --> D2 --> epsilon^ or sample
        --> loss(epsilon^, epsilon_t)
train D1 == (x_1 U x_2 U x_3)        time interval [0, t_1] (t1 < t2)
    --> random sample
        --> get t --> get alpha_t,sigma_t --> epsilon_t
        epsilon_t --> D1 --> epsilon^ or sample
        --> loss(epsilon^, epsilon_t)
Train 02
- 训练集范围???
train D == (x_1 U x_2 U x_3)        time interval [0, oo]
    --> random sample
        --> get t --> get alpha_t,sigma_t --> epsilon_t
        epsilon_t --> D --> epsilon^ or sample
        --> loss(epsilon^, epsilon_t)


Inference
inference:
    sample epsilon ~ N(0, I)
    --> lapa Pyramid epsilon --> (epsilon_1, epsilon_2, epsilon_3)
    --> epsilon_3 --> D3 --> x_3 (until time_2)
    x_3 --> upsample --> add epsilon_2 --> D2 --> x_2 (until time_1)
    x_2 --> upsample --> add epsilon_1 --> D1 --> x_1 (final time)