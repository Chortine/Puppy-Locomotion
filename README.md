## 介绍


## 设备
WaveGo四足小狗

## 方法
### 仿真训练
在这里我们用isaac gym作为仿真环境，训练代码基于开源项目legged_gym修改。训练算法使用连续动作PPO。
仿真里关节pd的定义和实际中pid控制的pd略有不同。仿真里的pd是为了计算最终作用到关节上的力矩，公式为：$$torque = p (\theta_{target} - \theta_{current}) - d \dot{\theta}$$。可以说，仿真里的pd是用来计算从当前关节状态到力矩的映射。
而实际中狗的舵机没有pd参数，它的最终作用力矩也是一系列复杂的机电结构导致的结果，其映射关系不符合上面的公式。这将是虚实迁移中一个比较大的gap。

### 细节


## 效果视频
| 光滑桌面  | 弹性表面 |
| ------------- | ------------- |
| <video src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/7d432b00-0fe5-4708-b102-10568e7e2b5d">  | <video src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/6b2e3eef-1445-47b2-9b3c-6167f3147ee7">|

| 光滑地面  | 柔软表面 |
| ------------- | ------------- |
| <video src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/951fb168-5261-4f72-9f1a-d5551a720602">  | <video src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/6933d48d-ca22-4ae6-96bb-05a455067802">|
  

#### 带RMA
  
| 光滑地面  | 柔软表面 |
| ------------- | ------------- |
| <video src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/e1050dd7-97f5-4d01-89a7-7043bc04d4c7">  | <video src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/5ce325f1-0723-4e14-8ddb-92ae9d711cf5">|
  
<!-- https://github.com/Chortine/Puppy-Locomotion/assets/107395103/e1050dd7-97f5-4d01-89a7-7043bc04d4c7 -->

<!-- 

https://github.com/Chortine/Puppy-Locomotion/assets/107395103/6933d48d-ca22-4ae6-96bb-05a455067802


https://github.com/Chortine/Puppy-Locomotion/assets/107395103/951fb168-5261-4f72-9f1a-d5551a720602 -->



<!-- https://github.com/Chortine/Puppy-Locomotion/assets/107395103/5ce325f1-0723-4e14-8ddb-92ae9d711cf5 -->

