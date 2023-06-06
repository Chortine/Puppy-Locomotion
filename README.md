## 介绍
四足机械小狗行走算法的训练和迁移

## 硬件
WaveGo四足小狗，每条腿有3个自由度

<a href="https://www.bilibili.com/video/BV1Eh4y1475R/?spm_id_from=333.999.0.0"><img height="216" src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/d3a4e92e-e5ec-4939-a282-3507bfc1f345" width="380"/></a>
<!-- ![puppy_legs](https://github.com/Chortine/Puppy-Locomotion/assets/107395103/d3a4e92e-e5ec-4939-a282-3507bfc1f345) -->

树莓派
ESP32下位机

## 方法
Policy输入：  
输出：  
算法：连续动作PPO。 

<a href="https://www.bilibili.com/video/BV1Eh4y1475R/?spm_id_from=333.999.0.0"><img height="216" src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/95777fc2-8796-4766-a631-8cc3e0469348" width="380"/></a>

<!-- ![image](https://github.com/Chortine/Puppy-Locomotion/assets/107395103/95777fc2-8796-4766-a631-8cc3e0469348) -->

在这里我们用isaac gym作为仿真环境，训练代码基于Nvidia的开源四足训练仓库legged_gym[1]修改。训练算法使用连续动作PPO。
仿真里关节pd的定义和实际中pid控制的pd略有不同。仿真里的pd是为了计算最终作用到关节上的力矩，公式为：
![4](http://latex.codecogs.com/svg.latex?torque=p(\theta_{target}-\theta_{current})-d\dot{\theta})
可以说，仿真里的pd是用来计算从当前关节状态到力矩的映射。
而实际中狗的舵机没有pd参数，它的最终作用力矩也是一系列复杂的机电结构导致的结果，其映射关系不符合上面的公式。这将是虚实迁移中一个比较大的gap。
#### 获得仿真中的行走baseline
  四足由于动作空间较大，对环境参数敏感，往往需要较多的工作来获得一个在仿真中
  1. 环境参数设定  
    
    重要参数有：  
      1. 关节初始角度和关节PD参数，设定原则：让action为0的时候、狗能平稳地直立在原地（如果PD太大会出现抖动，如果太小则狗会倒伏）
      2. 关节限定值： 根据世纪舵机的参数来设定，实际关节的最大力矩取是0.23N*m，关节的最大转速是60deg/0.1s.  
      3. action_scale: 最终取的是0.25，大于0.3训不出来
      4. 仿真dt: 对于仿真效果来说，dt越小越准确，但是小了会影响仿真的效率。 
         选择的方法是让狗直立在原地，此时各关节转速读数应该接近0，然后慢慢调大dt，使得
      
    除了以上经验调节方法，还能用CEM等进化算法大规模搜索最优参数。[3]
  3. Reward Shaping  
  
    重要的reward项有：
      1. 
    
#### 获得更自然的步态
AMP等方法

#### 虚实迁移



<!-- https://github.com/Chortine/Puppy-Locomotion/assets/107395103/73818b14-cadd-41c0-b4f4-f5e97af5e903 -->


  1. 随机环境参数
  1. RMA[2]
  2. 仿真PD对齐实际

#### 
### 重要细节
<!-- ![puppy_origin](https://github.com/Chortine/Puppy-Locomotion/assets/107395103/8d0ad096-e82c-46f8-9d15-76c84ebff9bc)
![puppy_modified](https://github.com/Chortine/Puppy-Locomotion/assets/107395103/b04f3160-1fb8-45d8-8f62-e6c9094c4427) -->
虚实迁移的时候
![4](http://latex.codecogs.com/svg.latex?\sum_{n=1}^\infty\frac{1}{n^2}=\frac{\pi^2}{6})


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

## 参考
[1] legged_gym
[2] RMA
[3] Flood
