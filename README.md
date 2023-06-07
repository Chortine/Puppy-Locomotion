## 介绍
四足机械小狗行走算法的训练和迁移。作为一个完成度较高的虚实迁移项目，下文介绍了项目的全流程，以及最终成果。

## 硬件
WaveGo四足小狗，每条腿有3个自由度，共12个自由度。

<a href="https://www.bilibili.com/video/BV1Eh4y1475R/?spm_id_from=333.999.0.0"><img height="216" src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/d3a4e92e-e5ec-4939-a282-3507bfc1f345" width="340"/></a>
<!-- ![puppy_legs](https://github.com/Chortine/Puppy-Locomotion/assets/107395103/d3a4e92e-e5ec-4939-a282-3507bfc1f345) -->

树莓派
ESP32下位机

## 方法
Policy输入：  
由于目标是迁移到实际的四足上，因此输入受限于实际的硬件传感器能读的值。主要有：机身的IMU数据(线加速度和角度)，
输出： 12个
算法：连续动作PPO。 



<!-- ![image](https://github.com/Chortine/Puppy-Locomotion/assets/107395103/95777fc2-8796-4766-a631-8cc3e0469348) -->

在这里我们用isaac gym作为仿真环境，
<a href="https://www.bilibili.com/video/BV1Eh4y1475R/?spm_id_from=333.999.0.0"><img height="216" src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/95777fc2-8796-4766-a631-8cc3e0469348" width="380"/></a>
训练代码基于Nvidia的开源四足训练仓库legged_gym[1]修改。训练算法使用连续动作PPO。
仿真里关节pd的定义和实际中pid控制的pd略有不同。仿真里的pd是为了计算最终作用到关节上的力矩，公式为：
![4](http://latex.codecogs.com/svg.latex?torque=p(\theta_{target}-\theta_{current})-d\dot{\theta})
可以说，仿真里的pd是用来计算从当前关节状态到力矩的映射。

#### 获得仿真中的行走baseline
  四足由于动作空间较大，对环境参数敏感，往往需要较多的工作来获得一个在仿真中可以行走的Baseline。之后再基于此进行调优。即便Legged_Gym[1]仓库提供了基于诸如unitree和anymal等大型四足狗的训练样，但直接复制同样的参数会导致在小狗上训练失败。以下是我们作的一些工作：  
  1. 环境参数设定  
    
    重要参数有：  
      1. 关节初始角度和关节PD参数，设定原则：让action为0的时候、狗能平稳地直立在原地（如果PD太大会出现抖动，如果太小则狗会倒伏）
      2. 关节限定值： 根据世纪舵机的参数来设定，实际关节的最大力矩取是0.23N*m，关节的最大转速是60deg/0.1s.  
      3. action_scale: 最终取的是0.25，大于0.3训不出来
      4. 仿真dt: 对于仿真效果来说，dt越小越仿真精度越高，但是仿真效率越低。
         选择的方法是让狗直立在原地，此时各关节转速读数应该接近0，然后慢慢调大dt，使得关节角速度的指出现异常跳变为止。  
         这时的dt就是可取的dt上限。在这之内取个能平衡计算速度和精度的dt值。

  除了以上经验调节方法，还能用CEM等进化算法大规模搜索最优的参数组合。[3]  
  
  2. Reward Shaping  
  
    重要的reward项有：
      1. 
    
#### 获得更自然的步态
AMP等方法

#### 虚实迁移
仿真中的目标转角和
实际中狗的舵机没有pd参数，它的最终作用力矩也是一系列复杂的机电结构导致的结果，其映射关系不符合上面的公式。这将是虚实迁移中一个比较大的gap。


<!-- https://github.com/Chortine/Puppy-Locomotion/assets/107395103/73818b14-cadd-41c0-b4f4-f5e97af5e903 -->

  1. 仿真PD对齐实际  
  对于同样
  3. 随机环境参数
  4. RMA[2]

#### 硬件调优
@ WJ 卡尔曼滤波


### 细节
| 光滑桌面  | 弹性表面 |
| ------------- | ------------- |
| <img src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/70a0d7f9-23b1-47fd-8b4f-eeeac5f73caa" height="200" width="10">  | <img src="https://github.com/Chortine/Puppy-Locomotion/assets/107395103/7c6b0735-e859-466d-be40-eaea77943a27" height="200" width="310">|


虚实迁移的时候


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
