# 机器学习快速入门

## 简介

下面介绍如下内容：

- 机器学习和数据科学
- 数据和问题定义
- 数据收集
- 数据预处理
- 无监督学习
- 监督学习
- 泛化与评估

## 机器学习和数据科学

数据科学家：比软件工程师更擅长统计，比统计学家更擅长软件工程。

**数据科学**采用统计学、计算机等领域的方法从数据中挖掘信息。在实践中，数据科学包括数据收集、清理、分析、可视化和部署。

**机器学习**主要关注数据科学中分析和建模阶段使用的算法和技术。

### 机器学习分类

机器学主要有三种方法：

- 监督学习
- 无监督学习
- 强化学习

监督学习和无监督学习很好理解。

强化学习从一个完全不同的角度来解决学习过程。它假设 agent 通过与动态环境交互以实现特定目标。环境用一组 states 来描述，agent 可以采取不同的 action 从一种状态移动到另一种状态。部分状态被标记为目标（goal），当 agent 达到这个状态，它会得到很大的奖励。而其它状态可能奖励更小、没有奖励或负数奖励。强化学习的目标是找到一个最优策略或映射函数，指定在每个 state 下采取的 action。

强化学习的一个例子是自动假设，其中 state 对应驾驶条件，例如当速度、路段信息、周围交通、速度限制和道路障碍物，对应的 action 为驾驶操作，如左转、右转、停车、加速和继续。学习算法产生一个策略，该策略指定在特定的驾驶条件下要采取的行动。

强化学习可参考 Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew Barto, MIT Press (2018)，本书只介绍监督学习和无监督学习。

### 机器学习流程

使用机器学习的典型流程：

1. 数据和问题定义：想要解决什么问题？问题类型
2. 收集数据：什么数据能回到所需问题，是否有现成数据？是否需要组合多个来源的数据？是否存在抽样偏差？需要多少数据？
3. 数据预处理：填充缺失值、平滑噪声数据、去除异常值，多个数据源的集成，数据归一化，降维等。
4. 数据分析和建模：数据分析和建模包括无监督和有监督学习、统计推断和预测。
5. 评价

## 收集数据



