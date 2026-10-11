# Parking Lot Charging Environment

一个用于学习强化学习环境接口的电动汽车停车场充电模拟器。

它包含随机到达的车辆、不同功率的充电桩、等待队列、基础负荷和电网越限惩罚。项目实现了 Gymnasium 的 `reset()` / `step()` / `render()` 接口，适合用来练习“状态—动作—奖励”的建模方式。

## 运行

```text
python -m pip install -r requirements.txt
python parking_lot_charging_env.py
python parking_lot_charging_env.py --policy random --seed 42
```

默认策略是所有充电桩全功率。随机策略只是为了展示环境接口，不代表最优策略。

## 我从这个项目学到什么

- 如何把现实问题抽象成观测空间、动作空间和奖励函数
- 如何用随机种子复现实验
- 如何把服务量、等待、拒绝和电网峰值放到同一个奖励里
- 为什么“能运行”不等于“策略已经最优”

## 当前边界

车辆用剩余需求表示，尚未加入 SOC、电池容量、预约离场、电价和实时碳强度。电网限制是软惩罚，环境允许越限。它是学习型仿真，不是充电站控制系统。

## 许可证

代码采用 MIT License。

