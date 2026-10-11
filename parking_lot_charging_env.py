"""基础停车场充电环境。安装: pip install -r requirements.txt

运行: python parking_lot_charging_env.py
随机动作: python parking_lot_charging_env.py --policy random
车辆以剩余充电需求(kWh)表示；完成后立即离场；队列按 FIFO 调度。
"""

from collections import deque
import argparse

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ParkingLotChargingEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, charger_powers_kw=(7.0, 22.0, 50.0),
                 queue_capacity=20, arrival_rate_per_hour=4.0,
                 dt_hours=0.25, episode_steps=96, grid_limit_kw=100.0,
                 render_mode=None):
        super().__init__()
        self.charger_powers = np.asarray(charger_powers_kw, dtype=np.float64)
        if (self.charger_powers.ndim != 1 or self.charger_powers.size == 0
                or not np.all(np.isfinite(self.charger_powers))
                or np.any(self.charger_powers <= 0)):
            raise ValueError("charger_powers_kw 必须是非空的正数序列")
        for name, value in (("queue_capacity", queue_capacity),
                            ("episode_steps", episode_steps)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name, value in (("dt_hours", dt_hours), ("grid_limit_kw", grid_limit_kw)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正数")
        if not np.isfinite(arrival_rate_per_hour) or arrival_rate_per_hour < 0:
            raise ValueError("arrival_rate_per_hour 必须为非负数")
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode 应为 None、human 或 ansi")

        self.n_chargers = len(self.charger_powers)
        self.queue_capacity = queue_capacity
        self.arrival_rate = float(arrival_rate_per_hour)
        self.dt_hours = float(dt_hours)
        self.episode_steps = episode_steps
        self.grid_limit_kw = float(grid_limit_kw)
        self.render_mode = render_mode
        self.efficiency = 0.95
        self.max_demand_kwh = 35.0
        self.base_load_max_kw = 120.0
        self.wait_weight, self.overload_weight, self.reject_weight = 2.0, 3.0, 10.0

        # 每个桩：0=暂停，1=额定功率的50%，2=额定功率的100%。
        self.action_space = spaces.MultiDiscrete([3] * self.n_chargers)
        # [时间比例, 基础负荷kW, 队列长度, 各桩剩余kWh, 队列需求kWh(补零)]。
        high = np.array([1.0, self.base_load_max_kw, queue_capacity]
                        + [self.max_demand_kwh] * (self.n_chargers + queue_capacity),
                        dtype=np.float32)
        self.observation_space = spaces.Box(np.zeros_like(high), high, dtype=np.float32)
        self._ready = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.queue = deque()
        self.remaining_kwh = np.zeros(self.n_chargers, dtype=np.float64)
        self.total_arrivals = self.total_completed = self.total_rejected = 0
        self.total_delivered_kwh = 0.0
        self.base_load_kw = self._sample_base_load()
        arrivals, rejected = self._arrive_and_assign()
        self._ready = True
        info = {"arrivals": arrivals, "rejected": rejected,
                "queue_length": len(self.queue), "total_arrivals": self.total_arrivals}
        return self._observation(), info

    def _sample_base_load(self):
        hour = (self.t * self.dt_hours) % 24.0
        # 日周期负荷 + 高斯噪声；裁剪避免负值或超出观测空间。
        load = 60.0 + 25.0 * np.sin(2 * np.pi * (hour - 8.0) / 24.0)
        return float(np.clip(load + self.np_random.normal(0.0, 5.0),
                             0.0, self.base_load_max_kw))

    def _dispatch(self):
        for i in range(self.n_chargers):
            if self.remaining_kwh[i] == 0.0 and self.queue:
                self.remaining_kwh[i] = self.queue.popleft()

    def _arrive_and_assign(self):
        # 时段边界先服务旧队列，再接收新车。空桩按编号顺序使用。
        self._dispatch()
        arrivals = int(self.np_random.poisson(self.arrival_rate * self.dt_hours))
        rejected = 0
        for _ in range(arrivals):
            demand = float(self.np_random.uniform(8.0, self.max_demand_kwh))
            idle = np.flatnonzero(self.remaining_kwh == 0.0)
            if idle.size:
                self.remaining_kwh[idle[0]] = demand
            elif len(self.queue) < self.queue_capacity:
                self.queue.append(demand)
            else:
                rejected += 1  # 等待区已满，新车离开。
        self.total_arrivals += arrivals
        self.total_rejected += rejected
        return arrivals, rejected

    def _observation(self):
        queue_demands = np.zeros(self.queue_capacity, dtype=np.float32)
        queue_demands[:len(self.queue)] = list(self.queue)
        return np.concatenate((
            [self.t / self.episode_steps, self.base_load_kw, len(self.queue)],
            self.remaining_kwh, queue_demands)).astype(np.float32)

    def step(self, action):
        if not self._ready:
            raise RuntimeError("请先调用 reset()；回合结束后也必须重新 reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"动作必须包含 {self.n_chargers} 个 0、1 或 2")
        action = np.asarray(action, dtype=np.int64)
        fractions = np.array([0.0, 0.5, 1.0])[action]
        waiting_vehicle_hours = len(self.queue) * self.dt_hours
        occupied = self.remaining_kwh > 0.0

        # 桩功率是电网侧功率；充入电池的能量还需乘充电效率。
        delivered = np.minimum(self.remaining_kwh,
                               self.charger_powers * fractions * self.dt_hours * self.efficiency)
        self.remaining_kwh -= delivered
        finished = occupied & (self.remaining_kwh <= 1e-9)
        self.remaining_kwh[finished] = 0.0
        completed = int(np.count_nonzero(finished))
        energy_kwh = float(delivered.sum())
        charging_power_kw = energy_kwh / self.efficiency / self.dt_hours
        interval_base_load = self.base_load_kw
        grid_load_kw = interval_base_load + charging_power_kw
        overload_kwh = max(0.0, grid_load_kw - self.grid_limit_kw) * self.dt_hours
        self.total_completed += completed
        self.total_delivered_kwh += energy_kwh

        self.t += 1
        truncated = self.t >= self.episode_steps
        arrivals = rejected = 0
        # 截断边界不再生成新车；其余边界释放空桩、排队补位、接收新车。
        if not truncated:
            arrivals, rejected = self._arrive_and_assign()
            self.base_load_kw = self._sample_base_load()
        self._ready = not truncated

        reward = (energy_kwh - self.wait_weight * waiting_vehicle_hours
                  - self.overload_weight * overload_kwh - self.reject_weight * rejected)
        info = {
            "step": self.t, "arrivals": arrivals, "rejected": rejected,
            "completed": completed, "queue_length": len(self.queue),
            "delivered_kwh": energy_kwh, "charging_power_kw": charging_power_kw,
            "base_load_kw": interval_base_load, "grid_load_kw": grid_load_kw,
            "overload_kwh": overload_kwh, "waiting_vehicle_hours": waiting_vehicle_hours,
            "total_arrivals": self.total_arrivals, "total_completed": self.total_completed,
            "total_rejected": self.total_rejected,
            "total_delivered_kwh": self.total_delivered_kwh,
        }
        return self._observation(), float(reward), False, truncated, info

    def render(self):
        if not hasattr(self, "queue"):
            raise RuntimeError("请先调用 reset()")
        piles = ", ".join(f"{i}:{power:g}kW/剩余{need:.1f}kWh"
                          for i, (power, need) in enumerate(zip(self.charger_powers, self.remaining_kwh)))
        text = (f"step={self.t:02d} | base={self.base_load_kw:.1f}kW | "
                f"queue={len(self.queue)} | piles=[{piles}]")
        if self.render_mode == "human":
            print(text)
        elif self.render_mode == "ansi":
            return text

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="停车场充电 Gymnasium 基础示例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=("all_on", "random"), default="all_on")
    args = parser.parse_args()
    env = ParkingLotChargingEnv(render_mode="human")
    observation, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)  # 动作随机数与环境随机数分别设种子。
    total_reward = 0.0
    env.render()
    while True:
        action = (env.action_space.sample() if args.policy == "random"
                  else np.full(env.n_chargers, 2, dtype=np.int64))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["step"] % 12 == 0 or terminated or truncated:
            env.render()
            print(f"  action={action.tolist()} | load={info['grid_load_kw']:.1f}kW "
                  f"| energy={info['delivered_kwh']:.2f}kWh | reward={reward:.2f}")
        if terminated or truncated:
            break
    print(f"\n总到达={info['total_arrivals']}，完成={info['total_completed']}，"
          f"拒绝={info['total_rejected']}，充入电量={info['total_delivered_kwh']:.2f}kWh，"
          f"累计奖励={total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
