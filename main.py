import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import csv
import xlsxwriter

# 读取数据文件
data_50 = pd.read_csv("D:/文件/全球校园人工智能算法精英挑战赛/code/dataset/data_50.csv", header=None)
distance_50 = pd.read_csv("D:/文件/全球校园人工智能算法精英挑战赛/code/dataset/distance_50.csv", header=None)
roads_50 = pd.read_csv("D:/文件/全球校园人工智能算法精英挑战赛/code/dataset/roads_50.csv", header=None)
speed_50 = pd.read_csv("D:/文件/全球校园人工智能算法精英挑战赛/code/dataset/speed_50.csv", header=None)
EVs_50 = pd.read_csv("D:/文件/全球校园人工智能算法精英挑战赛/code/dataset/EVs_50.csv", header=None)

# 将数据转换为numpy数组
distance_matrix = distance_50.values
speed_matrix = speed_50.values
charge_matrix = roads_50.values
# 提取公共巴士站点的经纬度
bus_stations_coords = data_50[[0,1]].values

# 提取电动汽车相关参数
start_longitudes = EVs_50[0].values
start_latitudes = EVs_50[1].values
end_longitudes = EVs_50[2].values
end_latitudes = EVs_50[3].values
initial_energy = EVs_50[4].values
battery_capacity = EVs_50[5].values
deadlines = EVs_50[6].values
energy_consumption = EVs_50[7].values


class EVDispatchEnvironment:
    def __init__(self, distance_matrix, speed_matrix, charge_matrix,
                 initial_energy, battery_capacity, deadlines, energy_consumption,
                 bus_stations_coords, start_coords, end_coords):
        self.distance_matrix = distance_matrix  # 路段距离矩阵
        self.speed_matrix = speed_matrix  # 路段行驶速度矩阵
        self.charge_matrix = charge_matrix  # 是否可以充电的矩阵
        self.initial_energy = initial_energy  # 电动汽车初始电量
        self.battery_capacity = battery_capacity  # 电池容量
        self.deadlines = deadlines  # 截止时间
        self.energy_consumption = energy_consumption  # 每100km能耗
        self.bus_stations_coords = bus_stations_coords  # 公交站点坐标
        self.start_coords = start_coords  # 电动汽车起点坐标
        self.end_coords = end_coords  # 电动汽车终点坐标
        self.n_cars = len(initial_energy)  # 电动汽车数量
        self.n_stations = len(bus_stations_coords)  # 公交站点数量
        self.reset()

    def reset(self):
        # 重置环境
        self.current_energy = np.copy(self.initial_energy)
        self.time_remaining = np.copy(self.deadlines)
        self.done = [False] * self.n_cars
        self.paths = [[] for _ in range(self.n_cars)]  # 每辆车的路径
        self.current_positions = self.get_nearest_stations(self.start_coords)  # 当前所在站点
        return self.get_state()

    def get_nearest_stations(self, coords):
        # 获取最接近的公交站点索引
        nearest_stations = []
        for coord in coords:
            distances = [geodesic(coord, station_coord).km for station_coord in self.bus_stations_coords]
            nearest_station = np.argmin(distances)
            nearest_stations.append(nearest_station)
        return nearest_stations

    def get_state(self):
        # 返回当前状态，包括当前能量、剩余时间和当前位置
        state = []
        for i in range(self.n_cars):
            state.extend([self.current_energy[i], self.time_remaining[i], self.current_positions[i]])
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        rewards = []
        for car, action in enumerate(actions):
            if not self.done[car]:
                current_pos = self.current_positions[car]
                next_pos = action  # 动作为下一个站点索引
                distance = self.distance_matrix[current_pos, next_pos]
                speed = self.speed_matrix[current_pos, next_pos]

                # 检查速度和距离是否为0
                if speed <= 0 or distance <= 0:
                    print(f"Warning: Invalid speed or distance for car {car} at step {action}")
                    rewards.append(0)
                    self.done[car] = True
                    continue

                charge = self.charge_matrix[current_pos, next_pos]
                time_taken = distance / speed
                energy_used = (distance / 100) * self.energy_consumption[car]  # 基于能耗计算
                energy_received = charge * 100 * (time_taken)  # MPT充电功率为100kW

                # 检查电池容量是否有效
                if np.isnan(self.battery_capacity[car]):
                    print(f"Warning: Invalid battery capacity for car {car}")
                    self.done[car] = True
                    continue

                # 更新电量并确保不超过电池容量
                new_energy = self.current_energy[car] - energy_used + energy_received
                if np.isnan(new_energy) or np.isnan(energy_used) or np.isnan(energy_received):
                    print(
                        f"Warning: Invalid energy calculation for car {car}. Energy used: {energy_used}, Energy received: {energy_received}")
                    self.done[car] = True
                    continue

                # 限制最大电量为电池容量
                self.current_energy[car] = min(new_energy, self.battery_capacity[car])
                self.time_remaining[car] -= time_taken

                # 检查是否到达终点
                end_station = self.get_nearest_stations([self.end_coords[car]])[0]
                if self.current_positions[car] == end_station:
                    self.done[car] = True

                # 如果电量耗尽或超出截止时间，则任务失败
                if self.current_energy[car] <= 0 or self.time_remaining[car] <= 0:
                    self.done[car] = True

                # 奖励为当前剩余电量
                rewards.append(self.current_energy[car])
            else:
                rewards.append(0)

        next_state = self.get_state()
        return next_state, rewards, self.done

    def get_total_remaining_energy(self):
        # 计算所有车辆的剩余电量总和
        return sum([self.current_energy[i] if self.done[i] else 0 for i in range(self.n_cars)])


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 状态空间维度
        self.action_size = action_size  # 动作空间维度
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # 折扣率
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        # 创建神经网络模型
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if np.random.rand() <= self.epsilon:
            # 随机选择动作
            return np.random.randint(0, self.action_size, size=(env.n_cars,))
        with torch.no_grad():
            q_values = self.model(state)
        actions = q_values.argmax(dim=1).cpu().numpy()
        return actions

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            action = torch.LongTensor(action).to(self.device)
            done = torch.BoolTensor(done).to(self.device)

            target = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
            next_q = self.model(next_state).max(1)[0]
            expected = reward + (1 - done.float()) * self.gamma * next_q

            loss = self.criterion(target, expected.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 准备电动汽车的起点和终点坐标
start_coords = list(zip(start_latitudes, start_longitudes))
end_coords = list(zip(end_latitudes, end_longitudes))

env = EVDispatchEnvironment(distance_matrix, speed_matrix, charge_matrix,
                            initial_energy, battery_capacity, deadlines,
                            energy_consumption, bus_stations_coords,
                            start_coords, end_coords)

state_size = env.n_cars * 3  # 每辆车的状态：当前能量、剩余时间、当前位置
action_size = env.n_stations  # 动作空间为所有公交站点

agent = DQNAgent(state_size, action_size)
episodes = 1000
batch_size = 64
rewards_history = []

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    for time_step in range(500):  # 每个episode的最大步数
        actions = agent.act(state)
        next_state, rewards, done = env.step(actions)
        total_reward += sum(rewards)
        agent.remember(state, actions, rewards, next_state, done)
        state = next_state
        if all(done):
            print(f"Episode {e+1}/{episodes} finished with total reward: {total_reward}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    rewards_history.append(total_reward)

# 保存训练的模型参数
torch.save(agent.model.state_dict(), 'dqn_model.pth')

# 保存累计奖励到 rewards.csv
with open('rewards.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Episode'] + [f'EV_{i}' for i in range(env.n_cars)])
    for idx, mem in enumerate(agent.memory):
        _, _, rewards, _, _ = mem
        writer.writerow([idx+1] + rewards)

# 绘制平均奖励图
plt.plot(rewards_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')
plt.savefig('episode_rewards.jpg')
plt.close()

# 生成 results.xlsx
workbook = xlsxwriter.Workbook('results.xlsx')
summary_sheet = workbook.add_worksheet('汇总')
detail_sheet = workbook.add_worksheet('调度详情')

# 汇总工作表
summary_sheet.write_row('A1', ['电动汽车序号', '到达终点的剩余电量', '到达终点的剩余截止时间'])
total_remaining_energy = 0
for i in range(env.n_cars):
    if env.done[i]:
        remaining_energy = env.current_energy[i]
        remaining_time = env.time_remaining[i]
    else:
        remaining_energy = 0
        remaining_time = 0
    summary_sheet.write_row(i+1, 0, [i, remaining_energy, remaining_time])
    total_remaining_energy += remaining_energy

# 在最后一行记录所有电动汽车剩余电量总和
summary_sheet.write_row(env.n_cars+1, 0, ['总和', total_remaining_energy, ''])

# 调度详情工作表
# 第1行记录电动汽车初始化信息
detail_sheet.write_row('A1', ['电动汽车序号', '起点序号', '终点序号', '初始电量', '电池容量', '截止时间'])
for i in range(env.n_cars):
    start_station = env.current_positions[i]
    end_station = env.get_nearest_stations([env.end_coords[i]])[0]
    detail_sheet.write_row(i+1, 0, [i, start_station, end_station, initial_energy[i],
                                    battery_capacity[i], deadlines[i]])

# 从第2行开始记录每辆车的调度路径和状态
row = env.n_cars + 2
for i in range(env.n_cars):
    detail_sheet.write(row, 0, f'电动汽车 {i} 调度路径')
    detail_sheet.write_row(row, 1, env.paths[i])
    row += 1
    detail_sheet.write(row, 0, f'电动汽车 {i} 剩余电量和时间')
    energies = []  # 记录每一步的剩余电量
    times = []     # 记录每一步的剩余时间
    for _ in env.paths[i]:
        energies.append(env.current_energy[i])
        times.append(env.time_remaining[i])
    detail_sheet.write_row(row, 1, energies)
    detail_sheet.write_row(row+1, 1, times)
    row += 2

workbook.close()
