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
from args import parse_args, get_default_args, args_to_dict, dict_to_args

# 从命令行获取参数
args = parse_args()

# 读取数据文件
data_50 = pd.read_csv("./dataset/data_50.csv", header=None)
distance_50 = pd.read_csv("./dataset/distance_50.csv", header=None)
roads_50 = pd.read_csv("./dataset/roads_50.csv", header=None)
speed_50 = pd.read_csv("./dataset/speed_50.csv", header=None)
EVs_50 = pd.read_csv("./dataset/EVs_50.csv", header=None)

# 将数据转换为numpy数组
distance_matrix = distance_50.values
speed_matrix = speed_50.values
charge_matrix = roads_50.values
# 提取公共巴士站点的经纬度
bus_stations_coords = data_50[[0, 1]].values

# 提取电动汽车相关参数
start_latitudes = EVs_50[0].values
start_longitudes = EVs_50[1].values
end_latitudes = EVs_50[2].values
end_longitudes = EVs_50[3].values
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
        self.current_positions = self.get_nearest_stations(
            self.start_coords)  # 当前所在站点
        return self.get_state()

    def get_nearest_stations(self, coords):
        # 获取最接近的公交站点索引
        nearest_stations = []
        for coord in coords:
            distances = [geodesic(
                coord, station_coord).km for station_coord in self.bus_stations_coords]
            nearest_station = np.argmin(distances)
            nearest_stations.append(nearest_station)
        # print("Nearest stations:", nearest_stations)
        return nearest_stations

    def get_state(self):
        # 返回当前状态，包括当前能量、剩余时间、当前位置和终点位置
        end_positions = self.get_nearest_stations(self.end_coords)
        state = np.array([
            [self.current_energy[i], self.time_remaining[i],
            self.current_positions[i], end_positions[i]]
            for i in range(self.n_cars)
        ], dtype=np.float32)
        
        return state

    # 在EVDispatchEnvironment类中添加一个新方法
    def get_path(self, car_index):
        path = [self.get_nearest_stations([self.start_coords[car_index]])[0]]
        path.extend(self.paths[car_index])
        path.append(self.get_nearest_stations([self.end_coords[car_index]])[0])
        return path
    def step(self, actions):
        rewards = []
        for car, action in enumerate(actions):
            if not self.done[car]:
                # 检查车辆是否超时
                if self.time_remaining[car] <= 0:
                    self.done[car] = True
                    rewards.append(-100)  # 超时惩罚
                    continue

                current_pos = self.current_positions[car]
                next_pos = action  # 动作为下一个站点的索引
                distance = self.distance_matrix[current_pos, next_pos]
                speed = self.speed_matrix[current_pos, next_pos]

                # 检查速度和距离是否有效，避免无效路径
                if speed <= 0 or distance <= 0:
                    rewards.append(-10)  # 无效路径，给予小惩罚
                    self.done[car] = True
                    continue

                charge = self.charge_matrix[current_pos, next_pos]
                time_taken = distance / speed
                energy_used = (distance / 100) * \
                              self.energy_consumption[car]  # 按每100公里的能耗计算
                energy_received = charge * 100 * time_taken  # 充电功率为100kW

                # 更新电量，确保电量不超过电池容量
                new_energy = self.current_energy[car] - \
                             energy_used + energy_received
                self.current_energy[car] = min(
                    new_energy, self.battery_capacity[car])

                # 更新剩余时间
                self.time_remaining[car] -= time_taken

                
                # 修改这里：使用更合理的奖励函数
                end_station = self.get_nearest_stations([self.end_coords[car]])[0]
                if self.current_positions[car] == end_station:
                    self.done[car] = True
                    if self.time_remaining[car] > 0:
                        rewards.append(100 + self.time_remaining[car] + self.current_energy[car])
                    else:
                        rewards.append(-100)  # 超时惩罚
                else:
                    rewards.append(-1)  # 每步给予小惩罚，鼓励尽快到达目的地

                # 检查电量是否耗尽
                if self.current_energy[car] <= 0:
                    self.done[car] = True
                    rewards.append(-100)  # 电量耗尽惩罚

                # 再次检查时间是否已耗尽
                if self.time_remaining[car] <= 0:
                    self.done[car] = True
                    rewards.append(-100)  # 超时惩罚

                # 更新当前位置
                self.current_positions[car] = action

            else:
                rewards.append(0)  # 如果车辆已完成，不再给予奖励


        next_state = self.get_state()
        return next_state, rewards, self.done


class ImprovedDQNModel(nn.Module):
    def __init__(self, state_size_per_car, action_size, n_cars):
        super(ImprovedDQNModel, self).__init__()
        self.state_size_per_car = state_size_per_car
        self.action_size = action_size
        self.n_cars = n_cars

        # 每辆车的独立处理网络
        self.car_net = nn.Sequential(
            nn.Linear(state_size_per_car, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 全局信息处理网络
        self.global_net = nn.Sequential(
            nn.Linear(64 * n_cars, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 修改这里：增加注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        # 修改输出层
        self.output_layer = nn.Linear(256, action_size)

    def forward(self, x):
        # x的形状应该是 (batch_size, n_cars, state_size_per_car)
        batch_size = x.size(0)

        # 处理每辆车的状态
        car_features = self.car_net(x.view(-1, self.state_size_per_car))
        car_features = car_features.view(batch_size, -1)

        # 添加注意力机制
        car_features = car_features.view(self.n_cars, batch_size, -1).permute(1, 0, 2)
        car_features, _ = self.attention(car_features, car_features, car_features)
        car_features = car_features.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # 处理全局信息
        global_features = self.global_net(car_features)

        # 生成每辆车的动作值
        q_values = self.output_layer(global_features)
        return q_values.unsqueeze(1).expand(-1, self.n_cars, -1)


class DQNAgent:
    def __init__(self, state_size_per_car, action_size, n_cars):
        self.state_size_per_car=state_size_per_car
        self.action_size = action_size
        self.n_cars = n_cars
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImprovedDQNModel(state_size_per_car=4, action_size=self.action_size, n_cars=self.n_cars).to(self.device)
        self.target_model = ImprovedDQNModel(state_size_per_car=4, action_size=self.action_size, n_cars=self.n_cars).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.target_update_frequency = 100
        self.steps = 0

    def remember(self, state, action, reward, next_state, done):
        # 确保 reward 是一个标量
        reward = np.array(reward).mean()  # 可以取平均值或其他方式
        # 存储经验，但不转换为张量
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        # 解压 minibatch
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 将它们转换为 NumPy 数组，再转换为张量，并移动到正确的设备
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 模型预测Q值
        outputs = self.model(states)  # 形状为 [batch_size, n_cars, action_size]

        # 使用目标网络计算下一个状态的最大 Q 值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(2)[0]

        # 计算目标值
        expected = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        # 选择当前动作对应的Q值
        actions_indices = actions.unsqueeze(2)
        targets = outputs.gather(2, actions_indices).squeeze(2)

        loss = self.criterion(targets, expected.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size, size=(env.n_cars,))
        with torch.no_grad():
            q_values = self.model(state)
        actions = q_values.argmax(dim=2).cpu().numpy()
        return actions.squeeze(0)


# 准备电动汽车的起点和终点坐标
start_coords = list(zip(start_latitudes, start_longitudes))
end_coords = list(zip(end_latitudes, end_longitudes))

env = EVDispatchEnvironment(distance_matrix, speed_matrix, charge_matrix,
                            initial_energy, battery_capacity, deadlines,
                            energy_consumption, bus_stations_coords,
                            start_coords, end_coords)

# 每辆车的状态：当前能量、剩余时间、当前位置、终点位置
action_size = env.n_stations  # 动作空间为所有公交站点
n_cars = len(initial_energy)

agent = DQNAgent(state_size_per_car=4,  action_size=action_size, n_cars=n_cars)
episodes = args.episodes
batch_size = args.batch_size
rewards_history = []

# 初始化保存最优模型的变量
best_reward = -float('inf')  # 初始化为负无穷

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    for time_step in range(500):  # 每个episode的最大步数
        actions = agent.act(state)
        next_state, rewards, done = env.step(actions)
        total_reward += sum(rewards)
        agent.remember(state, actions, rewards, next_state, done)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if all(done):
            break


    # 添加这里：学习率衰减
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'] * 0.99, 1e-4)

    rewards_history.append(total_reward)

    # 打印每轮的奖励
    print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # 保存表现最好的模型
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.model.state_dict(), 'best_dqn_model.pth')
        print(f"New best model saved with reward: {best_reward:.2f}")


    # 保存表现最好的模型
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(agent.model.state_dict(), 'best_dqn_model.pth')  # 保存最优模型
        print(f"New best model saved with reward: {best_reward}")

# 保存累计奖励到 rewards.csv
with open('rewards.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Episode'] + [f'EV_{i}' for i in range(env.n_cars)])
    for idx, mem in enumerate(agent.memory):
        _, _, rewards, _, _ = mem
        writer.writerow([idx + 1] + rewards)

# 绘制平均奖励图
plt.plot(rewards_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episode Rewards')
plt.savefig('episode_rewards.jpg')
plt.close()

# 在训练循环结束后，加载最佳模型并进行预测
best_model = ImprovedDQNModel(state_size_per_car=4, action_size=action_size, n_cars=n_cars).to(agent.device)
best_model.load_state_dict(torch.load('best_dqn_model.pth'))
best_model.eval()


# 在使用最佳模型进行预测时记录路径和状态
paths = [[] for _ in range(env.n_cars)]
energy_history = [[] for _ in range(env.n_cars)]
time_history = [[] for _ in range(env.n_cars)]

state = env.reset()
done = [False] * env.n_cars

while not all(done):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        actions = best_model(state_tensor).argmax(dim=2).cpu().numpy().squeeze(0)
    
    next_state, rewards, done = env.step(actions)
    
    # 记录每辆车的路径、能量和时间
    for i in range(env.n_cars):
        if not done[i]:
            paths[i].append(actions[i])
            energy_history[i].append(env.current_energy[i])
            time_history[i].append(env.time_remaining[i])
    
    state = next_state

# 创建结果工作簿和工作表
workbook = xlsxwriter.Workbook('results.xlsx')
summary_sheet = workbook.add_worksheet('汇总')
detail_sheet = workbook.add_worksheet('调度详情')

# 汇总工作表
summary_sheet.write_row('A1', ['电动汽车序号', '到达终点的剩余电量', '到达终点的剩余截止时间'])
total_remaining_energy = 0
for i in range(env.n_cars):
    if env.done[i]:  # 确保只计算已到达终点的电动汽车
        remaining_energy = env.current_energy[i]
        remaining_time = env.time_remaining[i]
        total_remaining_energy += remaining_energy
    else:
        remaining_energy = 0
        remaining_time = 0
    summary_sheet.write_row(i + 1, 0, [i, remaining_energy, remaining_time])

summary_sheet.write_row(env.n_cars + 1, 0, ['总和', total_remaining_energy, ''])

# 调度详情工作表
detail_sheet.write_row('A1', ['电动汽车序号', '起点序号', '终点序号', '初始电量', '电池容量', '截止时间'])
for i in range(env.n_cars):
    start_station = env.get_nearest_stations([env.start_coords[i]])[0]
    end_station = env.get_nearest_stations([env.end_coords[i]])[0]
    detail_sheet.write_row(i + 1, 0, [i, start_station, end_station, initial_energy[i],
                                      battery_capacity[i], deadlines[i]])

# 在写入Excel文件时修改路径记录
row = env.n_cars + 2
for i in range(env.n_cars):
    complete_path = env.get_path(i)
    detail_sheet.write(row, 0, f'电动汽车 {i} 调度路径')
    detail_sheet.write_row(row, 1, complete_path)
    row += 1
    detail_sheet.write(row, 0, f'电动汽车 {i} 剩余电量')
    detail_sheet.write_row(row, 1, energy_history[i])
    row += 1
    detail_sheet.write(row, 0, f'电动汽车 {i} 剩余时间')
    detail_sheet.write_row(row, 1, time_history[i])
    row += 2

workbook.close()