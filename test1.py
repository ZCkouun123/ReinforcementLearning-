import gym
import numpy as np

# 状态空间离散化的边界
state_bins = [
    np.linspace(-2.4, 2.4, 10),   # 车位置
    np.linspace(-3.0, 3.0, 10),   # 车速度
    np.linspace(-0.2095, 0.2095, 10),  # 杆角度 (-12度到12度)
    np.linspace(-3.5, 3.5, 10)    # 杆顶端的速度
]

# 将连续状态离散化为表格
def discretize_state(state):
    discretized = []
    #enumrate返回索引和对应的值
    for i, s in enumerate(state):
        discretized.append(np.digitize(s, state_bins[i]) - 1)
    return tuple(discretized)

# 初始化 Q 表
q_table = np.zeros([10, 10, 10, 10, 2])

# Q-learning 参数
alpha = 0.1   # 学习率
gamma = 0.99  # 折扣因子
epsilon = 0.1  # 探索率
episodes = 10000  # 训练回合数

# 创建环境
env = gym.make('CartPole-v1')


# 修改环境参数
env.env.theta_threshold_radians = 12 * 2 * 3.14159 / 360  # 修改杆角度限制为 ±12 度
env.env.x_threshold = 2.4  # 修改小车的位置限制为 ±2.4
env.env.max_episode_steps=1000 #修改最大步数


def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # 随机探索
    else:
        return np.argmax(q_table[state])  # 利用当前策略选择最优动作

# Q-learning 训练过程
for episode in range(episodes):
    observation, info = env.reset()  # 获取重置后的 observation 和 info
    state = discretize_state(observation)  # 对 observation 进行离散化
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated 
        next_state = discretize_state(next_observation)
        
        # 更新 Q 值
        best_next_action = np.argmax(q_table[next_state])
        q_value = q_table[state + (action,)]
        next_q_value = q_table[next_state + (best_next_action,)]
        q_table[state + (action,)] += alpha * (reward + gamma * next_q_value - q_value)
        
        state = next_state
        total_reward += reward
    
    if episode % 1000 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

# 测试训练好的策略
# 创建测试环境，指定渲染模式
test_env = gym.make('CartPole-v1', render_mode='human')
for _ in range(5):
    observation, info = test_env.reset()
    state = discretize_state(observation)
    done = False
    total_reward = 0
    
    while not done:
        action = np.argmax(q_table[state])
        next_observation, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated  # 新版 Gym 处理终止条件
        next_state = discretize_state(next_observation)
        state = next_state
        total_reward += reward
    
    print(f"Test Total Reward: {total_reward}")

test_env.close()
