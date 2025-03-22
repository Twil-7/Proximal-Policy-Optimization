import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# torch.distributions 模块提供了一组概率分布的实现，可以用来构建统计模型、生成随机样本、计算概率密度函数（PDF）、对数概率
# Categorical 是 torch.distributions 中的一个类，用于表示离散类别型分布（Categorical Distribution），这种分布适用于具有有限个类别的随机变量，每个类别都有一个对应的概率。

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 环境参数
TARGET_RANGE = (4.0, 6.0)
INITIAL_HEIGHT = 10.0
X_BOUNDS = (0.0, 10.0)


class DropBlockEnv:
    def __init__(self):
        self.x = None
        self.y = None
        self.reset()

    def reset(self):
        self.y = INITIAL_HEIGHT
        self.x = np.random.uniform(X_BOUNDS[0], X_BOUNDS[1])
        return self._get_state()

    def _get_state(self):
        return np.array([self.x / X_BOUNDS[1], self.y / INITIAL_HEIGHT], dtype=np.float32)

    def step(self, action):
        new_x = self.x + (1.0 if action else -1.0)
        new_x = np.clip(new_x, X_BOUNDS[0], X_BOUNDS[1])
        self.y -= 1.0
        self.x = new_x

        done = self.y <= 0
        reward = self._calculate_reward(done)
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, done):
        if done:
            if TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]:
                return 10.0
            return -10.0

        distance = abs(self.x - np.mean(TARGET_RANGE))
        in_target_air = TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]

        reward = 0.0
        reward += 0.2 if in_target_air else -0.1
        reward -= 0.05 * distance
        reward -= 0.1

        return reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=4, batch_size=64):
        # print(state_dim): 2
        # print(action_dim): 2
        # print(lr): 0.0003
        # print(gamma): 0.99
        # print(epsilon): 0.2
        # print(epochs): 4
        # print(batch_size): 64

        self.actor = Actor(state_dim, action_dim).to(device)    # (2, 2)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma    # 0.99
        self.epsilon = epsilon    # 0.2
        self.epochs = epochs    # 4
        self.batch_size = batch_size    # 64

    def update(self, states, actions, old_probs, rewards, dones):
        # print(len(states)): 100
        # print(len(actions)): 100
        # print(len(old_probs)): 100
        # print(len(rewards)): 100
        # print(len(dones)): 100

        """ 我终于明白了，这里dones的作用是为了区分不同轨迹的，因为在计算折扣累积回报时，需要是同一条轨迹上的奖励 """

        # 计算折扣回报
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)

        """
        # print(returns):
        # [-12.742948760590421, -12.417119960192345, -12.138505010295297, -11.806570717469999, -11.471283553, -11.183114699999999, -10.84153, -10.547, -10.3, -10.0, 
        #  -12.221124047168484, -12.075885935618972, -11.878675721932595, -11.6289684353807, -11.3262338025, -10.98609475, -10.693024999999999, -10.4475, -10.25, -10.0, 
        #  -10.268020568181893, -10.472070515434758, -10.324646219730583, -10.529268143261719, -10.68680697087192, -10.49240174623576, -10.24552778195681, -10.04666519177605, -10.199329242098518, -10.0, 
        #  -12.494597849434882, -12.295562797632424, -12.145022341266307, -11.942456223724772, -11.68733893327878, -11.379139649999997, -11.039534999999997, -10.696499999999999, -10.35, -10.0, 
        #  -12.008353090016715, -11.804394293115976, -11.648880356852603, -11.441290522243136, -11.282108871122462, -11.070814274030873, -10.806880337574722, -10.590785452265479, -10.322002739831898, -10.0, 
        #  7.188011384247763, 7.635322806225922, 8.03664747489073, 8.391520877582455, 8.800483910604399, 9.163072832848787, 9.57982931996433, 9.95029041806084, 9.9709581939159, 10.0, 
        #  -12.203222101537332, -11.990034372373602, -11.724188181299125, -11.506161725668342, -11.336438033111994, -11.114494909317704, -10.839804885283067, -10.612845265046062, -10.333088072887469, -10.0, 
        #  7.927295408934597, 7.90692213616788, 7.935721848784117, 7.915433691571436, 7.944319379494779, 8.327032195578964, 8.76411584818925, 9.155109436684493, 9.600557505871604, 10.0, 
        #  -9.476017668301274, -9.315132706467931, -9.506157997545362, -9.656761905498367, -9.851237996566004, -9.694143139058568, -9.888996818343985, -10.043467785092936, -9.845963711101978, -10.0, 
        #  -12.980759103351167, -12.657332427627441, -12.330638815785296, -12.000645268469997, -11.667318452999998, -11.3306247, -10.99053, -10.647, -10.3, -10.0]
        """

        # print(len(returns)): 100

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = returns - returns.mean()    # 取若干段轨迹的奖励的平均值，作为基线
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # print(advantages)
        # tensor([-0.7178, -0.6769, -0.6419, -0.6002, -0.5581, -0.5219, -0.4790, -0.4420,
        #         -0.4109, -0.3733, -0.6523, -0.6340, -0.6093, -0.5779, -0.5399, -0.4971,
        #         -0.4603, -0.4295, -0.4047, -0.3733, -0.4069, -0.4326, -0.4140, -0.4397,
        #         -0.4595, -0.4351, -0.4041, -0.3791, -0.3983, -0.3733, -0.6866, -0.6616,
        #         -0.6427, -0.6173, -0.5852, -0.5465, -0.5038, -0.4608, -0.4172, -0.3733,
        #         -0.6255, -0.5999, -0.5804, -0.5543, -0.5343, -0.5078, -0.4746, -0.4475,
        #         -0.4137, -0.3733,  1.7859,  1.8421,  1.8926,  1.9371,  1.9885,  2.0341,
        #          2.0864,  2.1330,  2.1355,  2.1392, -0.6500, -0.6232, -0.5899, -0.5625,
        #         -0.5411, -0.5133, -0.4788, -0.4502, -0.4151, -0.3733,  1.8788,  1.8763,
        #          1.8799,  1.8773,  1.8810,  1.9290,  1.9839,  2.0331,  2.0890,  2.1392,
        #         -0.3074, -0.2872, -0.3112, -0.3301, -0.3546, -0.3348, -0.3593, -0.3787,
        #         -0.3539, -0.3733, -0.7477, -0.7071, -0.6660, -0.6246, -0.5827, -0.5404,
        #         -0.4977, -0.4545, -0.4109, -0.3733])

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)    # 这里的action是有用的，需要提前被记录，用来计算新的网络采取该动作的概率
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(device)
        # print(states.shape): torch.Size([100, 2])
        # print(actions.shape): torch.Size([100])
        # print(old_probs.shape): torch.Size([100])

        # 多次epoch训练
        for _ in range(self.epochs):    # epoch = 4
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            # 每一次随机从中选取64个状态-动作对，用于更新网络参数
            for start in range(0, len(states), self.batch_size):    # batch = 64
                end = start + self.batch_size
                idx = indices[start:end]
                # print(start, end): 0 64
                # print(idx):
                # [44 47  4 55 26 64 73 10 40 74 18 62 11 36 89 91 86  0 88 23 65 45 31 70
                #  42 12 15 71 76 97 24 78 22 96 56 82 30 53 51  9 33 25 69 28 98 85  5 90
                #  68 39 49 35 16 66 34 60  7 43 72 67 83 27 19 95]

                # 无论新的网络、旧的网络，由于用的是同一组动作、状态，只是网络参数不一样，advantages自然也相同
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_probs = old_probs[idx]
                batch_advantages = advantages[idx]
                # print(batch_states.shape): torch.Size([64, 2])
                # print(batch_actions.shape): torch.Size([64])
                # print(batch_old_probs.shape): torch.Size([64])
                # print(batch_advantages.shape): torch.Size([64])

                # 计算新策略，执行同样动作的概率
                new_probs = self.actor(batch_states)
                # print(new_probs.shape): torch.Size([64, 2])
                new_probs = new_probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
                # print(new_probs.shape): torch.Size([64])

                # 计算比率和损失
                ratio = new_probs / batch_old_probs
                # print(ratio.shape): torch.Size([64])

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                # print(surr1.shape): torch.Size([64])
                # print(surr2.shape): torch.Size([64])

                loss = -torch.min(surr1, surr2).mean()
                # print(loss): tensor(0.0756, grad_fn=<NegBackward0>)

                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def evaluate(policy, num_episodes=1000):
    """使用独立环境评估当前策略性能"""
    env = DropBlockEnv()
    safe_landings = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():  # 关闭梯度计算
                action_probs = policy.actor(state_tensor)
                action = torch.argmax(action_probs).item()  # 使用确定性策略

            next_state, _, done, _ = env.step(action)
            state = next_state

        # 判断最终位置
        final_x = state[0] * X_BOUNDS[1]
        if TARGET_RANGE[0] <= final_x <= TARGET_RANGE[1]:
            safe_landings += 1

    return safe_landings / num_episodes


def train():
    env = DropBlockEnv()
    state_dim = 2    # 状态空间维度为2，表示水平坐标、竖直坐标
    action_dim = 2    # 动作空间维度为2，表示向左移动1格、向右移动1格
    ppo = PPO(state_dim, action_dim)    # 智能体，包含决策网络

    num_episodes_per_update = 10
    total_updates = 2501
    eval_updates = 10

    success_rate = evaluate(ppo, num_episodes=10000)  # 使用独立的评估函数
    print(f"Evaluation Safe Rate: {success_rate:.4f}")

    # 这里执行了2500次采样，而决策网络参数更新了2500*4=10000次，确实效率大大提升了
    for update in range(total_updates):
        states, actions, old_probs, rewards, dones = [], [], [], [], []

        # 收集数据
        for _ in range(num_episodes_per_update):
            state = env.reset()
            # print(state): [0.9507143 1.       ]
            episode_states, episode_actions = [], []
            episode_old_probs, episode_rewards, episode_dones = [], [], []
            done = False

            while not done:
                state_tensor = torch.FloatTensor(state).to(device)
                # print(state_tensor): tensor([0.9507, 1.0000])
                with torch.no_grad():
                    action_probs = ppo.actor(state_tensor)
                    # print(action_probs): tensor([0.5940, 0.4060])
                    dist = Categorical(action_probs)
                    # print(dist): Categorical(probs: torch.Size([2]))
                    action = dist.sample()
                    # print(action): tensor(1)
                    old_prob = action_probs[action.item()].item()
                    # print(old_prob): 0.4060162901878357

                next_state, reward, done, _ = env.step(action.item())
                # print(next_state): [1.  0.9]
                # print(reward): -0.44999999999999996
                # print(done): False
                # print(_): {}

                episode_states.append(state)
                episode_actions.append(action.item())
                episode_old_probs.append(old_prob)
                episode_rewards.append(reward)
                episode_dones.append(done)

                state = next_state
            """
            这里得到的是：每个时间步的当前状态、当前状态即将做出的动作、该动作对应的采样概率、该动作的奖励值
            """
            # print(episode_states):
            # [array([0.9507143, 1.], dtype=float32), array([1. , 0.9], dtype=float32),
            #  array([0.9, 0.8], dtype=float32), array([1. , 0.7], dtype=float32),
            #  array([1. , 0.6], dtype=float32), array([0.9, 0.5], dtype=float32),
            #  array([1. , 0.4], dtype=float32), array([0.9, 0.3], dtype=float32),
            #  array([0.8, 0.2], dtype=float32), array([0.9, 0.1], dtype=float32)]
            # print(episode_actions): [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
            # print(episode_old_probs):
            # [0.4060162901878357, 0.5872591733932495, 0.4245370328426361, 0.42960381507873535, 0.5617495775222778,
            #  0.4512947201728821, 0.5442335605621338, 0.530620813369751, 0.4834785461425781, 0.5127178430557251]
            # print(episode_rewards):
            # [-0.44999999999999996, -0.4, -0.44999999999999996, -0.44999999999999996, -0.4, -0.44999999999999996,
            # -0.4, -0.35, -0.4, -10.0]
            # print(episode_dones): [False, False, False, False, False, False, False, False, False, True]

            states.extend(episode_states)
            actions.extend(episode_actions)
            old_probs.extend(episode_old_probs)
            rewards.extend(episode_rewards)
            dones.extend(episode_dones)

        """ 这里采用old_theta，采集了100个状态-动作对，我发现其实在更新决策网络的时候，样本并不一定要是轨迹的形式，其实只需要样本对就行，
            因为本质上，决策网络只需要输入某一个时间步的状态，根本不需要一条轨迹所有状态. 
            
            这时我才真正理解，为什么在PG算法里只采样一条轨迹，依旧能够稳定的训练。因为哪怕只有一条轨迹，依然有若干个状态-动作对。  """
        # print(len(states)): 100
        # print(len(actions)): 100
        # print(len(old_probs)): 100
        # print(len(rewards)): 100
        # print(len(dones)): 100

        # 更新策略
        ppo.update(states, actions, old_probs, rewards, dones)

        # 评估性能
        # 每10次更新进行一次独立评估
        if (update + 1) % eval_updates == 0:
            success_rate = evaluate(ppo)  # 使用独立的评估函数
            print(f"Update {update + 1}, Evaluation Safe Rate: {success_rate:.3f}")

    success_rate = evaluate(ppo, num_episodes=10000)  # 使用独立的评估函数
    print(f"Evaluation Safe Rate: {success_rate:.4f}")

    # 保存网络权重
    torch.save(ppo.actor.state_dict(), 'policy_network_weights.pth')
    print("Policy network weights saved to 'policy_network_weights.pth'")


if __name__ == "__main__":
    train()

    """ 强化学习算法真的太神奇了，明明模型架构没有一点修改，只是调整了最优化方式，居然效率能够大幅上升，精度依旧很高
        这时突然觉得，强化学习相比深度学习灵活性太高了，可以在最优化方式上进行更多的改进  """
