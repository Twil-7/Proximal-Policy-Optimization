import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
        # 归一化状态到[0,1]范围
        return np.array([self.x / X_BOUNDS[1], self.y / INITIAL_HEIGHT], dtype=np.float32)

    def step(self, action):
        # 执行动作并更新位置
        new_x = self.x + (1.0 if action else -1.0)
        new_x = np.clip(new_x, X_BOUNDS[0], X_BOUNDS[1])  # 限制移动范围
        self.y -= 1.0
        self.x = new_x

        done = self.y <= 0
        reward = self._calculate_reward(done)

        return self._get_state(), reward, done, {}

    def _calculate_reward(self, done):
        """精细化的奖励计算"""
        if done:
            # 最终奖励
            if TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]:
                return 10.0
            return -10.0

        # 过程奖励
        distance = abs(self.x - np.mean(TARGET_RANGE))
        in_target_air = TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]

        reward = 0.0
        reward += 0.2 if in_target_air else -0.1  # 位置奖励
        reward -= 0.05 * distance  # 距离惩罚
        reward -= 0.1  # 时间惩罚
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


def visualize_state(env, step_num):
    os.makedirs("images/", exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.ylim(0, INITIAL_HEIGHT)
    plt.xlim(X_BOUNDS[0], X_BOUNDS[1])
    plt.axvspan(TARGET_RANGE[0], TARGET_RANGE[1], color='g', alpha=0.2, label='Target Range')
    plt.scatter(env.x, env.y, color='r', label='Block', s=200)
    plt.title(f'Step {step_num}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # 调整图例位置，避免遮挡轨迹
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    plt.savefig(f'images/step_{str(step_num).zfill(2)}.png')
    plt.close()


# 加载模型
path1 = 'policy_network_weights.pth'
agent = Actor(state_dim=2, action_dim=2, hidden_size=128)
agent.load_state_dict(torch.load(path1))
agent.eval()

# 初始化环境
env = DropBlockEnv()
state = env.reset()
step_num = 0

# 可视化初始情形
visualize_state(env, step_num)

done = False
reward = None
while not done:
    step_num += 1
    state_tensor = torch.FloatTensor(state)
    probs = agent(state_tensor)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    state, reward, done, _ = env.step(action)
    visualize_state(env, step_num)

if reward > 0:
    print("Success!")
else:
    print("Defeat!")

print("Visualization completed. Images saved as step_*.png")


# 将图片合成视频
image_folder = 'images/'
video_name = 'output.mp4'
fps = 2  # 每张图片停留 0.5 秒

images = [img for img in os.listdir(image_folder) if img.startswith("step_") and img.endswith(".png")]
images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

print(f"Video saved as {video_name}")