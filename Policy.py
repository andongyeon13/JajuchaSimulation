# 1. Prerequisitories
# 되도록이면 건들지 말기
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
# Server
import socket
import cv2
import pickle
import random
import math

device = 'cpu'

# 2. Environment
# 여기에서 자주차 Env를 불러와야 함 (파일명을 jajucha, 클래스명을 env 라고 가정)
# 이때 env 라는 클래스에는 env.reset(), env.step() 함수가 필요함
def reset(i):
    HOST = '127.0.0.1'
    PORT = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    try:
        data = ("reset "+str(i)).encode()
        client_socket.sendall(data)

        # print("go")

        img = client_socket.recv(1024 ** 3)
        gray_img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print('에러 발생:', e)

    # 클라이언트 소켓 닫기
    client_socket.close()

    return gray_img

def step(action):
    HOST = '127.0.0.1'
    PORT = 12345
    state = [None, None, None, None, None]

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    action = list(map(float, action))
    #print(action)

    try:
        data = ("get_state "+str(action[0])+" "+str(action[1])).encode()
        client_socket.sendall(data)
        #print("debug")
        state = pickle.loads(client_socket.recv(1024 ** 3))
        # print(state)

        #gray_array = (gray_img * 255).astype(np.uint8)
        #cv2.imwrite("gray_image.png", gray_array)
        #image = cv2.imread("gray_image.png", cv2.IMREAD_GRAYSCALE)
        #cv2.imshow("Gray Image", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


    except Exception as e:
        print('에러 발생:', e)

    # 클라이언트 소켓 닫기
    client_socket.close()

    img = cv2.imdecode(np.frombuffer(state[0], np.uint8), cv2.IMREAD_GRAYSCALE)
    velocity = state[1]
    steer = state[2]
    reward = state[3]
    done = state[4]

    return [img, np.array([velocity, steer])], reward, done

# 3. Model Hyperparameters
# hyperparameter를 원하는대로 설정해야 함 !!
model_hyperparameters = {
    "h_size": 8,
    "n_training_episodes": 1000,  # 훈련 몇 번 시킬건지
    "n_evaluation_episodes": 10,  # 모델 검증 (안해도 됨) 몇 번 시킬건지
    "max_t": 1000,  # 최대 시행 횟수, 이걸 횟수를 넘어가면 Episode 종료
    "gamma": 0.9,  # Discount Factor
    "lr": 1e-4,  # learning rate
    "env_id": None,  # 무시하셈
    "state_space": 800,  # 이거는 상수, 건들지 말기
    "action_space": 2,  # 이거도 상수, 건들지 말기
}


# 4. Policy Class
# 건들지 말기
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 4)),
            nn.MaxPool2d(kernel_size=(3, 4))
            # out = 20*40
        )

        self.fc = nn.Sequential(
            nn.Linear(s_size, h_size),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(h_size + 2, a_size),
            nn.Tanh()
        )

    def forward(self, img, vs):
        pooled = self.pool(img)
        # Fixed

        # 텐서를 넘파이 배열로 변환
        #numpy_data = pooled.squeeze().numpy()

        # 이미지 출력
        #plt.imshow(numpy_data, cmap='gray')
        #plt.show()

        torch.nan_to_num(pooled)
        out = pooled.view(pooled.shape[0], -1)
        out = self.fc(out)
        out = torch.cat([out, vs], dim=1)
        res = self.mu(out)
        return res[0]

    def act(self, state, sig):
        img = torch.from_numpy(state[0]).float().unsqueeze(0).to(device)
        vs = torch.from_numpy(state[1]).float().unsqueeze(0).to(device)

        v_mu, s_mu = self.forward(img, vs)

        if math.isnan(v_mu):
            v_mu = torch.tensor(1e-3)
            print('c')

        if math.isnan(s_mu):
            s_mu = torch.tensor(1e-3)
            print('c')

        v_norm = Normal(v_mu, sig)
        s_norm = Normal(s_mu, sig)

        v_action = np.clip(v_norm.sample(), -1, 1)
        s_action = np.clip(s_norm.sample(), -1, 1)

        return [v_action, s_action], [v_norm.log_prob(v_action), s_norm.log_prob(s_action)]




# 5. Reinforce Function
# 건들지 말기
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []

    sig_start = 2
    sig_end = 0.05
    decay_factor = (sig_end - sig_start) / 1000

    sig = sig_start

    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        i = random.choice([5,11,17,23,29])
        img = reset(i)
        vs = np.array([40, 0])
        sig = sig - decay_factor
        sig = max(sig, sig_end)
        state = (img, vs)
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state, sig)
            saved_log_probs.append(log_prob)
            state, reward, done = step(action)
            # env 실행: velocity += action[0] / steer += action[1]
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-sum(log_prob) * disc_return)
        # policy_loss = torch.cat(policy_loss).sum()
        policy_loss = sum(policy_loss)

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))

    torch.save(policy, 'policy.pt')

    return scores


# 6. 실행
# 되도록 건들지 말기
model_policy = Policy(
    model_hyperparameters["state_space"],
    model_hyperparameters["action_space"],
    model_hyperparameters["h_size"],
).to(device)
model_optimizer = optim.Adam(model_policy.parameters(), lr=model_hyperparameters["lr"])
scores = reinforce(
    model_policy,
    model_optimizer,
    model_hyperparameters["n_training_episodes"],
    model_hyperparameters["max_t"],
    model_hyperparameters["gamma"],
    50,
)




# 7. Evaluation
# 이건 일단 하지 말기
"""
def evaluate_agent(env, max_steps, n_eval_episodes, policy):

    # Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    # :param env: The evaluation environment
    # :param n_eval_episodes: Number of episode to evaluate the agent
    # :param policy: The Reinforce agent

    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward



eval_env = gym.make(env_id)

evaluate_agent(
    eval_env, model_hyperparameters["max_t"], model_hyperparameters["n_evaluation_episodes"], model_policy
)
"""