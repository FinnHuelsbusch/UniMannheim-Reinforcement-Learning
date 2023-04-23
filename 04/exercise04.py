import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import tqdm
from collections import namedtuple
from typing import Iterator, List, Tuple
import collections
import copy
import random
from PIL import Image
import ale_py
from ale_py import ALEInterface

ale = ALEInterface()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# In this exercise we will use deep Q-learning to train an agent to play the Atari game of breakout.
# We will use some ideas from the Rainbow DQN paper to improve upon the original DQN paper from Mnih.

# Notes:
# (i) the training this time will be time-consuming! You should start coding pretty fast so that you will have enough time to train your agent.
# (ii) in order to get good results, the replay buffer for DQN needs to be pretty large. You will need a machine with 8GB RAM at least.
# (iii) You will need many training iterations until your model picks up anything useful. If in the beginning your model does not appear to learn, this might be the reason.



# For transforming uint8 frames to float32 tensors
def transform(x):
    mean = torch.Tensor([0.1])
    std = torch.Tensor([255*0.2])
    x = torch.Tensor(np.array(x)).to(device)
    x = torchvision.transforms.Normalize(mean = mean, std = std)(x)
    return x


def get_epsilon_action(qnet, state, epsilon, nr_actions):
    if random.uniform(0.0, 1.0) < epsilon:
        action = random.randrange(nr_actions)
    else:
        qvals = qnet.forward(state)
        action = torch.argmax(qvals).item()
    return action


# Named tuple for storing experience steps gathered in training
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool), np.array(next_states))

def reset(env):
    env.reset()
    state, _, done, truncated, _ = env.step(env.action_space.sample())
    if done or truncated:
        return reset(env)
    return np.array(state)


def DQN(qnet, env, optimizer, epsilon=0.1, gamma=1.0, nr_episodes=5000, max_t=100, 
        replay_buffer_size=1000000, batch_size=32, warm_start_steps=1000, sync_rate=1024, train_frequency=8):
    
    print(f"Train policy with DQN for {nr_episodes} episodes using at most {max_t} steps, gamma = {gamma}, epsilon = {epsilon}, replay buffer size = {replay_buffer_size}, sync rate = {sync_rate}, warm starting steps for filling the replay buffer = {warm_start_steps}")

    target_qnet = copy.deepcopy(qnet)
    buffer = ReplayBuffer(replay_buffer_size)
    nr_actions = env.action_space.n
    episode_returns = []
    episode_lengths = []
    nr_terminal_states = []

    # populate buffer
    state = reset(env)
    for i in range(warm_start_steps):
        action = random.randrange(nr_actions)
        new_state, reward, done, truncated, _ = env.step(action)
        exp = Experience(state, action, reward, done, new_state)
        buffer.append(exp)
        state = new_state
        if done or truncated:
            state = reset(env)
        if i % 10 == 0:
            state = reset(env)

    step_counter = 0
    with tqdm.trange(nr_episodes, desc='DQN Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            state = reset(env)
            episode_return = 0.0

            # Collect trajectory
            for t in range(max_t):
                step_counter = step_counter + 1

                # step through environment with agent
                with torch.no_grad():
                    action = get_epsilon_action(qnet, transform(state), epsilon, nr_actions)

                new_state, reward, done, truncated, _ = env.step(action)
                buffer.append(Experience(np.array(state), action, reward, done, new_state))
                state = new_state
                episode_return += (gamma ** t) * reward

                # calculate training loss on sampled batch
                if step_counter % train_frequency == 0:
                    states, actions, rewards, dones, next_states = buffer.sample(batch_size)

                    qvalues = qnet.forward(transform(states))
                    qvalues = torch.gather(qvalues.squeeze(0),1,torch.from_numpy(actions).to(device).unsqueeze(-1)).squeeze(1)

                    with torch.no_grad():
                        next_qvalues = target_qnet(transform(next_states))
                        next_qvalues, _ = torch.max(next_qvalues.squeeze(0), dim=1)
                        next_qvalues[dones] = 0.0
                        next_qvalues = next_qvalues.detach()
                        nr_terminal_states.append(dones.sum())

                    expected_qvalues = gamma * next_qvalues + torch.Tensor(rewards).to(device)
                    loss = nn.MSELoss()(qvalues, expected_qvalues)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Soft update of target network
                if step_counter % sync_rate == 0:
                    target_qnet.load_state_dict(qnet.state_dict())

                if done:
                    break

            episode_lengths.append(t+1)
            episode_returns.append(episode_return)

            # if e % 200 == 0:
            #     gym_video(qnet, env, filename=f"dqn_breakout_{e}", nr_steps=1000)

            tepisodes.set_postfix({
                'mean episode reward': "{:3.2f}".format(np.mean(episode_returns[-25:])),
                'mean episode length': "{:3.2f}".format(np.mean(episode_lengths[-25:])),
                'nr terminal states in batch': "{:3.2f}".format(np.mean(nr_terminal_states[-25:])),
                'global step': step_counter,
            })


# network: Use a small CNN of the following type:
# First, use 3 layers of 2D-convolutions with filter sizes 8x8, 4x4 and 3x3 and strides 4, 2, and 1.
# The number of filters is supposed to be 32, 64 and 64.
# Then, use 2 fully connected layers with hidden sizes 1024 and #actions.
# Use leaky ReLUs with parameter 0.01 in between
# You can also experiment with deeper networks with smaller filters, residual connections and batch norm. Whatever brings benefits!
class Model(nn.Module):
    def __init__(self, nr_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, nr_actions)

        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(torch.flatten(x,-3,-1)), 0.01)
        x = self.fc2(x)
        return x
    
    def act(self, x):
        with torch.no_grad():
            q_values = self.forward(x)
            action = torch.argmax(q_values).item()
            return action, q_values

env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
#env = gym.wrappers.RecordEpisodeStatistics(env)
#env = gym.wrappers.RecordVideo(env, 'video', episode_trigger = lambda x: x % 2 == 0)
#env = NoopResetEnv(env, noop_max=30) # does not work
env = MaxAndSkipEnv(env, skip=4)
env = EpisodicLifeEnv(env)
if "FIRE" in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
env = ClipRewardEnv(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.FrameStack(env, 4)
env = gym.wrappers.RecordVideo(env, f"videos/", episode_trigger=lambda episode: episode % 200 == 0)
#env.seed(seed)
#env.action_space.seed(seed)
#env.observation_space.seed(seed)

print('gym:', gym.__version__)
print('ale_py:', ale_py.__version__)

model = Model(env.action_space.n).to(device)

env.reset()
env.step(0)
env.render()

epsilon = 0.1
nr_episodes = 2000000
max_t = 4000
gamma = 0.99
replay_buffer_size = 200000 # 1M is the DQN paper default

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5 , eps=1.5e-4) 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
DQN(model, env, optimizer, gamma=gamma, epsilon=epsilon, nr_episodes=nr_episodes, max_t=max_t, warm_start_steps=500, sync_rate=128, replay_buffer_size=replay_buffer_size, train_frequency=8)