import collections
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import tqdm
import random
from collections import namedtuple, deque
import copy 

def gym_video(policy, env, filename, nr_steps = 1000):
    """
    Writes a video of policy acting in the environment.
    """
    env = gym.wrappers.RecordVideo(env, video_folder='.', name_prefix=filename)

    state = env.reset()[0]
    done = False
    for t in range(nr_steps):
        action = policy.act_greedy(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        if done:
            break
    env.close()

def evaluate_greedy_policy(env, policy, nr_episodes=1000, t_max=1000):
    reward_sums = []
    for t in range(nr_episodes):
        state = env.reset()[0]
        rewards = []
        for i in range(t_max):
            action = policy.act_greedy(state)
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            if done or truncated:
                break

        reward_sums.append(np.sum(rewards))
    return np.mean(reward_sums)

class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        layers = []
        layers.append(nn.Linear(state_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, action_size))

        self.linear_relu_stack = nn.Sequential(*layers)
        for layer in self.linear_relu_stack: 
            if isinstance(layer, nn.Linear):
                 nn.init.xavier_uniform_(layer.weight)

    def forward(self, state):
        """return q-values and the highest q-value action"""
        x = torch.from_numpy(state)
        q_s = self.linear_relu_stack(x)
        a = torch.argmax(q_s)
        return q_s, a

    def act_epsilon_greedy(self, state, epsilon):
        """ return with probability epsilon a random action and with probability 1-epsilon the greedy action """
        greedy = np.random.choice([False, True], p=[epsilon, 1-epsilon])
        if greedy:
            return self.act_greedy(state)
        else:
            return torch.randint(self.action_size, (1,), dtype=torch.long).item()

    def act_greedy(self, state):
        """ return the greedy action """
        _, action = self.forward(state)
        return action.item()

def MC_func_approx(env, qnet, optimizer, epsilon=0.1, nr_episodes=50000, max_t=1000, gamma=0.99):
    SAR = namedtuple('SAR', ['state', 'action', 'reward'])
    episode_returns = []
    episode_lengths = []

    with tqdm.trange(nr_episodes, desc='Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            trajectory = []
            # generate trajectory
            state = env.reset()[0]

            for t in range(max_t):
                # choose action according to epsilon greedy
                action = qnet.act_epsilon_greedy(state, epsilon)
                # take a step
                observation, reward, terminated, truncated, info = env.step(action)
                trajectory.append(SAR(state, action, reward))
                # update current state 
                state = observation
                # if the environment is in a terminal state stop the sampling
                if terminated:
                    break

            # compute episode reward
            discounts = [gamma ** i for i in range(len(trajectory) + 1)]
            R = sum([a * b for a, (_, _, b) in zip(discounts, trajectory)])
            episode_returns.append(R)
            episode_lengths.append(len(trajectory))

            # update q-values from trajectory
            loss = []
            g = 0
            for state, action, reward in reversed(trajectory):
                g = reward + gamma*g
                q_hat = qnet.forward(state)[0][action]
                mse = F.mse_loss(torch.tensor(g), q_hat)
                loss.append(mse)
                
            optimizer.zero_grad()
            torch.mean(torch.hstack(loss)).backward() # No alpha needed since emp. mean => 1/N = alpha
            optimizer.step()
            
            # print average return of the last 100 episodes
            if(e % 100 == 0):
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                # avg_backtrack_percentage = np.mean(backtrack_percentages[-100:])
                tepisodes.set_postfix({
                'episode return': "{:.2f}".format(avg_return),
                'episode length': "{:3.2f}".format(avg_length),
                # 'backtrack': "{:.2f}%".format(avg_backtrack_percentage)
                })
    return qnet

# Named tuple for storing experience steps gathered in training
RB_Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: RB_Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        samples = random.sample(self.buffer, batch_size)
        for sample in samples: 
            states.append(sample.state)
            actions.append(sample.action)
            rewards.append(sample.reward)
            dones.append(sample.done)
            next_states.append(sample.new_state)

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))

def DQN(qnet, env, optimizer, epsilon=0.1, gamma=1.0, nr_episodes=5000, max_t=100, 
        replay_buffer_size=1000000, batch_size=32, warm_start_steps=1000, sync_rate=1024, train_frequency=8):
    
    print(f"Train policy with DQN for {nr_episodes} episodes using at most {max_t} steps, gamma = {gamma}, epsilon = {epsilon}, replay buffer size = {replay_buffer_size}, sync rate = {sync_rate}, warm starting steps for filling the replay buffer = {warm_start_steps}")

    buffer = ReplayBuffer(replay_buffer_size)
    target_qnet = copy.deepcopy(qnet)
    episode_returns = []
    episode_lengths = []
    nr_terminal_states = []

    # populate buffer
    state = env.reset()[0]
    for i in range(warm_start_steps):
        # your code here: populate the buffer with warm_start_steps experiences #
        # in the warm start just behave randomly 
        action = random.choice(range(env.action_space.n))
        observation, reward, terminated, truncated, info = env.step(action)
        experience = RB_Experience(state, action, reward, terminated, observation)
        buffer.append(experience)
        if terminated or truncated:
            state = env.reset()[0]
        else:
            state = observation

    with tqdm.trange(nr_episodes, desc='DQN Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            state = env.reset()[0]
            episode_return = 0.0
            step_counter = 0
            # Collect trajectory
            for t in range(max_t):
                step_counter = step_counter + 1  
                # step through environment with agent and add experience to buffer
                with torch.no_grad():
                    action = qnet.act_epsilon_greedy(state, epsilon)
                    observation, reward, terminated, truncated, info = env.step(action)
                    experience = RB_Experience(state, action, reward, terminated, observation)
                    buffer.append(experience)
                    episode_return += reward

                # calculate training loss on sampled batch
                if step_counter % train_frequency == 0:
                    states, actions, rewards, dones, next_states = buffer.sample(batch_size)
                    nr_terminal_states.append(np.sum(dones))
                    q_expected = torch.zeros(batch_size)
                    q = torch.zeros(batch_size)
                    for i in range(batch_size): 
                        if dones[i]:
                            q_expected[i] = torch.from_numpy(np.array(rewards[i]))
                        else:
                            qvals_of_target, best_action_according_to_target = target_qnet.forward(next_states[i])
                            best_q = qvals_of_target[best_action_according_to_target]
                            q_expected[i] = rewards[i] + gamma * best_q
                        qvals_of_behaviour, _ = qnet.forward(states[i])
                        qval_of_current_action = qvals_of_behaviour[actions[i]]
                        q[i] = qval_of_current_action
                    loss = F.mse_loss(q,q_expected)
                    # update qnet
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Soft update of target network
                if step_counter % sync_rate == 0:
                    target_qnet.load_state_dict(qnet.state_dict())

                if terminated or truncated:
                    break
                else: 
                    state = observation

            episode_lengths.append(t+1)
            episode_returns.append(episode_return)

            tepisodes.set_postfix({
                'mean episode reward': "{:3.2f}".format(np.mean(episode_returns[-25:])),
                'mean episode length': "{:3.2f}".format(np.mean(episode_lengths[-25:])),
                'nr terminal states in batch': "{:3.2f}".format(np.mean(nr_terminal_states[-25:])),
            })
    return target_qnet

epsilon = 0.1
nr_episodes = 20000
max_t = 400
gamma = 0.9999
replay_buffer_size = 10000

cartpole_env = gym.make('CartPole-v1', render_mode="rgb_array")
cartpole_observation_space_size = cartpole_env.observation_space.shape[0]
cartpole_nr_actions = cartpole_env.action_space.n
cartpole_qnet = QNet(cartpole_observation_space_size, cartpole_nr_actions, 8)
cartpole_optimizer = torch.optim.SGD(cartpole_qnet.parameters(), lr=1e-3)

mountaincar_env = gym.make('MountainCar-v0', render_mode="rgb_array", max_episode_steps = 4000)
mountaincar_observation_space_size = mountaincar_env.observation_space.shape[0]
mountaincar_nr_actions = mountaincar_env.action_space.n
mountaincar_qnet = QNet(mountaincar_observation_space_size, mountaincar_nr_actions, 8)
mountaincar_optimizer = torch.optim.SGD(mountaincar_qnet.parameters(), lr=1e-3)

cartpole_MC = MC_func_approx(cartpole_env, cartpole_qnet, cartpole_optimizer, epsilon, nr_episodes, max_t, gamma)
torch.save(cartpole_MC, './cartpole-MC.model')
gym_video(cartpole_MC, cartpole_env, 'cartpole-MC', 1000)
print(   "Mean episode reward MC on cartpole policy: ",  evaluate_greedy_policy(cartpole_env, cartpole_MC, 10, 4000))


mountaincar_MC = MC_func_approx(mountaincar_env, mountaincar_qnet, mountaincar_optimizer, epsilon, nr_episodes, 4000, gamma)
torch.save(mountaincar_MC, './mountain-car-MC.model')
gym_video(mountaincar_MC, mountaincar_env, 'mountain-cart-MC', 1000)
print(   "Mean episode reward MC on mountaincar policy: ",  evaluate_greedy_policy(mountaincar_env, mountaincar_MC, 10, 4000))

qnet = QNet(cartpole_observation_space_size, cartpole_nr_actions, 8)
#optimizer = torch.optim.SGD(qnet.parameters(), lr=1e-2)
optimizer = torch.optim.RMSprop(qnet.parameters(), lr=0.01)
cartpole_DQN = DQN(
    qnet, 
    cartpole_env, 
    optimizer, 
    gamma=0.9999, 
    epsilon=0.2, 
    nr_episodes=1500, 
    max_t=4000, 
    warm_start_steps=4000, 
    sync_rate=256, 
    replay_buffer_size=5000, 
    train_frequency=8, 
    batch_size=128
)
torch.save(cartpole_DQN, './cartpole-DQN.model')
gym_video(cartpole_DQN, cartpole_env, 'cartpole-DQN', 1000)
print(   "Mean episode reward DQN on cartpole policy: ",  evaluate_greedy_policy(cartpole_env, cartpole_DQN, 10, 4000))

qnet = QNet(mountaincar_observation_space_size, mountaincar_nr_actions, 10)
optimizer = torch.optim.RMSprop(qnet.parameters(), lr=0.01)
mountaincar_DQN = DQN(
    qnet, 
    mountaincar_env, 
    optimizer, 
    gamma=0.9999, 
    epsilon=0.05, 
    nr_episodes=1500, 
    max_t=4000, 
    warm_start_steps=4000, 
    sync_rate=256, 
    replay_buffer_size=5000, 
    train_frequency=8, 
    batch_size=128
)
torch.save(mountaincar_DQN, './mountain-car-DQN.model')
gym_video(mountaincar_DQN, mountaincar_env, 'mountain-cart-DQN', 1000)
print("Mean episode reward DQN on mountaincar policy: ", evaluate_greedy_policy(mountaincar_env, mountaincar_DQN, 10, 4000))