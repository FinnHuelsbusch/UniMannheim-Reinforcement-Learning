import gymnasium as gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import random
from collections import namedtuple

def flatdim(space):
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return int(np.prod([flatdim(s) for s in space.spaces]))
    else:
        RuntimeWarning("space not recognized")

def _flatten(space, x, y):
    if isinstance(space, gym.spaces.Discrete):
        n = flatdim(space)
        y = y * n + x
    elif isinstance(space, gym.spaces.Tuple):
        for x_part, s in zip(x, space.spaces):
            y = _flatten(s, x_part, y)
    else:
        raise NotImplementedError
    return y

def flatten(space, x):
    return _flatten(space, x, 0)

class FlattenedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = flatdim(env.observation_space)
        self.wrapped_observation_space = env.observation_space
        self.observation_space = gym.spaces.Discrete(self.n)
    
    def observation(self, obs):
        return flatten(self.wrapped_observation_space, obs)

def unflatten(space, x):
    pass

def render_FrozenLake(env, policy, filename, max_t=1000):
    frames = []
    state = env.reset()[0]
    frames.append(env.render())
    for t in range(max_t):
        action = policy[state]
        state, _, done, _, _ = env.step(action)
        frames.append(env.render())
        if done:
            break
    
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=1250)
    anim.save(filename, writer='imagemagick', fps=6)

def sample_epsilon_greedy_from_q(q, epsilon, state):
    """
    return random action with probability epsilon and the best action according to q-value otherwis
    """
    if np.random.random() <= epsilon:
        # choose a random action 
        return random.choice(range(len(q[state,])))
    else:
        # get all actions that have the max value in q
        best_actions = np.argwhere(q[state,] == np.amax(q[state,])).flatten()
        # from the best actions choose a random one
        return np.random.choice(best_actions)

def visualize_lake_policy(env, policy):
    # Create a grid to hold the arrow plots
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, env.desc.shape[1])
    ax.set_ylim(0, env.desc.shape[0])
    ax.set_xticks(np.arange(0.5, env.desc.shape[1], 1))
    ax.set_yticks(np.arange(0.5, env.desc.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    
    # Define the arrow colors
    arrow_colors = {
        0: 'blue',   # move left
        1: 'red',    # move down
        2: 'purple', # move right
        3: 'green'   # move up
    }
    
    # Define the state mapping
    state_mapping = {}
    for s in range(env.action_space.n):
        x, y = np.unravel_index(s, env.desc.shape)
        state_mapping[s] = (x, y)
    
    # Plot the arrows and terminal states
    for s in range(env.observation_space.n):
        x, y = np.unravel_index(s, env.desc.shape)
        if env.desc[x][y] == b'H' or env.desc[x][y] == b'G':
            # Terminal state
            cell = plt.Rectangle((y, env.desc.shape[0] - x - 1), width=1, height=1, facecolor='white', edgecolor='black')
            ax.add_artist(cell)
            if env.desc[x][y] == b'H':
                # Hole
                circle = plt.Circle((y + 0.5, env.desc.shape[0] - x - 0.5), radius=0.3, color='black')
                ax.add_artist(circle)
            else:
                # Goal
                circle = plt.Circle((y + 0.5, env.desc.shape[0] - x - 0.5), radius=0.3, color='blue')
                ax.add_artist(circle)
        else:
            # Non-terminal state
            action = policy[s]
            dx, dy = {
                0: (0, -1),   # move left
                1: (1, 0),    # move down
                2: (0, 1),    # move right
                3: (-1, 0)    # move up
            }[action]
            arrow_color = arrow_colors[action]
            cell = plt.Rectangle((y, env.desc.shape[0] - x - 1), width=1, height=1, facecolor='white', edgecolor='black')
            ax.add_artist(cell)
            ax.arrow(y + 0.5, env.desc.shape[0] - x - 0.5, dy * 0.3, -dx * 0.3, head_width=0.2, head_length=0.2, color=arrow_color)
    
    plt.show()

def MCOffPolicyControl(env, epsilon=0.1, nr_episodes=5000, max_t=1000, gamma=0.99):
    """
    MC-based off-policy control using weighted importance sampling
    """
    nr_actions = env.action_space.n
    nr_states = env.observation_space.n
    
    q = np.full((nr_states, nr_actions), 0.0, dtype=np.float32)
    c = np.full((nr_states, nr_actions), 0.0, dtype=np.float32)
    pi = np.zeros(nr_states)

    SAR = namedtuple('SAR', ['state', 'action', 'reward'])
    episode_returns = []
    episode_lengths = []
    backtrack_percentages = []

    with tqdm.trange(nr_episodes, desc='Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            trajectory = []
            # generate trajectory
            state = env.reset()[0]
            for t in range(max_t):
                action = sample_epsilon_greedy_from_q(q, epsilon, state)
                observation, reward, terminated, truncated, info  = env.step(action)
                trajectory.append((state, action, reward))
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
            g = 0 # running return
            w = 1 # running importance sampling ratio
            for state, action, reward in reversed(trajectory):
                g = gamma * g + reward
                c[state, action] = c[state, action] + w
                # update q 
                q[state, action] = q[state, action] + (w / c[state, action]) * (g- q[state, action])
                pi[state] = np.argmax(q[state, ])
                if action != pi[state]:
                    break
                else: 
                    w = w / (1-epsilon)
            
            # print average return of the last 100 episodes
            if(e % 100 == 0):
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                avg_backtrack_percentage = np.mean(backtrack_percentages[-100:])
                tepisodes.set_postfix({
                'episode return': "{:.2f}".format(avg_return),
                'episode length': "{:3.2f}".format(avg_length),
                'backtrack': "{:.2f}%".format(avg_backtrack_percentage)
                })
    visualize_lake_policy(env, pi)
    return np.argmax(q, 1)


def SARSA(env, epsilon=0.1, alpha=0.01, nr_episodes=50000, max_t=1000, gamma=0.99):
    """
    On-policy SARSA with epsilon-greedy policy
    """
    nr_actions = env.action_space.n
    nr_states = env.observation_space.n

    # SARSA usees an epsilon-greedy policy
    # The underlying deterministic policy is derived from the q-values
    q = np.full((nr_states, nr_actions), 0, dtype=np.float32)
    # count how often a state action pair is sampled
    c = np.zeros_like(q, dtype=np.int64)

    # history of episode returns
    episode_returns = [] 
    episode_lengths = []

    # iterate over episodes
    with tqdm.trange(nr_episodes, desc='Training', unit='episodes') as tepisodes:
        for e in tepisodes:
            state = env.reset()[0]
            action = sample_epsilon_greedy_from_q(q, epsilon, state)
            c[state, action] += 1
            rewards = []

            # Collect trajectory
            for t in range(max_t):
                next_state, reward, done, _, _ = env.step(action)
                rewards.append(reward)
                next_action = sample_epsilon_greedy_from_q(q, epsilon, next_state)

                q[state, action] = q[state, action] + alpha * (reward + gamma *q[next_state, next_action] - q[state, action])
                action = next_action 
                state = next_state
                if done: 
                    break; 


            discounts = [gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a, b in zip(discounts, rewards)])
            episode_returns.append(R)
            episode_lengths.append(len(rewards))

            # print average return of the last 100 episodes
            if(e % 100 == 0):
                avg_return = np.mean(episode_returns[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                tepisodes.set_postfix({
                'episode return': avg_return,
                'episode length': avg_length
                })

    visualize_lake_policy(env, np.argmax(q, 1))   
    return np.argmax(q, 1)

def evaluate_greedy_policy(env, policy, nr_episodes=1000, t_max=1000):
    reward_sums = []
    for t in range(nr_episodes):
        state = env.reset()[0]
        rewards = []
        for i in range(t_max):
            action = policy[state]
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
    
        reward_sums.append(np.sum(rewards))
    
    return np.mean(reward_sums)


env_frozenlake_small = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")
env_frozenlake_small_slippery = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="rgb_array")
env_frozenlake_medium = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="rgb_array")
env_frozenlake_medium_slippery = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode="rgb_array")
env_blackjack = FlattenedObservationWrapper(gym.make('Blackjack-v1', render_mode="rgb_array"))

epsilon = 0.1
alpha = 0.1
nr_episodes = 10000
max_t = 400
gamma = 0.9


# below are some default parameters for the control algorithms. You might want to tune them to achieve better results.
for env, name in {
    env_frozenlake_small: "frozenlake_small",
    env_frozenlake_small_slippery: "frozenlake_small_slippery",
    env_frozenlake_medium: "frozenlake_medium",
    env_frozenlake_medium_slippery: "frozenlake_medium_slippery",
    }.items():
    MC_policy = MCOffPolicyControl(env, epsilon=epsilon, nr_episodes=nr_episodes, max_t=max_t, gamma=gamma)
    print("Mean episode reward from MC trained policy on", name, ": ", evaluate_greedy_policy(env, MC_policy))
    render_FrozenLake(env, MC_policy, name + "_MC.gif", max_t=100)

    SARSA_policy = SARSA(env, epsilon=epsilon, alpha=alpha, nr_episodes=nr_episodes, max_t=max_t, gamma=gamma)
    print("Mean episode reward from SARSA trained policy on", name, ": ", evaluate_greedy_policy(env, SARSA_policy))
    render_FrozenLake(env, SARSA_policy, name + "_SARSA.gif", max_t=max_t)


# MC_blackjack_policy = MCOffPolicyControl(env_blackjack, epsilon=0.051, nr_episodes=10000, max_t=1000, gamma=0.99)
# print("Mean episode reward from MC trained policy on BlackJack: ", evaluate_greedy_policy(env_blackjack, MC_blackjack_policy))

# SARSA_blackjack_policy = SARSA(env_blackjack, alpha=0.1, epsilon=0.051, nr_episodes=10000, max_t=1000, gamma=0.99)
# print("Mean episode reward from SARSA trained policy on BlackJack: ", evaluate_greedy_policy(env_blackjack, SARSA_blackjack_policy))