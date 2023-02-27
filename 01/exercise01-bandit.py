import numpy as np
import gym
import gym_bandits
import matplotlib.pyplot as plt
import statistics

np.random.seed(42) # make runs deterministic for numpy random number generator
env = gym.make('BanditTenArmedGaussian-v0')

print('observation space:', env.observation_space.n, 'dimensional')
print('action space:', env.action_space.n, 'dimensional')
seeds = [42,4711,11,22,33,44,55]
# structure to save total rewards of the different seeds
total_rewards = []
for seed in seeds: 
    env.seed(seed)
    observation = env.reset()
    rewards = []
    # use the fact that we know our states
    action_reward_dict = {v: [0] for v in range(10)}
    probability_to_explore = 0.9
    discount_factor = 0.975
    # save the best action over time
    best_action = []
    # save the probabilety to explore over time
    probability = []
    for i_episode in range(5000):
        print("episode Number is", i_episode)   
        # choos if the agent is explored randomly
        if np.random.random() < probability_to_explore:
            action = np.random.randint(0,10)
        else: 
            # use the maximum median instead of average or max in order to ignore outliers
            action = max(action_reward_dict.items(), key=lambda k: statistics.median(k[1]))[0]
        # here we taking the next "step" in our environment by taking in our action variable randomly selected above
        observation, reward, done, info = env.step(action) 
        rewards.append(reward)
        action_reward = action_reward_dict.get(action, [])
        action_reward.append(reward)
        action_reward_dict[action] = action_reward
        # get the best action and append it for plotting
        best_action.append(max(action_reward_dict.items(), key=lambda k: statistics.median(k[1]))[0])
        # save the probability of exploreing
        probability.append(probability_to_explore)
        # discount the probabilety of exploreing
        probability_to_explore = probability_to_explore * discount_factor   
        print("observation space is: ",observation)
        print("reward variable is: ",reward)
        print("done flag is: ",done)
        print("info variable is: ",info)

    print("sum of rewards: " + str(np.sum(rewards)))
    total_rewards.append(int(np.sum(rewards)))

    plt.plot(rewards)
    plt.ylabel('rewards')
    plt.xlabel('steps')
    plt.show()

    fig, ax1 = plt.subplots() 
    ax1.set_xlabel('steps') 
    ax1.set_ylabel('action', color = 'red') 
    ax1.plot(best_action, color = 'red') 
    ax1.tick_params(axis ='y', labelcolor = 'red') 
    # Adding Twin Axes
    ax2 = ax1.twinx() 
    ax2.set_ylabel('probability to explore', color = 'blue') 
    ax2.plot(probability, color = 'blue') 
    ax2.tick_params(axis ='y', labelcolor = 'blue') 
    plt.show()
print(total_rewards)