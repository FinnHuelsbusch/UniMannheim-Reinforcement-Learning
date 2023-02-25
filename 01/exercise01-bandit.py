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
total_rewards = []
for seed in seeds: 
    env.seed(seed)
    observation = env.reset()

    rewards = []
    average_rewards = np.zeros(env.action_space.n)
    nr_steps_per_action = np.zeros(env.action_space.n)
    # use the fact that we know our states
    action_reward_dict = {v: [0] for v in range(10)}
    propabilety_to_explore = 1
    discount_factor = 0.9999
    best_action = []
    propabilety = []
    for i_episode in range(5000):
    
        print("episode Number is", i_episode)   
        
        if np.random.random() < propabilety_to_explore:
            action = env.action_space.sample()
        else: 
            # use median instead of average or max in order to ignore outliers
            action = max(action_reward_dict.items(), key=lambda k: statistics.median(k[1]))[0]


        # here we taking the next "step" in our environment by taking in our action variable randomly selected above
        observation, reward, done, info = env.step(action) 
        rewards.append(reward)
        action_reward = action_reward_dict.get(action, [])
        action_reward.append(reward)
        action_reward_dict[action] = action_reward

        best_action.append(max(action_reward_dict.items(), key=lambda k: statistics.median(k[1]))[0])
        propabilety.append(propabilety_to_explore)

        propabilety_to_explore = propabilety_to_explore * discount_factor   

        print("observation space is: ",observation)
        print("reward variable is: ",reward)
        print("done flag is: ",done)
        print("info variable is: ",info)

    print("sum of rewards: " + str(np.sum(rewards)))
    total_rewards.append(int(np.sum(rewards)))

    # plt.plot(rewards)
    # plt.ylabel('rewards')
    # plt.xlabel('steps')
    # plt.show()

    # fig, ax1 = plt.subplots() 
    # ax1.set_xlabel('steps') 
    # ax1.set_ylabel('action', color = 'red') 
    # ax1.plot(best_action, color = 'red') 
    # ax1.tick_params(axis ='y', labelcolor = 'red') 
    # # Adding Twin Axes
    # ax2 = ax1.twinx() 
    # ax2.set_ylabel('Propabilety to explore', color = 'blue') 
    # ax2.plot(propabilety, color = 'blue') 
    # ax2.tick_params(axis ='y', labelcolor = 'blue') 
    # plt.show()
print(total_rewards)