import numpy as np
import gym
import gym_bandits
import matplotlib.pyplot as plt

np.random.seed(42) # make runs deterministic for numpy random number generator

env = gym.make('BanditTenArmedGaussian-v0')

print('observation space:', env.observation_space.n, 'dimensional')
print('action space:', env.action_space.n, 'dimensional')

env.seed(4711)
observation = env.reset()

rewards = []
average_rewards = np.zeros(env.action_space.n)
nr_steps_per_action = np.zeros(env.action_space.n)

action_reward_dict = {}
for i_episode in range(5000):
  
    print("episode Number is", i_episode)   
    
    #action = env.action_space.sample() # sampling the "action" array which in this case only contains 10 "options" because there is 10 bandits
    action = i_episode % env.action_space.n
        
    print("action is", action)
    if i_episode < 1000: 
        action = env.action_space.sample()
    else: 
        help = {k: np.average(v) for k, v in action_reward_dict.items()}
        action = max(help.items(), key=lambda k: k[1])[0]
        
    # here we taking the next "step" in our environment by taking in our action variable randomly selected above
    observation, reward, done, info = env.step(action) 
    rewards.append(reward)
    action_reward = action_reward_dict.get(action, [])
    action_reward.append(reward)
    action_reward_dict[action] = action_reward

    print("observation space is: ",observation)
    print("reward variable is: ",reward)
    print("done flag is: ",done)
    print("info variable is: ",info)

print("sum of rewards: " + str(np.sum(rewards)))

plt.plot(rewards)
plt.ylabel('rewards')
plt.xlabel('steps')
plt.show()