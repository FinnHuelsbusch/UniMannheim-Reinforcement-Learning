import numpy as np
import gym
import gym_bandits
import matplotlib.pyplot as plt

np.random.seed(42) # make runs deterministic for numpy random number generator

env = gym.make('BanditTenArmedGaussian-v0')

print('observation space:', env.observation_space.n, 'dimensional')
print('action space:', env.action_space.n, 'dimensional')

env.seed(34) # make each run the same 
observation = env.reset()

rewards = []
average_rewards = np.zeros(env.action_space.n)
nr_steps_per_action = np.zeros(env.action_space.n)

steps = 2000
k = 30 # exploration prob. decay factor: can be optimized
for i_episode in range(steps):
  
    print("episode Number is", i_episode)   
    
    #action = env.action_space.sample() # sampling the "action" array which in this case only contains 10 "options" because there is 10 bandits
    #action = i_episode % env.action_space.n
    #print("action is", action)

    ### your code goes here ###
    # We have to trade-off Exploitation and Exploration.
    # For that we consider: 1) The expected best (action) value, 2) The uncertainty of all (action) values, 3) the remaining iterations
    # We would like to explore early and exploit later in the episode
    # Since all bandits are Gaussian, we estimate the Gaussian parameters (μ,σ) using the MLE (sample mean & sample variance)
    # Idea: q differs depending on the #of iterations (explore early, exploit later)
    # Exploration: Sample from bandit with highest variance (what if variance differs???)
    # Exploitation: Pick action with highest expected value

    def update_exected_value(action):
        nr_steps_per_action[action]+=1
        average_rewards[action] = (average_rewards[action] * (nr_steps_per_action[action]-1) + rewards[-1]) / nr_steps_per_action[action]

    # prob of exploration
    probability_to_explore = np.exp(-i_episode/(steps/k))

    # choose an action
    exploring = np.random.binomial(n=1,p=probability_to_explore) == 1
        
    if exploring:
        action = env.action_space.sample()
        print("here")
    else:
        action = np.argmax(average_rewards)
        
    # here we taking the next "step" in our environment by taking in our action variable randomly selected above
    observation, reward, done, info = env.step(action) 
    rewards.append(reward)

    update_exected_value(action)
    print("action", action)

    #print("observation space is: ",observation)
    print("reward variable is: ",reward)
    #print("done flag is: ",done)
    #print("info variable is: ",info)

print("sum of rewards: " + str(np.sum(rewards)))

plt.plot(rewards)
plt.ylabel('rewards')
plt.xlabel('steps')
plt.show()