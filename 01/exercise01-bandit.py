import numpy as np
import gym
import gym_bandits
import matplotlib.pyplot as plt

np.random.seed(351791) # make runs deterministic for numpy random number generator

env = gym.make('BanditTenArmedGaussian-v0')

print('observation space:', env.observation_space.n, 'dimensional')
print('action space:', env.action_space.n, 'dimensional')

env.seed(351791) # make each run the same 
observation = env.reset()

rewards = []
average_rewards = np.zeros(env.action_space.n)
nr_steps_per_action = np.zeros(env.action_space.n)

steps = 5000
for i_episode in range(steps):

    print("episode Number is", i_episode)
       
    ### your code goes here ###
    # We have to trade-off Exploitation and Exploration.
    # For that we consider: 1) The expected best (action) value and 2) the remaining iterations
    # We would like to explore early and exploit later in the episode
    # Since all bandits are Gaussian, we estimate the Gaussian parameters μ using the MLE (sample mean)
    # Our final solution uses a 'probability_to_explore' which changes (over time) depending on the #of iterations (explore early, exploit later)
    # Exploration: Sample from bandit randomly
    # Exploitation: Pick action with highest expected value

    def update_exected_value(action):
        average_rewards[action] = (average_rewards[action] * (nr_steps_per_action[action]) + rewards[-1]) / (nr_steps_per_action[action] + 1)
        nr_steps_per_action[action]+=1

    # prob of exploration
    k = 300 # exploration prob. decay factor
    probability_to_explore = np.exp(-i_episode/(steps/k))

    # choose an action
    exploring = np.random.binomial(n=1,p=probability_to_explore) == 1
        
    if exploring:
        action = np.random.randint(0,10)
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
    print()

print("sum of rewards: " + str(np.sum(rewards)))
plt.plot(rewards)
plt.ylabel('rewards')
plt.xlabel('steps')
plt.show()

summed_rewards = []
for i in range(len(rewards)): 
    summed_rewards.append(np.sum(rewards[0:i+1]))
plt.plot(summed_rewards)
plt.ylabel('summed_rewards')
plt.xlabel('steps')
plt.show()
