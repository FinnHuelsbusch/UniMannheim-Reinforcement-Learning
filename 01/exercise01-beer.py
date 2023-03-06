import numpy as np


# Start: Home
# Auld Triangle
# Lötlampe
# Globetrotter
# Black Sheep
# Limericks
# Fat Louis
# End: Home

gamma = 0.9 # discount factor

# it is assumed, that bot actions (up and down) can be executed in the states Li and F. 
# Tough both actions in both states will lead to the same result state End. 
# In order to model this, both actions are set with the same reward. 
r = np.array((
    [-3,-2,-3,-4,-5, -6, -7, 0], # reward of going up
    [-1,-4,-5,-5,-6, -6, -7, 0] # reward when going down
    ))

# state transition matrix for always going up
P_up = np.array ((
    # Start, A  , LÖ , G  , B  , Li  , F  , End
    [ 0    , 1  , 0  , 0  , 0  ,  0  , 0  , 0  ],  # Start # 
    [ 0    , 0  , 0  , 1  , 0  ,  0  , 0  , 0  ],  # A     # 
    [ 0    , 0  , 0  , 1  , 0  ,  0  , 0  , 0  ],  # Lö    # 
    [ 0    , 0  , 0  , 0  , 0  ,  1  , 0  , 0  ],  # G     # 
    [ 0    , 0  , 0  , 0  , 0  ,  1  , 0  , 0  ],  # B     #  
    [ 0    , 0  , 0  , 0  , 0  ,  0  , 0  , 1  ],  # Li    # 
    [ 0    , 0  , 0  , 0  , 0  ,  0  , 0  , 1  ],  # F     # 
    [ 0    , 0  , 0  , 0  , 0  ,  0  , 0  , 1  ],  # End   # 
)) 

# state transition matrix for always going down
P_down = np.array((
    # Start, A  , LÖ , G  , B  , Li  , F  , End
    [ 0    , 0  , 1  , 0  , 0  ,  0  , 0  , 0  ], # Start # 
    [ 0    , 0  , 0  , 0  , 1  ,  0  , 0  , 0  ], # A     #  
    [ 0    , 0  , 0  , 0  , 1  ,  0  , 0  , 0  ], # Lö    #  
    [ 0    , 0  , 0  , 0  , 0  ,  0  , 1  , 0  ], # G     #  
    [ 0    , 0  , 0  , 0  , 0  ,  0  , 1  , 0  ], # B     #    
    [ 0    , 0  , 0  , 0  , 0  ,  0  , 0  , 1  ], # Li    #  
    [ 0    , 0  , 0  , 0  , 0  ,  0  , 0  , 1  ], # F     #   
    [ 0    , 0  , 0  , 0  , 0  ,  0  , 0  , 1  ], # End   #   
))


# Question 1:
# compute the state value by matrix inversion

### your code here ###
pi_up = pi_down = 0.5

# compute state transitions for the 50/50 policy
state_transitions = P_up*pi_up + P_down*pi_down
print(state_transitions)

# compute expected rewards for the 50/50 policy
expected_rewards = r[0]*pi_up + r[1]*pi_down
print(expected_rewards)

# compute state values
state_values = np.linalg.inv(np.identity(8) - gamma*state_transitions) @ expected_rewards
print(state_values)

######################
######################

### your code here ###
pi_up = pi_down = 0.5

# compute state transitions for the 50/50 policy
state_transitions = P_up*pi_up + P_down*pi_down
print(state_transitions)
# compute expected rewards for the 50/50 policy
expected_reward = r[0]*pi_up + r[1]*pi_down
print(expected_reward)

# compute state values
state_values = np.linalg.inv(np.identity(8) - gamma*state_transitions) @ expected_reward
print(state_values)
######################
print()
print()
# Question 2:
# compute the state values of the 50/50 policy by Richardson iteration.
# Let the iteration run so long as the nomr of the state values vector does not change by more than 0.001
v = np.zeros(8)

### your code here ###
accuracy_threshold = 0.001

def richardson_iteration(state_transitions, expected_rewards, gamma):
    """
    This is the Richardson Algorithm to iteratively estimate the state value functions. 
    This implementation is asynchronous and in-place.
    
    state_transitions: Our state transition matrix
    expected_rewards: Our expected rewards vector
    gamma: Our gamma factor
    """
    v_current = np.zeros(8)
    terminate = False
    
    while not terminate:
        delta = np.float64(0)
        
        # policy improvement
        # we can solve this in one sweep if we loop over the states in reversed order: reversed(list(enumerate(v_current)))
        for s_i, v_cur in enumerate(v_current):
            v_old = v_cur
            v_current[s_i] = expected_rewards[s_i] + np.sum(gamma * state_transitions[s_i] @ v_current) # v_new
            delta = np.max([delta, np.abs(v_current[s_i] - v_old)]) 
                        
        # terminate if error small enough
        if delta < accuracy_threshold:
            terminate = True
            
    return v_current

estimated_state_values = richardson_iteration(state_transitions=state_transitions, expected_rewards=expected_rewards, gamma=gamma)
print(estimated_state_values)
######################
print()
print()
# Question 3:
# compute the optimal state values by dynamic programming
# Determine the number of iterations as for the Richardson iteration
### your code here ###
def value_iteration(state_transitions_a, rewards_a, gamma):
    """
    By assumption, this value iteration algorithm returns action "up" as the best action to take
    when both actions "down" and "up" are equally good and when we can only choose one action (i.e. 'Li' and 'F', 'End')
    
    state_transitions: Our state transition matrix
    expected_rewards: Our expected rewards vector
    gamma: Our gamma factor
    """
    v_current = np.zeros(8)
    policy = [None]*8
    terminate = False
    actions = ("up", "down")
    iteration = 0
    while not terminate:
        iteration += 1
        delta = np.float64(0)
        
        # policy improvement
        for s_i, v_cur in enumerate(v_current):
            v_old = v_cur
            
            # find optimal policy            
            q_values = []
            for a_i, _ in enumerate(actions):
                q_values.append(rewards_a[a_i][s_i] + gamma*np.sum(state_transitions_a[a_i][s_i] @ v_current))
            
            q_best = np.argmax(q_values)
            v_current[s_i] = q_values[q_best]
            policy[s_i] = actions[q_best]
            
            delta = np.max([delta, np.abs(v_current[s_i] - v_old)]) 
        print(f"Current valuefunction after iteration {iteration}: {v_current}")
        # terminate if error small enough
        if delta < accuracy_threshold:
            terminate = True
    
    # optimal policy
    for i, s in enumerate(['Start', 'A'  , 'LÖ' , 'G'  , 'B'  , 'Li'  , 'F'  , 'End']):
        print(s, policy[i])
              
    return v_current

v = value_iteration(state_transitions_a=(P_up,P_down), rewards_a=r, gamma=gamma)
print(v)
######################