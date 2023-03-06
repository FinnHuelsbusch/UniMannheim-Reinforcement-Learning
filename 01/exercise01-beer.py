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

# For all first three questions: after computing state values always print them out!

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
print("------")
######################
######################

# Question 2:
# compute the state values of the 50/50 policy by Richardson iteration.
# Let the iteration run so long as the nomr of the state values vector does not change by more than 0.001

### your code here ###
accuracy_threshold = 0.001

def richardson_iteration():
    """
    This is the Richardson Algorithm to iteratively estimate the state value functions. 
    This implementation is asynchronous and in-place.
    """
    v_current = np.zeros(8)
    terminate = False
    
    iteration = 1
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
            
        print(f"Richardson Iteration {iteration}:")
        print(v_current)
        iteration+=1
            
    return v_current

estimated_state_values = richardson_iteration()
print("Richardson Iteration result:", estimated_state_values)
print("------")
######################

# Question 3:
# compute the optimal state values by dynamic programming
# Determine the number of iterations as for the Richardson iteration
### your code here ###
def value_iteration():
    """
    By assumption, this value iteration alg. implementation returns the action "up" as the best action to take
    when there is only one possible action to choose from (i.e. 'Li' and 'F', 'End') or when both actions "down" and "up" are equally good.
    """
    actions = ("up", "down")
    policy = [None]*8
    state_transitions_a = (P_up, P_down)
    terminate = False
    v_current = np.zeros(8)
    
    iteration = 1
    while not terminate:
        delta = np.float64(0)
        
        # policy improvement
        for s_i, v_cur in enumerate(v_current):
            v_old = v_cur
            
            # find optimal policy            
            q_values = []
            for a_i, _ in enumerate(actions):
                q_values.append(r[a_i][s_i] + gamma*np.sum(state_transitions_a[a_i][s_i] @ v_current))
            
            q_best = np.argmax(q_values)
            v_current[s_i] = q_values[q_best]
            policy[s_i] = actions[q_best]
            
            delta = np.max([delta, np.abs(v_current[s_i] - v_old)]) 

        # terminate if error small enough
        if delta < accuracy_threshold:
            terminate = True

        print(f"Value Iteration {iteration}:")
        print(v_current)
        for i, s in enumerate(['Start', 'A'  , 'LÖ' , 'G'  , 'B'  , 'Li'  , 'F'  , 'End']):
            print(s, policy[i], end=", ")
        print()  
        iteration+=1
    return (v_current, policy)

# Value iteration - optimal policy
_, policy = value_iteration()
print("Value Iteration optimal policy:")
for i, s in enumerate(['Start', 'A'  , 'LÖ' , 'G'  , 'B'  , 'Li'  , 'F'  , 'End']):
    print(s, policy[i], end=", ")
print("\n------")
######################