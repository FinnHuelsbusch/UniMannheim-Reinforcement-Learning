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
expected_reward = r[0]*pi_up + r[1]*pi_down
print(expected_reward)

# compute state values
state_values = np.linalg.inv(np.identity(8) - gamma*state_transitions) @ expected_reward
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

delta = 0.001
terminate = False
iterations = 0
while not terminate: 
    iterations += 1
    Delta = np.float64(0) 
    # reversed in order to terminate in one iteration. 
    for state_index, s in reversed(list(enumerate(v))): # sweep through space by traversing from end to start state
        v_old = s
        # Up
        inner_sum = 0
        for next_state_index, next_state_prob in enumerate(P_up[state_index]):
            inner_sum += next_state_prob*v[next_state_index]
        s = 0.5 * (r[0][state_index] + gamma * inner_sum)
        # Down 
        inner_sum = 0
        for next_state_index, next_state_prob in enumerate(P_down[state_index]):
            inner_sum += next_state_prob*v[next_state_index]
        s += 0.5 * (r[1][state_index] + gamma * inner_sum )
        v[state_index] = s
        Delta = np.max([Delta, np.abs(s - v_old)])
        print(state_index, s, v, v_old)
        # terminate?
        if Delta < delta:
            terminate = True
print(v)
print(f"Finished task 2 (Richardson Iteration) in {iterations} iterations.")
######################
print()
print()
# Question 3:
# compute the optimal state values by dynamic programming
# Determine the number of iterations as for the Richardson iteration
v = np.zeros(8)
terminate = False
iterations = 0
while not terminate: 
    iterations += 1
    Delta = np.float64(0) 
    # reversed in order to terminate in one iteration. 
    for state_index, s in reversed(list(enumerate(v))): # sweep through space by traversing from end to start state
        v_old = s
        # Up
        inner_sum = 0
        for next_state_index, next_state_prob in enumerate(P_up[state_index]):
            inner_sum += next_state_prob*v[next_state_index]
        value_up =  (r[0][state_index] + gamma * inner_sum)
        # Down 
        inner_sum = 0
        for next_state_index, next_state_prob in enumerate(P_down[state_index]):
            inner_sum += next_state_prob*v[next_state_index]
        value_down =  (r[1][state_index] + gamma * inner_sum )
        s = max(value_up, value_down)
        v[state_index] = s
        Delta = np.max([Delta, np.abs(s - v_old)])
        print(state_index, s, v, v_old)
    # terminate?
    if Delta < delta:
        terminate = True
print(v)
print(f"Finished task 3 (Dynamic Programming) in {iterations} iterations.")
# Compute the optimal policy 
policy = [None] * 8 
for state_index, s in reversed(list(enumerate(v))): # sweep through space by traversing from end to start state
    inner_sum = np.float64('-inf')
    for next_state_index, next_state_prob in enumerate(P_up[state_index]):
        test = next_state_prob*v[next_state_index]
        if test != 0: 
            if np.isinf(inner_sum):
                inner_sum = test
            else: 
                inner_sum += test
    value_up =  (r[0][state_index] + gamma * inner_sum)
    inner_sum = np.float64('-inf')
    for next_state_index, next_state_prob in enumerate(P_down[state_index]):
        test = next_state_prob*v[next_state_index] 
        if test != 0 : 
            if np.isinf(inner_sum):
                inner_sum = test
            else: 
                inner_sum += test
        
    value_down =  (r[1][state_index] + gamma * inner_sum )
    if value_up <= value_down:
        policy[state_index] = 'down'
    else: 
        policy[state_index] = 'up'

print("The optimal policy computed by in task 3 is:")
for index, element in enumerate(['Start', 'A'  , 'LÖ' , 'G'  , 'B'  , 'Li'  , 'F'  , 'End']):
    print(element, policy[index])

######################