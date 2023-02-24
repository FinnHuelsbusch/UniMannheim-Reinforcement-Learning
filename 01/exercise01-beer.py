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
# compute state value for always choosing up. 
state_value_function_up = np.linalg.inv(np.identity(8) - 0.9 * P_up) * r[0]
# compute state value for always choosing down.
state_value_function_down = np.linalg.inv(np.identity(8) - 0.9 * P_down) * r[1]
# compute the combined state_value_function
state_value_function= 0.5 * state_value_function_up + 0.5 * state_value_function_down
print(state_value_function)
# compute state transitions for the 50/50 policy
...
# compute expected rewards for the 50/50 policy
...
# compute state values
...
######################

# Question 2:
# compute the state values of the 50/50 policy by Richardson iteration.
# Let the iteration run so long as the nomr of the state values vector does not change by more than 0.001

### your code here ###
...
######################

# Question 3:
# compute the optimal state values by dynamic programming
# Determine the number of iterations as for the Richardson iteration

### your code here ###
...
######################