import numpy as np

gamma = 0.9 # discount factor
r = np.array((
    [1,2,3], # reward of going up
    [-1,-2,-3] # reward when going down
    ))

# state transition matrix for always going up
P_up = np.array((
    [ ... ]
    [ ... ]
))

# state transition matrix for always going down
P_down = np.array((
    [ ... ]
    [ ... ]
))

# For all first three questions: after computing state values always print them out!

# Question 1:
# compute the state value by matrix inversion

### your code here ###
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