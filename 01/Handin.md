# Task 4 

For the bandit we tried the following approaches. First we tested random as a baseline this resulted in a mean score of 2228 and  median score of 2207 tested with 30 seeds. The next approach was to explore for a set amount of iterations and exploit in all other iterations. On the same 30 seeds with 1000 exploration steps this  approach yielded a mean score of 6668 an a median score of 6646. The submitted approach is $\epsilon$​ -greedy. For the probability of exploration we used the following function: 
$$
e^{-\frac{iteration}{2000}}
$$

The factor 2000 was found by testing different factors on the same 30 seeds. With the factor of 2000 we archived an mean score of 7841 and a median score of 7795. 
