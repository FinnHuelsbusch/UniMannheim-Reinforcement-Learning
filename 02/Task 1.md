# Task 1 

## a 

We were not successful in proving the unweighted importance sampling is unbiased. Although we show our approaches below: 
$$
\begin{align}
\mathbb{E}_b\left[\frac{\sum_{k \in T(s) p_k-\tau(k)} g_k}{\left|T\left(s_k\right)\right|}\right]=\frac{1}{\left|T\left(s_1\right)\right|} \sum_{\left.k \in S_{(s)}\right)} \mathbb{E}_b\left[p_{k T(k)} g_k\right] \\

= \frac{1}{\left|T\left(s_1\right)\right|} \sum_{\left.k \in S_{(s)}\right)} \mathbb{E}_b\left[p_{k T(k)} g_k\right]
\end{align}
$$


## c 

Weighted importance sampling is biased when only a small number of trajectories is sampled. In this case the state value can be dominated by a small number of samples. A example is the following scenario: 

Take 3 states:  $s_1$ the start state, $s_2$ a terminal  state to the right and  $s_3$ a state to the left of $s_1$ which is also a terminal state. The actions for $s_1$ are left ($a_{left}$) with reward $-10$ and right ($a_{right}$) with reward $10$. 

For the behavior policy $b(a_k | s_k)$ the probability of going left in $s_1$ is $95\%$ and right is $5\%$. For $\pi(a_k | s_k)$ it is the opposite. 

Given one sampled trajectory which denotes as follows: $ s_1 \rightarrow  a_{left} \rightarrow s_3 $ 
The state value would be estimated as follows: 
$$
\begin{align}
V(s) &\doteq \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t: T(t)-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t: T(t)-1}}\\
&= \frac{\sum_{t \in \mathcal{T}(s)} \frac{\prod_{i=k}^T \pi\left(a_k \mid s_k\right)}{\prod_{i=k}^T b\left(a_k \mid s_k\right)} G_t}{\sum_{t \in \mathcal{T}(s)} \frac{\prod_{i=k}^T \pi\left(a_k \mid s_k\right)}{\prod_{i=k}^T b\left(a_k \mid s_k\right)}}\\
&= G_t
\end{align}
$$
The fractions for one trajectory can be canceled out thus only $G_t$ remains. In this case $G_t$ is the unbiased estimation of the state value according to the behavior policy. Which we also use to estimate the state value of our policy $\pi$ which is the bias.  
In our example the estimated state value of the $s_1$ would be $-10 = \hat v_\pi(s_1)$ This is far from the true state value of $v_\pi(s_1)$ is: $0,95 \cdot 10 +0,05 \cdot -10 = 9 = v_\pi(s_1)$ If we sample more trajectories the bias converges asymptotically to zero.

