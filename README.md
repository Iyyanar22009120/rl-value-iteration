# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.
## VALUE ITERATION ALGORITHM
- Value iteration is a method of computing an optimal MDP policy  and its value.
- It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation.
- The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

The algorithm is as follows:
1. Initialize the value function `V(s)` arbitrarily for all states `s`.
2. Repeat until convergence:
    - Initialize aaction-value function `Q(s, a)` arbitrarily for all states `s` and actions `a`.
    - For all the states s and all the action a of every state:
        - Update the action-value function `Q(s, a)` using the Bellman equation.
        - Take the value function `V(s)` to be the maximum of `Q(s, a)` over all actions `a`.
        - Check if the maximum difference between `Old V` and `new V` is less than `theta`, where theta is a **small positive number** that determines the **accuracy of estimation**.
3. If the maximum difference between Old V and new V is greater than theta, then
    - Update the value function `V` with the **maximum action-value** from `Q`.
    - Go to **step 2**.
4. The optimal policy can be constructed by taking the **argmax** of the action-value function `Q(s, a)` over all actions `a`.
5. Return the optimal policy and the optimal value function.
## VALUE ITERATION FUNCTION
#### Name: NATHIN R
#### Register Number:212222230090
```
envdesc  = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 15
P = env.env.P
```
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = V[s]
            action_values = []
            for a in range(len(P[s])):
                action_value = 0
                for prob, next_state, reward, done in P[s][a]:
                    action_value += prob * (reward + gamma * V[next_state] * (not done))
                action_values.append(action_value)
            V[s] = max(action_values)
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break

    pi = lambda s: np.argmax([sum([p * (r + gamma * V[s_]) for p, s_, r, _ in P[s][a]]) for a in range(len(P[s]))])

    return V, pi
```
## OUTPUT:
#### Optimal Pollicy

#### Optimal Value Function

#### Success Rate

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
