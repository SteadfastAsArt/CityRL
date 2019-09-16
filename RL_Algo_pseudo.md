
# Dynamic Programming

## Policy Evaluation
Evaluate a given policy π

```pseudocode
function policy_eval(policy, env)
{
	/* policy is a function of state [agent specific], returns action's prob distribution */
	
	Val = dict()  /* value function, of state, returns "value" */
	
	while true  /* set a terminal state when update diff undistinguished */
	{
		for s in env.state  /* to update the whole state set */
		{
			v = 0
			for action, action_prob in policy(s) /* one step lookahead */
			{
				for tran_prob, next_state, reward, done in env.P[s][action]
				{
					v += action_prob * (reward + discount_factor * tran_prob * Val[next_state])
				}
			}
			
			Val[s] = v  /* backup, newly assign | v_k+1 = ... */
		}
		
	}
	
	return Val
}
```

## Policy Iteration
Improve a policy

```pseudocode
function 

```




# Q Learning
Off-policy TD control

```pseudocode
create env
override env.step(), env.
Determine action_space

/* Q-function: A dict of action-value {state_i:[action_space], ..., } */
Init Q, init Q_hat = Q


for e in [1..episode]
{
    /* get initial observation */
    s = env.reset()
    
    for t in [1..T]
    {
        /* *** epsilon-greedy *** (pai is built upon)
         * Sample from policy given certain state s
         */
        _action = sample_from( pai(a|s) )

        /* Actor Do _action and "env" emits reward & new observation 
         * done: whether reach terminal states
         */
        next_state, reward, done, _ = env.step(_action)

        /* *** Experience Memory ***
         * store history and sample batch
         */
        store_to_replay_buffer(s, _action, reward, next_state)
        sample_from_replay_buffer()
        
        /* *** TD Update ***
         * Following the theory of q-learning always gets a better pai',
         * choose the argmax(next_state), compute target
         * update Q(s, _action)
         */
        best_next_action = argmax(Q_hat[next_state])
        td_target = reward + discount_factor * Q_hat[next_state][best_next_action]
        td_delta = td_target - Q[s][_action]
        Q[s][_action] += alpha * td_delta  /* Gradient Ascent */
        
        every C steps: Q_hat = Q

        if done then
            break
        
        s = next_state /* update s, state rolling */
    }

}
	
```


# Deep Q Learning

