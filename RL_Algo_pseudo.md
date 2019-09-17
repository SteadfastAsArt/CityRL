# Dynamic Programming

Fill up and update the table of (action, state) pair using greedy policy, which is proven to converge to a certain point.

## Policy Evaluation
Evaluate a given policy π

```pseudocode
function policy_eval(policy, env)
{
	/* policy is a function of state [any policy as agent specified], returns action's prob distribution */
	
	Val = dict()  /* value function, of state, returns "value" */
	
	while true  /* set a terminal state when update diff undistinguished */
	{
		for s in env.state  /* to update the whole state set */
		{
			v = 0
			for action, action_prob in policy(s) /* summon up all techniques */
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
Goal is to improve a policy

`Evaluation` part has done most of the improving jobs

```pseudocode

while true
{
	/* 1: Evaluate current policy, returns a value function */
	Val = policy_eval(policy, env)
	
	for s in env.state
	{
		/* one step lookahead start */
		for action, action_prob in policy(s)
		{
			for tran_prob, next_state, reward, done in env.P[s][action]
			{
				A[action] += discount_factor * tran_prob * Val[next_state]
			}
			A[action] += rewards
		}
		/* one step lookahead end */
		
		/* 2: Greedy update each state of the certain policy */
		policy[s] = find_best_action(Q_tmp)
		
	}
	
}


```

## Value Iteration

Goal is to find an optimal policy


```pseudocode

while true:
{
	init Val
	for s in env.state
	{
		/* one step lookahead start */
		for action, action_prob in policy(s)
		{
			for tran_prob, next_state, reward, done in env.P[s][action]
			{
				A[action] += discount_factor * tran_prob * Val[next_state]
			}
			A[action] += rewards
		}
		/* one step lookahead end */
        
        best_action_val = max(A)
        Val[s] = best_action_val
	}
}

for s in env.state
{
	one step lookahead to find best action in s
	update policy
}


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

