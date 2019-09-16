
# Dynamic Programmings






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

