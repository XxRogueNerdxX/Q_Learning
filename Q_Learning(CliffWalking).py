import gym 
import numpy as np 

env = gym.make('CliffWalking-v0')

q_table = np.zeros((env.observation_space.n, env.action_space.n))

def take_action(state):
    e = np.random.uniform(0,1)
    if e < epsilon:
        action = env.action_space.sample()
    
    else: 
        action = np.argmax(q_table[state, :])
    return action 

def update(state,action,reward,next_state):
    Target = np.max(q_table[next_state , : ])
    q_table[state,action] += reward + gamma * Target - q_table[state, action]

gamma = 0.9
epsilon = 0.9 
eps_min = 0.01
eps_dec = 0.01
total_episodes = 100

score_list  = []
for episode in range(total_episodes):
    score = 0 
    state = env.reset()
    done = False
    while not done:
        action = take_action(state)
        next_state, reward, done, info = env.step(action)
        update(state, action, reward, next_state)
        score+= reward
        state = next_state

    score_list.append(score)
    epsilon = epsilon - eps_dec if epsilon > eps_min else eps_min

    if episode % 10 == 0:
        print('episode', episode, 'epsilon %.2f' %  (epsilon), 'previous_score', score)
