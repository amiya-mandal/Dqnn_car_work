#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import os


# In[2]:


env = gym.make("CartPole-v0")


# In[5]:


state_size = env.observation_space.shape[0]


# In[6]:


state_size


# In[8]:


action_size = env.action_space.n


# In[9]:


action_size


# In[10]:


batch_size = 32


# In[11]:


n_episodes = 1001


# In[12]:


output_dir = 'model/cartpole'


# In[13]:


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[15]:


from model import DQNAgent


# In[16]:


agent = DQNAgent(state_size, action_size)


# In[17]:


agent.model.summary()


# In[20]:


done = False

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(5000):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if not done:
            reward = reward
        else:
            reward = -10
        
        next_step = np.reshape(next_state, (1, state_size))
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            print("episode {}/{}, score :{}, e:{}".format(e, n_episodes, time, agent.epsilon))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
        
    if e % 50 == 0:
        agent.save(output_dir+"weights_"+"{:04d}".format(e)+".hdf5")


# In[ ]:




