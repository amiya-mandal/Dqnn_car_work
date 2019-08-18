import random
import torch
import torch.nn as nn
import gym
import numpy as np
import os
from collections import deque
import torch.optim as optim

torch.cuda.set_device(0)
env = gym.make('CartPole-v0') 

state_size = env.observation_space.shape[0]
state_size

action_size = env.action_space.n
action_size

batch_size = 32

n_episodes = 1001 

output_dir = 'model_output_/MountainCar-v0/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNModel(nn.Module):

    def __init__(self,state_size, action_size):
        super(DQNModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_size, 120),
            nn.ReLU(),
            nn.Linear(120, 24),
            nn.ReLU(),
            nn.Linear(24, action_size),
        )

    def forward(self, x):
        return self.main(x)



class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.learning_rate = 0.001
        self.model = self.__build_model()

        self.model.cuda()
        self.criterion.cuda()
    
    def __build_model(self):
        model = DQNModel(self.state_size, self.action_size)
        print(model)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model

    def __train(self):
        self.model.train()

    def fit(self, state, target_f, epochs=1):
        self.__train()
        for i in range(epochs):
            state_t = torch.from_numpy(state).float()
            state_t = state_t.cuda()
            output = self.model(state_t)
            loss = self.criterion(output, torch.from_numpy(target_f).float().cuda())
            loss.backward()
            self.optimizer.step()
        self.__eval()

    def __eval(self):
        self.model.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        self.__eval()
        state_t = torch.from_numpy(state).float()
        state_t = state_t.cuda()
        act_val = self.model(state_t)
        act_val = act_val.cpu().detach().numpy()
        return np.argmax(act_val[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                self.__eval()
                next_state_t = torch.from_numpy(next_state).float()
                next_state_t = next_state_t.cuda()
                pred = self.model(next_state_t)
                pred = pred.cpu().detach().numpy()
                target = (reward + self.gamma * np.amax(pred[0]))
            state_t = torch.from_numpy(state).float()
            state_t = state_t.cuda()
            target_f = self.model(state_t)
            target_f = target_f.cpu().detach().numpy()
            target_f[0][action] = target
            self.fit(state, target_f, epochs=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, name):
        torch.save(self.model.state_dict(), name+'.pt')
    
    def load(self, name):
        self.model = torch.load(name)


agent = DQNAgent(state_size, action_size)

done = False

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
    for time in range(5000):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if not done:
            reward = reward
        else:
            reward = -10
        
        next_state = np.reshape(next_state, [1, state_size])
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            print("episode {}/{}, score :{}, e:{}".format(e, n_episodes, time, agent.epsilon))
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
        
    if e % 50 == 0:
        agent.save(output_dir+"weights_"+"{:04d}".format(e))

env.close()
