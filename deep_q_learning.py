import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from FlappyBird_env import FlappyBird

import time

def rand_action():
    x = random.randint(0, 1)
    print(x)
    return x

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=64, target_update=10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0

        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.training = True
    
    def act(self, state):
        if np.random.rand() <= self.epsilon and self.training:
            return random.randrange(self.output_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q = self.model(states).gather(1, actions.view(-1, 1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min and self.training:
            self.epsilon *= self.epsilon_decay

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def set_train_off(self):
        self.training = False
    
    def set_train_on(self):
        self.training = True

    def print_stats(self):
        print("Epsilon:", self.epsilon)

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, file_path)
        #print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        #print(f"Model loaded from {file_path}")

if __name__ == "__main__":
    agent = DQNAgent(7, 2, batch_size = 500)

    #agent.load_model("NN")

    episodes_per_epoch = 2500

    epoch = 0
    
    while(True):
        print("Epoch", epoch)
        epoch += 1
        #training
        agent.set_train_on()
        env = FlappyBird()
        state = env.reset()
        for l in range(episodes_per_epoch):
            #time.sleep(0.01)
            #print(l)
            action = agent.act(state)
            new_state, reward, done = env.action(action)
            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()
            if done:
                new_state = env.reset()
            state = new_state
        
        #saving
        agent.save_model("NN")

        #evaluating
        agent.set_train_off()
        env.close()
        env = FlappyBird()
        state = env.reset()
        done = False
        while(done == False):
            state, _, done = env.action(agent.act(state))
            time.sleep(0.01)
        env.close()
