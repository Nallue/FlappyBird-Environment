import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

import matplotlib
matplotlib.use('Agg')  # Usa el backend "Agg" (no interactivo)
import matplotlib.pyplot as plt

from FlappyBird_env import FlappyBird
import pygame
import time

import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")
        self.model = DQN(input_dim, output_dim).to(self.device)
        self.target_model = DQN(input_dim, output_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.training = True
    
    def act(self, state):
        if np.random.rand() <= self.epsilon and self.training:
            return random.randrange(self.output_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
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

        return float(loss.item())

    def set_train_off(self):
        self.training = False
    
    def set_train_on(self):
        self.training = True

    def print_stats(self):
        print("Epsilon:", self.epsilon)

    def save_model(self, epoch):
        if not os.path.exists('NN'):
            os.makedirs('NN')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, f"NN/NN_epoch_{epoch}.pth")

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

if __name__ == "__main__":
    def plot_draw(reward_per_epoch, name_of_plot):
        plt.figure(figsize=(10, 5), dpi=80)
        plt.plot(reward_per_epoch)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Reward per Epoch')
        # create "plots" directory
        if not os.path.exists('plots'):
            os.makedirs('plots')

        #Save plot in "plots" directory
        plt.savefig(f'plots/{name_of_plot}.png', bbox_inches='tight', pad_inches=0.1)


    agent = DQNAgent(FlappyBird.observation_space(), FlappyBird.action_space())

    reward_per_epoch = []
    loss = []

    episodes_per_epoch = 1000

    epoch = 0

    # Initialize environment once
    env = FlappyBird()

    while True:
        print("Epoch", epoch)
        epoch += 1

        # Training
        agent.set_train_on()
        state = env.reset()
        total_reward = 0
        total_loss = 0
        for l in range(episodes_per_epoch):
            #pygame event check
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                
            action = agent.act(state)
            new_state, reward, done = env.action(action)
            agent.store_transition(state, action, reward, new_state, done)
            total_loss += agent.learn()
            total_reward += reward
            if done:
                new_state = env.reset()
            state = new_state
        
        print(f"Total reward after epoch {epoch}: {total_reward}")
        reward_per_epoch.append(total_reward)
        loss.append(total_loss)
        agent.print_stats()

        # Save model periodically
        if True: # You can use something like epoch % 100 == 0
            env.close()
            agent.save_model(epoch)
            plot_draw(reward_per_epoch, f'reward_per_epoch{epoch}')
            #plot_draw(loss, f'loss_per_epoch{epoch}') #If you want to save the loss plot
            env = FlappyBird()

        # Evaluating
        agent.set_train_off()
        
        state = env.reset()
        done = False
        while not done:
            #pygame event check
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                
            if (env.get_score() > 20):
                state = env.reset()
            action = agent.act(state)
            new_state, reward, done = env.action(action)
            agent.store_transition(state, action, reward, new_state, done)
            state = new_state
            time.sleep(0.01)
        print("Evaluation completed, environment reset for next epoch.")
