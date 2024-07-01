from deep_q_learning import DQNAgent
from FlappyBird_env import FlappyBird
import pygame 
import time

if __name__ == "__main__":

    agent = DQNAgent(FlappyBird.observation_space(), FlappyBird.action_space())
    agent.load_model('Trained/NN_epoch_9.pth') #copy here the relative path to the saved NN

    # Initialize environment once
    env = FlappyBird()

    # Evaluating
    agent.set_train_off()
    
    state = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
        #time.sleep(0.01)
        action = agent.act(state)
        state, _, done = env.action(action)
    env.close()
