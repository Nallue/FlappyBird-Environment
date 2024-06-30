from deep_q_learning import DQNAgent
from FlappyBird_env import FlappyBird



if __name__ == "__main__":
   

    agent = DQNAgent(4, 2)
    agent.load_model('Trained_1_4in_2out/NN_epoch_34.pth')

    reward_per_epoch = []
    loss = []

    episodes_per_epoch = 1000

    epoch = 0

    # Initialize environment once
    env = FlappyBird()

    # Evaluating
    agent.set_train_off()
    
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, _, _ = env.action(action)
    env.close()
