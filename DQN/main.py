# coding: utf-8
import DeepQLearning
import DeepQLearning_keras
import gym 

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    RL = DeepQLearning_keras.DeepQLearning(action_dim=2, state_dim=4)
    for episode in range(100):
        # initialize observation
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            RL.save_transtion(observation, action, reward, observation_, done)
            RL.train_Q_network()
            observation = observation_

            if done:
                break

