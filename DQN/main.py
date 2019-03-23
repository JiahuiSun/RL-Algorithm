# coding: utf-8
import DeepQLearning
import DeepQLearning_keras
import gym 
import time


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    RL = DeepQLearning.DeepQLearning(action_dim=2, state_dim=4, epsilon=0.2)
    for episode in range(1000):
        # initialize observation
        observation = env.reset()
        while True:
            # env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            RL.save_transtion(observation, action, reward, observation_, done)
            RL.train_Q_network()
            observation = observation_

            if done:
                break
    # eval
    RL.saver.save(RL.sess, RL.save_path)
    obs = env.reset()
    st = time.time()
    while True:
        env.render()
        action = RL.choose_action(obs, eval=True)
        obs_, rew, done, _ = env.step(action)

        if done:
            print("keep on time:{}".format(time.time()-st))
            break
        obs = obs_
