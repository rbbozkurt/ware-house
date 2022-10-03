# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import os
from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import ware_house.envs


def test_gym(environment):
    episodes = 5
    for episode in range(1, episodes + 1):
        state, info = environment.reset()
        done = False
        score = 0
       # print(info)
        while not done:
            action = environment.action_space.sample()
            n_state, reward, done, info = environment.step(action)
            score += reward
            # print(f"{info}")
        print('Episode:{} Score:{}'.format(episode, score))
    environment.close()


def train(environment):
    log_path = os.path.join('Training', 'Logs')
    model = A2C("MultiInputPolicy", environment, verbose=1)
    model.learn(total_timesteps=400000)
    model.save('a2c-rware')
    evaluate_policy(model, environment, n_eval_episodes=10, render=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make("rware-v3")
    #test_gym(env)
    train(env)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
