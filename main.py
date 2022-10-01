# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import ware_house.envs


def test_gym(env):
    episodes = 5
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        done = False
        score = 0
        print(info)
        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
            #print(f"{info}")
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()
def train(env):
    log_path = os.path.join('Training', 'Logs')
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=400000)
    model.save('PPO')
    evaluate_policy(model, env, n_eval_episodes=10, render=True)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make("rware-v1")
    #test_gym(env)
    train(env)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
