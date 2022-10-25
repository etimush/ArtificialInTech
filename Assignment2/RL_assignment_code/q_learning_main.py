import matplotlib.pyplot as plt
import numpy as np

import simple_grid
from q_learning_skeleton import *
import gym

def act_loop(env, agent, num_episodes,exploration_reward):
    for episode in range(num_episodes):
        state = env.reset()
        tot_t = 0
        exploration_reward = exploration_reward.copy()

        print('---episode %d---' % episode)
        renderit = False
        if episode % 10 == 0:
            renderit = True

        for t in range(MAX_EPISODE_LENGTH):
            if renderit:
                env.render()
            printing=False
            if t % 500 == 499:
                printing = True

            if printing:
                print('---stage %d---' % t)
                agent.report(False)
                print("state:", state)

            action = agent.select_action(state)
            new_state, reward, done, info = env.step(action)
            reward += exploration_reward[new_state,state]
            exploration_reward[new_state,state] = 0


            if printing:
                print("act:", action)
                print("reward=%s" % reward)

            agent.process_experience(state, action, new_state, reward, done)
            state = new_state

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                env.render()
                tot_t = t + 1
                agent.reset_episode(tot_t)
                agent.report(False)
                break


    episode_durations = agent.report(True)
    rolling_average_size = 8
    rolling_average = []
    plt.plot(range(len(episode_durations)), episode_durations)

    for i in range(len(episode_durations)-rolling_average_size):
        rolling_average.append(sum(episode_durations[i:i+rolling_average_size])/rolling_average_size)

    plt.plot(range(len(episode_durations)-rolling_average_size), rolling_average)
    plt.title("E-greedy exploration strategy")
    plt.xlabel("Episode")
    plt.ylabel("Episode duration")
    plt.show()
    env.close()


if __name__ == "__main__":
    env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
    # env = simple_grid.DrunkenWalkEnv(map_name="theAlley")
    num_a = env.action_space.n

    if (type(env.observation_space)  == gym.spaces.discrete.Discrete):
        num_o = env.observation_space.n
    else:
        raise("Qtable only works for discrete observations")


    discount = DEFAULT_DISCOUNT
    ql = QLearner(num_o, num_a, discount) #<- QTable
    exploration_rewards = np.ones((num_o,num_o))*0
    act_loop(env, ql, NUM_EPISODES,exploration_rewards)