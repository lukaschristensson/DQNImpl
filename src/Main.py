import gym
import matplotlib.pyplot as plt
import tqdm
from DQN import *

time_before = time.time()
env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
dqn = DQN(state_dim, action_dim, epsilon_0=.9, opt='Adam')
tot_rewards = []
num_episodes = 100

for e in tqdm.tqdm(range(num_episodes)):
    state = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        action = dqn.action(state)
        next_state, reward, done, *_ = env.step(action)
        dqn.store_in_replay(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        dqn.replay()
    tot_rewards.append(total_reward)
time_taken = time.time() - time_before

print('idx time:\t', 100*IDX_TIME/time_taken, '%')
print('train time:\t', 100*TRAINING_TIME/time_taken,'%')
print('copy time:\t', 100*COPY_TIME/time_taken, '%')
figs, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(losses)
ax2.plot(tot_rewards)
plt.show()
