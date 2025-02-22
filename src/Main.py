import gym
import matplotlib.pyplot as plt
import tqdm
import socket
import json


env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
tot_rewards = []
num_episodes = 100

ser = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ser.connect(('localhost', 100))
create_DQN_json = {
    "fun": "new_agent",
    "name": "test",
    "state_dim": state_dim,
    "action_dim": action_dim
}
ser.send(json.dumps(create_DQN_json).encode())

for e in tqdm.tqdm(range(num_episodes)):
    state = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        # send state, receive action
        get_action_json = {
            "fun": "get_action",
            "name": "test",
            "state": state.tolist()
        }
        ser.send(json.dumps(get_action_json).encode())
        action = int(ser.recv(1024).decode())

        # play action
        next_state, reward, done, *_ = env.step(action)

        # update replay
        update_replay_json = {
            "fun": "store_in_replay",
            "name": "test",
            "state": state.tolist(),
            "action": action,
            "reward": reward,
            "next_state": next_state.tolist(),
            "done": done
        }
        ser.send(json.dumps(update_replay_json).encode())
        state = next_state
        total_reward += reward
    tot_rewards.append(total_reward)

plt.plot(tot_rewards)
plt.legend()
plt.show()
