import chainer
import chainerrl
import numpy as np
import pandas as pd

class RandomActor:
  def __init__(self):
    self.random_count = 0
  def random_action_func(self):
    self.random_count += 1
    return np.random.randint(0, 3)

class Market():
  def __init__(self, df, start_id, end_id):
    self.df = df
    self.start_id = start_id
    self.end_id = end_id
  
  def reset(self):
    self.idx = self.start_id
    self.asset = 0.0
    self.hold = 0
    self.buy_price = 0.0
    self.done = False
  
  def current_env(self):
    return np.array([
      self.df.at[self.idx-1, "future_0"],
      self.df.at[self.idx-1, "future_1"],
      self.df.at[self.idx-1, "future_2"],
      self.df.at[self.idx-1, "future_3"],
      self.df.at[self.idx-1, "future_4"],
      self.df.at[self.idx-1, "future_5"],
      self.df.at[self.idx-1, "future_6"],
      self.df.at[self.idx-1, "future_7"],
      self.df.at[self.idx-1, "future_8"],
      self.df.at[self.idx-1, "future_9"],
      self.asset,
      self.buy_price,
      self.hold,
    ], dtype=np.float32)
  
  def move(self, act):
    reward = 0

    if act == 0:
      # hold
      reward = 0
    elif act == 1 and self.hold == 0:
      # buy
      self.buy_price = self.df.at[self.idx, "opening_price"]
      self.asset -= self.buy_price
      self.hold += 1
    elif act == 2 and self.hold > 0:
      # sell
      sell_price = self.df.at[self.idx, "close_price"]
      self.asset += sell_price
      self.hold -= 1
      reward = sell_price - self.buy_price
      self.buy_price = 0.0
    
    self.idx += 1
    if self.idx >= self.end_id:
      self.done = True
    
    return reward
  
  def print(self):
    print(self.df.loc[self.idx])
    print("idx:", self.idx, ", asset:", self.asset, ", hold:", self.hold, ", buy_price:", self.buy_price)

# 環境
df_result = pd.read_csv("result.csv", index_col=0)
env = Market(df_result, 18841, 19089)

# Explorer用のランダム・アクター
ra = RandomActor()

# 環境と行動の次元数
obs_size = 13     # future_0 - 9、asset、buy_price, hold
n_actions = 3     # 0...hold, 1...buy, 2...sell

# Q関数とオプティマイザー
q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
  obs_size,
  n_actions,
  n_hidden_layers=3,
  n_hidden_channels=39)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# 報酬の割引率
gamma = 0.95

# ε-greedy法による探索
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
  start_epsilon=1.0,
  end_epsilon=0.3,
  decay_steps=50000,
  random_action_func=ra.random_action_func)

# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Agentを生成
agent = chainerrl.agents.DoubleDQN(
  q_func,
  optimizer,
  replay_buffer,
  gamma,
  explorer,
  replay_start_size=500,
  update_interval=1,
  target_update_interval=100)

n_episodes = 50000
total_reward = 0

for i in range(1, n_episodes + 1):
  env.reset()
  reward = 0
  
  while not env.done:
    # アクションを取得
    action = agent.act_and_train(env.current_env(), reward)
    reward = env.move(action)
    total_reward += reward

  reward = env.move(2)
  agent.stop_episode_and_train(env.current_env(), reward, True)
  
  # コンソールに進捗を表示
  print("episode:", i, ", total_reward:", total_reward, "asset:", env.asset, ", rnd:", ra.random_count, ", statistics:", agent.get_statistics(), ", epsilon:", agent.explorer.epsilon)
  ra.random_count = 0
  total_reward = 0

  if i % 100 == 0:
    agent.save("result_" + str(i))
    print("agent saved.")

print("Training finished.")

