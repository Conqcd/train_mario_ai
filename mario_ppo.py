import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# 创建马里奥环境
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# 创建PPO模型
model = PPO('CnnPolicy', env, verbose=1, device='cuda')

# 训练模型
model.learn(total_timesteps=100000)

# 保存模型
model.save("ppo_mario")

# 测试模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()