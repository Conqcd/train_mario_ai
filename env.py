import gym
from gym import spaces
import cv2
import numpy as np
import time
import torch.multiprocessing as mp
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

obsShape = (4, 84, 84)

class CustumEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustumEnv, self).__init__(env)
        self.env = env
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        self.s_t = None
        self.state = None
        self.last_render_time = time.perf_counter()
        self.fps = 240  # 设定目标帧率
        self.last_score = 0
        self.hold_jump = 0

    def reset(self):
        obs = self.env.reset()
        self.state = obs
        x_t = cv2.cvtColor(cv2.resize(obs, (84, 84)), cv2.COLOR_BGR2GRAY)
        x_t = self.image_to_frames(x_t)
        self.last_score = 0
        return x_t

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.state = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        x_t = cv2.cvtColor(cv2.resize(obs, (84, 84)), cv2.COLOR_BGR2GRAY)
        x_t = self.image_to_frames(x_t)

        if self.hold_jump > 25:
            self.hold_jump = 0

        if action == 5 or action == 2 or action == 4:
            self.hold_jump += 1
            reward += (self.hold_jump - 1) * 0.01
        else:
            self.hold_jump = 0


        if info['x_pos_screen'] < 5:
            reward -= 1

        reward += info['score'] - self.last_score
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        # if info['status'] == 'tall':
        #     reward += 1

        self.last_score = info['score']

        return x_t, reward, done, info

    def image_to_frames(self, x_t):
        if self.s_t is None:
            self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        else:
            x_t = np.reshape(x_t, (84, 84, 1))
            self.s_t = np.append(x_t, self.s_t[:, :, :3], axis=2)
        return self.s_t

    def render(self, mode='human'):
        current_time = time.perf_counter()
        elapsed = current_time - self.last_render_time
        wait_time = max(0, 1 / self.fps - elapsed)
        time.sleep(wait_time)
        self.last_render_time = time.perf_counter()

        if mode == 'human':
            # 如果是人类模式，则显示图片（例如使用 OpenCV 显示）
            cv2.imshow("Super Mario", self.state)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            # 返回当前帧的图像数据
            return self.state


class CustumSingleEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustumSingleEnv, self).__init__(env)
        self.env = env
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        self.state = None
        self.last_render_time = time.perf_counter()
        self.fps = 240  # 设定目标帧率
        self.last_score = 0
        self.hold_jump = 0

    def reset(self):
        obs = self.env.reset()
        self.state = obs
        x_t = cv2.cvtColor(cv2.resize(obs, (84, 84)), cv2.COLOR_BGR2GRAY)[None, :, :]
        self.last_score = 0
        return x_t

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.state = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        x_t = cv2.cvtColor(cv2.resize(obs, (84, 84)), cv2.COLOR_BGR2GRAY)[None, :, :]

        # if self.hold_jump > 25:
        #     self.hold_jump = 0
        #
        # if action == 5 or action == 2 or action == 4:
        #     self.hold_jump += 1
        #     reward += (self.hold_jump - 1) * 0.01
        # else:
        #     self.hold_jump = 0


        # if info['x_pos_screen'] < 5:
        #     reward -= 1

        reward += (info['score'] - self.last_score) / 40
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        # if info['status'] == 'tall':
        #     reward += 1

        self.last_score = info['score']

        return x_t, reward / 10, done, info


    def render(self, mode='human'):
        current_time = time.perf_counter()
        elapsed = current_time - self.last_render_time
        wait_time = max(0, 1 / self.fps - elapsed)
        time.sleep(wait_time)
        self.last_render_time = time.perf_counter()

        if mode == 'human':
            # 如果是人类模式，则显示图片（例如使用 OpenCV 显示）
            cv2.imshow("Super Mario", self.state)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            # 返回当前帧的图像数据
            return self.state

class SkipEnv(gym.Wrapper):
    def __init__(self, env):
        super(SkipEnv, self).__init__(env)
        self.env = env
        self.skip = 4
        self.states = np.zeros((self.skip, 84, 84), dtype=np.float32)
        self.last_shape = (self.skip, 84, 84)

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states.astype(np.float32)

    def step(self, action):

        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state

        if self.states.shape[0] == 1 :
            print(action)

        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

class MultipleEnvironments:
    def __init__(self, num_envs, num_states = 4):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.num_states = num_states
        self.num_actions = len(COMPLEX_MOVEMENT)
        self.processes = []
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index, 1))
            # self.processes.append(process)
            process.start()

    def run(self, index, a):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = CustumSingleEnv(env)
        env = SkipEnv(env)
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(env.step(action.item()))
            elif request == "reset":
                self.env_conns[index].send(env.reset())
            else:
                raise NotImplementedError

    def close(self):
        for process in self.processes:
            process.close()
