import torch
from mario_ppo import PolicyNetwork
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from env import CustumEnv,CustumSingleEnv,SkipEnv

def play(max_action, global_model, num_actions, device):
    local_model = PolicyNetwork(num_actions).to(device)
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = CustumSingleEnv(env)
    env = SkipEnv(env)
    state = env.reset()
    done = True
    curr_step = 0
    while True:
        if done:
            local_model.load_state_dict(global_model.state_dict())

        env.render()
        state = torch.FloatTensor(state).to(device)
        actions,_ = local_model(state)
        action = actions.argmax().item()
        state, reward, done, _ = env.step(action)

        curr_step += 1
        if curr_step == max_action:
            done = True
        if done:
            curr_step = 0
            state = env.reset()
        # print(f'Return: {reward}')

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = CustumSingleEnv(env)
    env = SkipEnv(env)
    state = env.reset()
    save_ppo_path = "actor.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ppo = torch.load(save_ppo_path, map_location=device, weights_only=False)
    iter = 0
    done = False
    while not done:
        env.render()
        state = torch.FloatTensor(state).to(device)
        # state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        actions, _ = ppo(state)
        action = actions.argmax().item()
        # print(actions)
        # if iter % 2 == 0:
        #     action = 2
        # else:
        #     action = 4
        # action = 9
        next_state, reward, done, _ = env.step(action)
        state = next_state
        iter += 1
        print(f'Return: {reward}')

if __name__ == '__main__':
    main()