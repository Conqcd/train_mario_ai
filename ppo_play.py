import torch
from mario_ppo import CustumEnv, PolicyNetwork
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = CustumEnv(env)
    state = env.reset()
    save_ppo_path = "actor.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ppo = torch.load(save_ppo_path, map_location=device, weights_only=False)
    iter = 0
    done = False
    while not done:
        env.render()
        state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        actions = ppo(state)
        action = actions.argmax().item()
        if iter % 2 == 0:
            action = 5
        else:
            action = 4
        next_state, reward, done, _ = env.step(action)
        state = next_state
        iter += 1
        print(f'Return: {reward}')

if __name__ == '__main__':
    main()