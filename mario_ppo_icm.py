import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as _mp
from torch.distributions import Categorical
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from env import MultipleEnvironments,obsShape
import ppo_play
import numpy as np
from mario_ppo import PolicyNetwork,ppo_update, RolloutStorage

class Encoder(nn.Module):
    def __init__(self,state_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, latent_dim)


    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(-1,1600)
        x = torch.relu(self.fc1(x))
        ac = self.fc2(x)
        return torch.softmax(ac,dim=-1)

class Decoder(nn.Module):
    def __init__(self,state_dim, latent_dim):
        super().__init__()

class Step(nn.Module):
    def __init__(self,action_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x


class ICMNetWork(nn.Module):
    def __init__(self,state_dim, action_dim, latent_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(state_dim,latent_dim)
        self.decoder = Decoder(state_dim,latent_dim)
        self.step = Step(action_dim, latent_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        return x

def main():

    if torch.cuda.is_available():
        torch.cuda.manual_seed(996)
    else:
        torch.manual_seed(996)
    replay_buffer_size = 256
    use_save = False
    save_actor_path = "actor-icm.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    world = 'SuperMarioBros-1-1-v0'

    # 创建马里奥环境
    action_dim = len(COMPLEX_MOVEMENT)
    nums_processes = 8
    envs = MultipleEnvironments(nums_processes,world)
    max_actions = 500

    if use_save:
        policy_net = torch.load(save_actor_path, map_location = device, weights_only=False)
    else:
        policy_net = PolicyNetwork(action_dim).to(device)


    policy_net.share_memory()

    mp = _mp.get_context("spawn")
    process = mp.Process(target=ppo_play.play, args=(max_actions, policy_net, action_dim, world, device))
    process.start()

    if use_save:
        policy_net.load_state_dict(torch.load(save_actor_path, map_location = device, weights_only=False).state_dict())
        policy_net.to(device)
    else:
        policy_net._initialize_weights()

    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True,weight_decay=0.001)


    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    state = [agent_conn.recv() for agent_conn in envs.agent_conns]

    max_episodes = 1000000
    gamma = 0.90 #0.99
    gae_lambda = 0.90

    rollouts = RolloutStorage(
        replay_buffer_size,
        nums_processes,
        obsShape,
        action_dim,
        0,
    )

    state_tensor = torch.FloatTensor(state).to(device)
    rollouts.observations[0].copy_(state_tensor)
    rollouts.to(device)

    for episode in range(max_episodes):

        for step in range(replay_buffer_size):
            with torch.no_grad():
                action_probs, value = policy_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            state, reward, done, _ = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            masks = 1 - torch.FloatTensor(done).to(device)
            reward = torch.FloatTensor(reward).to(device)
            # for states in state:
            #     print(states.shape)
            # print(step)
            state = torch.from_numpy(np.concatenate(state, 0))
            state_tensor = torch.FloatTensor(state).to(device)
            rollouts.insert(
                state_tensor, action.unsqueeze(1), log_prob.unsqueeze(1), value, reward.unsqueeze(1), masks.unsqueeze(1)
            )


        # print(episode)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            next_value = policy_net.get_value(state_tensor)
        rollouts.compute_returns(next_value, True, gamma, gae_lambda)

        ppo_update(policy_net, optimizer, rollouts)
        rollouts.after_update()

        if episode % 10 == 0:
            torch.save(policy_net, save_actor_path)
    envs.close()

if __name__ == '__main__':
    main()