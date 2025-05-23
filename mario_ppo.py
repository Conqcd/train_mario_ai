import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as _mp
from torch.distributions import Categorical
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from env import CustumEnv,CustumSingleEnv,SkipEnv,MultipleEnvironments,obsShape
import ppo_play
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(-1,1600)
        x = torch.relu(self.fc1(x))
        ac = self.fc2(x)
        val = self.critic(x)
        return torch.softmax(ac,dim=-1),val

    def get_value(self,x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(-1,1600)
        x = torch.relu(self.fc1(x))
        val = self.critic(x)
        return val

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_dim, state_size):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.observations = self.observations.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
            self, current_obs, action, action_log_prob, value_pred, reward, mask
    ):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps
    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self,next_value,use_gae,gamma,gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.num_steps)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                # gae = gae * self.bad_masks[step + 1]
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.num_steps)):
                self.returns[step] = (
                (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )
                # * self.bad_masks[step + 1]
                    +
                # + (1 - self.bad_masks[step + 1]) *
                    self.value_preds[step]
                )

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False
        )
        for indices in sampler:
            # indices.sort()
            observations_batch = self.observations[:-1].view(
                -1, *self.observations.size()[2:]
            )[indices]
            # observations_batch = self.observations[:-1][indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            values_batch = self.value_preds[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield observations_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ #, values_batch, indices

def ppo_update(policy_net, optimizer, rollouts, clip_epsilon=0.2,max_grad_norm=1.0):
    wa = 1
    wv = 1
    we = 0.01 #0.0001
    num_mini_batch = 16

    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(10):  # Update for 10 epochs

        data_generator = rollouts.feed_forward_generator(
            advantages, num_mini_batch
        )

        # action_probs, values = policy_net(rollouts.observations[:-1].view(
        #         -1, *rollouts.observations.size()[2:]
        #     ))
        print("-----------------")
        for sample in data_generator:
            (
                states,
                actions,
                returns,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                # values_batch,
                # indices,
            ) = sample

            action_probs,values = policy_net(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions.squeeze(1)).unsqueeze(1)
            # for i,id in enumerate(indices):
            #     print(torch.sum(states[i] - rollouts.observations[:-1].view(-1, *rollouts.observations.size()[2:])[id]))
            #     print(torch.sum(states[i] - rollouts.observations[:-1][id]))
            #     print(torch.sum(actions[i] - rollouts.actions[id]))
            #     print(torch.sum(returns[i] - rollouts.returns[id]))
            #     print(torch.sum(masks_batch[i] - rollouts.masks[id]))
            #     print(torch.sum(old_action_log_probs_batch[i] - rollouts.action_log_probs[id]))
            #     print(torch.sum(new_log_probs[i] - rollouts.action_log_probs[id]))
            #     print(torch.sum(values_batch[i] - rollouts.value_preds[id]))
            #     print(torch.sum(values[i] - rollouts.value_preds[id]))

            entropy = dist.entropy().unsqueeze(1).sum(-1).mean()
            # dist = [Categorical(a) for a in action_probs]
            # new_log_probs = torch.stack([dist[i].log_prob(actions[i]) for i in range(len(dist))])
            # entropy = torch.stack([d.entropy() for d in dist]).mean()

            ratio = torch.exp(new_log_probs - old_action_log_probs_batch)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_targ
            policy_loss = -torch.min(surr1, surr2).mean()

            # values = policy_net.get_value(states)
            value_loss = (returns - values).pow(2).mean()

            optimizer.zero_grad()
            loss = value_loss * wv - entropy * we + policy_loss * wa
            loss.backward()
            nn.utils.clip_grad_norm_(
                policy_net.parameters(), max_grad_norm
            )
            optimizer.step()
            print(policy_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy(), entropy.detach().cpu().numpy(),loss.detach().cpu().numpy())

def main():

    if torch.cuda.is_available():
        torch.cuda.manual_seed(996)
    else:
        torch.manual_seed(996)
    replay_buffer_size = 256
    use_save = False
    save_actor_path = "actor.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    world = 'SuperMarioBros-2-2-v0'

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