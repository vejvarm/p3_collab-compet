# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(24, 2, (8, 16, 32), (8, 4, 2), (2, 1, 1), (32, 16, 8),  # actor settings
                                       26, (8, 16, 32), (8, 4, 2), (2, 1, 1), (32, 16, 8)),    # critic settings
                             DDPGAgent(24, 2, (8, 16, 32), (8, 4, 2), (2, 1, 1), (32, 16, 8),  # actor settings
                                       26, (8, 16, 32), (8, 4, 2), (2, 1, 1), (32, 16, 8))]    # critic settings
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return torch.stack(actions)

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def td_error(self, obs, actns, next_obs, reward, done, num_agents):

        obs_full = torch.tensor(obs).unsqueeze(1).to(device)  # 2x1x24
        actions = torch.tensor(actns).unsqueeze(1).to(device)  # 2x1x2
        next_obs_full = torch.tensor(next_obs).unsqueeze(1).to(device)  # 2x1x24
        rewards = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(device)  # 2x1
        dones = torch.tensor(done, dtype=torch.int8).unsqueeze(1).to(device)  # 2x1

        target_actions = torch.stack(self.target_act(next_obs_full))  # out: 2x1x2
        target_critic_input = torch.cat((next_obs_full, target_actions), dim=2).to(device)  # 2x64x26

        psum = 0.

        for agent_number in range(num_agents):
            agent = self.maddpg_agent[agent_number]
            with torch.no_grad():
                critic_input = torch.cat((obs_full[agent_number], actions[agent_number]), dim=1).to(device)  # 1x26
                q = agent.critic(critic_input)  # 64x1
                q_next = agent.target_critic(target_critic_input[agent_number])  # 64x1
                y = rewards[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - dones[agent_number].view(-1, 1))  # 1x1

            psum += (torch.square(y - q)).cpu().detach().numpy()

        return psum/num_agents

    def update(self, samples, num_agents, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        # we don't need [parralel_agent] info
        obs, actns, reward, next_obs, done = samples  # 5x64x(var)

        obs_full = torch.tensor(obs).permute(1, 0, 2).to(device)  # 2x64x24
        actions = torch.tensor(actns).permute(1, 0, 2).to(device)  # 2x64x2
        next_obs_full = torch.tensor(next_obs).permute(1, 0, 2).to(device)  # 2x64x24
        rewards = torch.tensor(reward, dtype=torch.float).permute(1, 0).to(device)  # 2x64
        dones = torch.tensor(done, dtype=torch.int8).permute(1, 0).to(device)  # 2x64

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = torch.stack(self.target_act(next_obs_full))  # out: 2x64x2
        target_critic_input = torch.cat((next_obs_full, target_actions), dim=2).to(device)  # 2x64x26

        huber_loss = torch.nn.SmoothL1Loss()

        for agent_number in range(num_agents):
            agent = self.maddpg_agent[agent_number]
            agent.critic_optimizer.zero_grad()

            with torch.no_grad():
                q_next = agent.target_critic(target_critic_input[agent_number])  # 64x1

            y = rewards[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - dones[agent_number].view(-1, 1))  # 64x1
            action = actions[agent_number]  # 64x2
            critic_input = torch.cat((obs_full[agent_number], action), dim=1).to(device)  # 64x26
            q = agent.critic(critic_input)  # 64x1

            critic_loss = huber_loss(q, y.detach())
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            #update actor network using policy gradient
            agent.actor_optimizer.zero_grad()
            # make input to agent
            # detach the other agents to save computation
            # saves some time for computing derivative
            q_input = [self.maddpg_agent[i].actor(ob) if i == agent_number
                       else self.maddpg_agent[i].actor(ob).detach()
                       for i, ob in enumerate(obs_full)]  # list(64x2, 64x2)

            q_input = torch.cat(q_input, dim=1)  # 64x4
            # combine all the actions and observations for input to critic
            # many of the obs are redundant, and obs[1] contains all useful information already
            q_input2 = torch.cat((obs_full[agent_number, :, 0:-2], q_input), dim=1)  # 64x26

            # get the policy gradient
            actor_loss = -agent.critic(q_input2).mean()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor_optimizer.step()

            al = actor_loss.cpu().detach().item()
            cl = critic_loss.cpu().detach().item()
            logger.add_scalars('agent%i/losses' % agent_number,
                               {'critic loss': cl,
                                'actor_loss': al},
                               self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




