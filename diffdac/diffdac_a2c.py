# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
""" Diff-A2C agent """

import torch.nn as nn
import torch.optim as optim
import torch


class DiffDAC_A2C():
    def __init__(self, actor_critic, value_loss_coef, entropy_coef, lr=None,
                 eps=None, alpha=None, max_grad_norm=None,
                 rank=0, gossip_buffer=None, link_drop_prob=0.0):
        """ DiffDAC_A2C """

        self.rank = rank
        self.gossip_buffer = gossip_buffer
        self.actor_critic = actor_critic

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.link_drop_prob = link_drop_prob
        self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        self.update_iteration = 0

    def update(self, rollouts):
        """
        Updates both value and policy function parameters using gradient-based optimisation and
        diffusion updates.
        """
        # Attain required dimension sizes
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # Calculate values using the value function for bootstrapping purposes where necessary.
        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        # Build loss functions.
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        # Perform gradient update
        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Local diffusion ('combination') update step (including parameter writing).
        if self.gossip_buffer is not None:
            # Account for simulating of broken links.
            if torch.rand(()) >= self.link_drop_prob:
                self.gossip_buffer.write_message(self.rank, self.actor_critic)
            self.gossip_buffer.aggregate_message(self.rank, self.actor_critic)

        self.update_iteration += 1
        return value_loss.item(), action_loss.item(), dist_entropy.item()
