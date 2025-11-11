# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable

# rum hack
from rsl_rl.modules.info_reward import InformationReward
from rsl_rl.modules.goal_reward import GoalReward
import rum

# The most horrible hack of all.
def get_rnd_state_hack(obs, obs_groups):
    obs_list = []
    for obs_group in obs_groups["rnd_state"]:
        obs_list.append(obs[obs_group])
    return torch.cat(obs_list, dim=-1)


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticRecurrent
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        rewarder_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.rnd_optimizer = None # Warning: This variable is used to check whether RND rewards are used.

        # rum hack create schedules
        alpha_schedule = None
        beta_schedule = None
        if rewarder_cfg is not None and 'alpha_schedule_cfg' in rewarder_cfg and rewarder_cfg['alpha_schedule_cfg']:
            alpha_schedule_cfg = rewarder_cfg.pop('alpha_schedule_cfg')
        else:
            alpha_schedule_cfg = None
        if rewarder_cfg is not None and 'beta_schedule_cfg' in rewarder_cfg and rewarder_cfg['beta_schedule_cfg']:
            beta_schedule_cfg = rewarder_cfg.pop('beta_schedule_cfg')
        elif rnd_cfg is not None and 'beta_schedule_cfg' in rnd_cfg and rnd_cfg['beta_schedule_cfg']:
            beta_schedule_cfg = rnd_cfg.pop('beta_schedule_cfg')
        else:
            beta_schedule_cfg = None
        if alpha_schedule_cfg:
            schedule_cfg = alpha_schedule_cfg 
            schedule_class_name = schedule_cfg.pop('name')
            schedule_class = getattr(
                rum.schedule,
                schedule_class_name
            )
            alpha_schedule = schedule_class(**schedule_cfg)
        if beta_schedule_cfg:
            schedule_cfg = beta_schedule_cfg 
            schedule_class_name = schedule_cfg.pop('name')
            schedule_class = getattr(
                rum.schedule,
                schedule_class_name
            )
            beta_schedule = schedule_class(**schedule_cfg)

        # RND components
        self.rnd = None
        self.rnd_optimizer = None # Warning: This variable is used to check whether RND rewards are used.
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg, beta_schedule=beta_schedule)
            # Create RND optimizer
            if rewarder_cfg is None: # rum hack
                # Only need optimizer if not using info or goal rewards.
                params = self.rnd.predictor.parameters()
                self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)

        # rum hack
        self.info_reward = None
        self.goal_reward = None
        self.geom = None
        self.density = None
        if rewarder_cfg is not None:
            # Below is pretty horrible but necessary to have rum vs rsl_rl compability without refactoring.
            # The problem is that 
            # 1) rum creates objects separately based on config dicts and passes them into each other, while in rsl_rl, objects are created within other objects based on config dicts passed into them.
            # 2) The rum rewarders need custom wrapping to correctly and efficiently compute intrinsic rewards and updates within rsl_rl.
            # Part of most horrible hack.
            self.obs_groups = rewarder_cfg.pop("obs_groups")

            # Create geometry model.
            density_num_states = rewarder_cfg['num_states']
            if self.rnd: # Hack, use rnd random network as representation.
                density_num_states = self.rnd.target[-1].out_features 
            elif 'geometry_cfg' in rewarder_cfg:
                geom_cfg = rewarder_cfg.pop('geometry_cfg')
                geom_cls_name = geom_cfg.pop('name')
                if geom_cls_name == 'EmbeddingGeometry':
                  geom_cfg['device'] = self.device
                  # Create graph for embedding geometry.
                  assert 'graph_cfg' in rewarder_cfg
                  graph_cfg = rewarder_cfg.pop('graph_cfg')
                  graph_cls_name = graph_cfg.pop('name')
                  graph_cls = getattr(rum.graph, graph_cls_name)
                  graph_geometry = rum.geometry.EuclideanGeometry(dim=rewarder_cfg['num_states']) 
                  graph = graph_cls(**graph_cfg, geometry=graph_geometry, device=self.device)
                  geom_cfg['graph'] = graph
                  density_num_states = geom_cfg['embedding_dim']
                geom_cls = getattr(rum.geometry, geom_cls_name)
                self.geom = geom_cls(**geom_cfg, dim=rewarder_cfg['num_states'])

            # Create information geometry.
            info_geom = None
            if 'info_geom_cfg' in rewarder_cfg:
                info_geom_cfg = rewarder_cfg.pop('info_geom_cfg')
                info_geom_class_name = info_geom_cfg.pop('name')
                info_geom_class = getattr(
                    rum.information_geometry, 
                    info_geom_class_name
                )
                info_geom = info_geom_class(**info_geom_cfg)


            # Create occupancy estimator.
            if 'density_cfg' in rewarder_cfg:
                density_cfg = rewarder_cfg.pop('density_cfg')
                density_class_name = density_cfg.pop('name')
                density_class = getattr(
                    rum.density, 
                    density_class_name,
                )
                self.density = density_class(
                    **density_cfg,
                    dim = density_num_states, #rewarder_cfg['num_states'],
                    information_geometry = info_geom,
                    geometry = rum.geometry.EuclideanGeometry(rewarder_cfg['num_states']),
                    device = device,
                )
                if self.geom is not None and geom_cls_name == 'EmbeddingGeometry': # Hacky due to circularity issue when bootstrapping geometry.
                    self.geom.graph.hack(self.geom, self.density)


            # Create info or goal reward.
            rewarder_cls_name = rewarder_cfg.pop('name')
            if rewarder_cls_name == 'DensityRewarder':
                assert info_geom is not None
                assert self.density is not None
                self.info_reward = InformationReward(
                    device=self.device, 
                    geom=self.geom, 
                    density=self.density, 
                    alpha_schedule=alpha_schedule,
                    beta_schedule=beta_schedule,
                    **rewarder_cfg)
            elif rewarder_cls_name == 'GoalRewarder':
                assert self.geom is not None and geom_cls_name == 'EmbeddingGeometry'
                del rewarder_cfg['num_states'] # Not used by GoalReward as it is inferred from geom.
                self.goal_reward = GoalReward(
                    geom=self.geom, 
                    beta_schedule=beta_schedule,
                    device=self.device, 
                    **rewarder_cfg)
            else:
                raise ValueError(rewarder_cls_name)

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        using_intrinsic_reward = self.rnd or self.info_reward or self.goal_reward
        # update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if using_intrinsic_reward:
            # rum hack
            if self.info_reward:
                rewarder = self.info_reward
                rnd_state = obs['rnd_state']
                if self.rnd: # Use RND embedding.
                    rnd_state = self.rnd.target(rnd_state).detach()
                    rnd_state = rnd_state.reshape(
                        (-1, self.rnd.num_outputs)
                    )               
                elif self.geom and isinstance(self.geom, rum.geometry.EmbeddingGeometry): # Use embedding geometry.
                    rnd_state = self.geom.network(rnd_state).detach()
                    rnd_state = rnd_state.reshape(
                        (-1, self.geom.embedding_dim) #this should be embedding dim
                    )
                # Compute the intrinsic rewards
                self.intrinsic_rewards = self.info_reward.get_intrinsic_reward(rnd_state)
            elif self.goal_reward:
                # Compute the intrinsic rewards
                self.intrinsic_rewards = self.goal_reward.get_intrinsic_reward(obs['rnd_state'])
            else:
                # Compute the intrinsic rewards
                self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # rum hack: Hack to correctly compute goal rewards.
        if self.goal_reward:
            self.goal_reward.new_trajectory = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd_optimizer:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # compute number of augmentations per sample
                # we assume policy group is always there and needs augmentation
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hidden_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # TODO: Move this processing to inside RND module.
            if self.rnd_optimizer:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # rum hack
        # Update density.
        if self.density:
            with torch.no_grad(): # TODO. Detaches unnecessary?
                density_states = get_rnd_state_hack(self.storage.observations, self.obs_groups)
                if self.rnd: # Use RND embedding.
                    density_states = self.rnd.target(density_states).detach()
                    density_states = density_states.reshape(
                        (-1, self.density.dim)
                    )               
                elif self.geom and isinstance(self.geom, rum.geometry.EmbeddingGeometry): # Use embedding geometry.
                    density_states = self.geom.network(density_states).detach()
                    density_states = density_states.reshape(
                        (-1, self.geom.embedding_dim) #this should be embedding dim
                    )
                    #density_states = density_states.reshape(
                    #    (-1, self.density.dim)
                    #)  
                elif self.info_reward: # Use info reward embedding.
                    density_states = density_states.reshape(
                        (-1, self.density.dim)
                    )               
                update_distances = (self.info_reward is not None)
                self.density.learn(density_states, update_distances = update_distances)
        # Update geometry model.
        if self.geom:
            rnd_state = get_rnd_state_hack(self.storage.observations, self.obs_groups)
            geom_state = rnd_state.reshape((-1, rnd_state.shape[-1]))
            self.geom.learn(geom_state)
        # Change goal.
        if self.goal_reward:
            rnd_state = get_rnd_state_hack(self.storage.observations, self.obs_groups)
            self.goal_reward.update_goal(rnd_state[0,0,:])

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd_optimizer:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd_optimizer:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd_optimizer:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd_optimizer:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
