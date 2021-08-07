"""PEARL and PEARLWorker in Pytorch.

Code is adapted from https://github.com/katerakelly/oyster.
"""

import copy

import akro
from dowel import logger, tabular
import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F

from garage import EnvSpec, InOutSpec, StepType, TimeStep
from garage.experiment import MetaEvaluator
from garage.np.algos import MetaRLAlgorithm
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker
from garage.torch import global_device
from garage.torch._functions import np_to_torch, zero_optim_grads
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import SkillConditionedPolicy


class SRMRL_h(MetaRLAlgorithm):
    # pylint: disable=too-many-statements
    def __init__(
            self,
            env,
            inner_policy,
            qf_low_1,
            qf_low_2,
            vf_low,
            qf_high_1,
            qf_high_2,
            vf_high,
            sampler,
            *,  # Mostly numbers after here.
            num_train_tasks,
            num_test_tasks,
            latent_dim,
            encoder_hidden_sizes,
            test_env_sampler,
            train_env_sampler=None,
            policy_class=SkillConditionedPolicy,
            encoder_class=MLPEncoder,
            dist_class='Categorical',
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3E-4,
            policy_mean_reg_coeff=1E-3,
            policy_std_reg_coeff=1E-3,
            policy_pre_activation_coeff=0.,
            soft_target_tau=0.005,
            kl_lambda=.1,
            optimizer_class=torch.optim.Adam,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            meta_batch_size=64,
            num_skills=10,
            n_test_episodes=10,
            num_steps_per_epoch=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=0,
            num_extra_rl_steps_posterior=100,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            update_post_train=1):

        self._env = env
        self._qf_low_1 = qf_low_1
        self._qf_low_2 = qf_low_2#copy.deepcopy(qf)
        self._vf_low = vf_low
        self._qf_high_1 = qf_high_1
        self._qf_high_2 = qf_high_2
        self._vf_high = vf_high
        self._num_train_tasks = num_train_tasks
        self._latent_dim = latent_dim

        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau
        self._kl_lambda = kl_lambda
        self._use_information_bottleneck = use_information_bottleneck
        self._use_next_obs_in_context = use_next_obs_in_context

        self._num_skills = num_skills
        self._dist_class = dist_class
        self._meta_batch_size = meta_batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_initial_steps = num_initial_steps
        self._num_tasks_sample = num_tasks_sample
        self._num_steps_prior = num_steps_prior
        self._num_steps_posterior = num_steps_posterior
        self._num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self._batch_size = batch_size
        self._embedding_batch_size = embedding_batch_size
        self._embedding_mini_batch_size = embedding_mini_batch_size
        self._discount = discount
        self._replay_buffer_size = replay_buffer_size
        self._reward_scale = reward_scale
        self._update_post_train = update_post_train
        self._task_idx = None
        self._single_env = env[0]()
        self.max_episode_length = self._single_env.spec.max_episode_length

        self._sampler = sampler

        self._is_resuming = False

        if num_test_tasks is None:
            num_test_tasks = test_env_sampler.n_tasks
        if num_test_tasks is None:
            raise ValueError('num_test_tasks must be provided if '
                             'test_env_sampler.n_tasks is None')

        worker_args = dict(deterministic=True, accum_context=True)
        self._test_tasks_evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
                                        n_test_tasks=num_test_tasks,
                                        n_test_episodes=n_test_episodes,
                                        worker_class=Myalgo_hWorker,
                                        worker_args=worker_args)
       
        self._train_tasks_evaluator = None
        if train_env_sampler is not None:
            self._train_tasks_evaluator = MetaEvaluator(test_task_sampler=train_env_sampler,
                                            n_test_tasks=num_train_tasks,
                                            n_test_episodes=n_test_episodes,
                                            prefix='MetaTrain',
                                            worker_class=Myalgo_hWorker,
                                            worker_args=worker_args)

        encoder_spec = self.get_env_spec(self._single_env, latent_dim,
                                         'encoder')
        encoder_in_dim = int(np.prod(encoder_spec.input_space.shape))
        encoder_out_dim = int(np.prod(encoder_spec.output_space.shape))
        context_encoder = encoder_class(input_dim=encoder_in_dim,
                                        output_dim=encoder_out_dim,
                                        hidden_sizes=encoder_hidden_sizes)

        # skill encoder input(obs, context_z)
        obs_dim = int(np.prod(self._single_env.observation_space.shape))
        action_dim = int(np.prod(self._single_env.action_space.shape))
        skill_in_dim = obs_dim + latent_dim 
        skill_out_dim = num_skills if dist_class == 'Categorical' else 2 * num_skills
        print(f'skill out dim : {skill_out_dim}')
        skill_encoder = encoder_class(input_dim=skill_in_dim,
                                      output_dim=skill_out_dim,
                                      hidden_sizes=encoder_hidden_sizes)

        # discriminator input(next_obs, action, context_z)
        discriminator_in_dim = obs_dim + action_dim + latent_dim
        discriminator_out_dim = num_skills if dist_class == 'Categorical' else 2 * num_skills
        print(f'disc out dim : {discriminator_out_dim}')
        self._discriminator = encoder_class(input_dim=discriminator_in_dim,
                                            output_dim=discriminator_out_dim,
                                            hidden_sizes=encoder_hidden_sizes)
        
        self._policy = policy_class(
            latent_dim=latent_dim,
            context_encoder=context_encoder,
            skill_encoder=skill_encoder,
            policy=inner_policy,
            use_information_bottleneck=use_information_bottleneck,
            use_next_obs=use_next_obs_in_context,
            dist_class=dist_class,
            H=1,
            num_skills=num_skills)

        # buffer for training RL update
        self._replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self.target_vf_low = copy.deepcopy(self._vf_low)
        self.target_vf_high = copy.deepcopy(self._vf_high)
        self.vf_criterion = torch.nn.MSELoss()

        self._policy_optimizer = optimizer_class(
            self._policy.networks[1].parameters(),
            lr=policy_lr,
        )
        self.qf_low_1_optimizer = optimizer_class(
            self._qf_low_1.parameters(),
            lr=qf_lr,
        )
        self.qf_low_2_optimizer = optimizer_class(
            self._qf_low_2.parameters(),
            lr=qf_lr,
        )
        self.vf_low_optimizer = optimizer_class(
            self._vf_low.parameters(),
            lr=vf_lr,
        )
        self.qf_high_1_optimizer = optimizer_class(
            self._qf_high_1.parameters(),
            lr=qf_lr,
        )
        self.qf_high_2_optimizer = optimizer_class(
            self._qf_high_2.parameters(),
            lr=qf_lr,
        )
        self.vf_high_optimizer = optimizer_class(
            self._vf_high.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self._policy.networks[0].parameters(),
            lr=context_lr,
        )
        self.skill_optimizer = optimizer_class(
            self._policy.networks[2].parameters(),
            lr=context_lr,
        )
        self.discriminator_optimizer = optimizer_class(
            self._discriminator.parameters(),
            lr=context_lr,
        )

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['_replay_buffers']
        del data['_context_replay_buffers']
        return data

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        self.__dict__.update(state)
        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._context_replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }
        self._is_resuming = True

    def train(self, trainer):
        """Obtain samples, train, and evaluate for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`Trainer..step_epochs()`, which provides services
                such as snapshotting and sampler control.

        """
        for _ in trainer.step_epochs():
            epoch = trainer.step_itr / self._num_steps_per_epoch

            # obtain initial set of samples from all train tasks
            if epoch == 0 or self._is_resuming:
                for idx in range(self._num_train_tasks):
                    self._task_idx = idx
                    self._obtain_samples(trainer, epoch,
                                         self._num_initial_steps, np.inf)
                    self._is_resuming = False

            # obtain samples from random tasks
            for _ in range(self._num_tasks_sample):
                idx = np.random.randint(self._num_train_tasks)
                self._task_idx = idx
                self._context_replay_buffers[idx].clear()
                # obtain samples with z ~ prior
                if self._num_steps_prior > 0:
                    self._obtain_samples(trainer, epoch, self._num_steps_prior,
                                         np.inf)
                # obtain samples with z ~ posterior
                if self._num_steps_posterior > 0:
                    self._obtain_samples(trainer, epoch,
                                         self._num_steps_posterior,
                                         self._update_post_train)
                # obtain extras samples for RL training but not encoder
                if self._num_extra_rl_steps_posterior > 0:
                    self._obtain_samples(trainer,
                                         epoch,
                                         self._num_extra_rl_steps_posterior,
                                         self._update_post_train,
                                         add_to_enc_buffer=False)

            logger.log('Training...')
            # sample train tasks and optimize networks
            self._train_once()
            trainer.step_itr += 1

            if self._train_tasks_evaluator is not None:
                logger.log('Evaluating on training tasks...')
                # evaluate
                self._policy.reset_belief()
                self._train_tasks_evaluator.evaluate(self)
            
            logger.log('Evaluating...')
            # evaluate
            self._policy.reset_belief()
            self._test_tasks_evaluator.evaluate(self)

    def _train_once(self):
        """Perform one iteration of training."""
        train_loss = {'Kl_loss' : 0,
                      '2qf_low_loss' : 0,
                      '2qf_high_loss' : 0,
                      'Vf_low_loss' : 0,
                      'Vf_high_loss' : 0,
                      'Policy_loss' : 0,
                      'Discriminator_loss' : 0,
                      'Skill_loss' : 0}

        for _ in range(self._num_steps_per_epoch):
            indices = np.random.choice(range(self._num_train_tasks),
                                       self._meta_batch_size)
            self._optimize_policy(indices, train_loss)
        
        for key, val in train_loss.items():
            tabular.record(key, val / self._num_steps_per_epoch)
        # logger.log(tabular)

    def _optimize_policy(self, indices, train_loss):
        """Perform algorithm optimizing.

        Args:
            indices (list): Tasks used for training.

        """
        num_tasks = len(indices)
        context = self._sample_context(indices)
        # clear context and reset belief of policy
        self._policy.reset_belief(num_tasks=num_tasks)

        # data shape is (task, batch, feat)
        obs, actions, rewards, next_obs, terms, skill_embed = self._sample_data(indices)
        policy_outputs, task_z, new_skill_dist, new_skill_embed = self._policy(obs, context, hierarchical=True, rsample=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flatten out the task dimensions
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        skill_embed = skill_embed.view(t * b, -1)

        # optimize qf and encoder networks
        q1_low_pred = self._qf_low_1(torch.cat([obs, actions, skill_embed], dim=1), task_z.detach())
        q2_low_pred = self._qf_low_2(torch.cat([obs, actions, skill_embed], dim=1), task_z.detach())
        v_low_pred = self._vf_low(torch.cat([obs, skill_embed], dim=1), task_z.detach())
        if self._dist_class == 'Categorical':
            q1_high_pred = self._qf_high_1(torch.cat([obs, task_z], dim=1))
            q2_high_pred = self._qf_high_2(torch.cat([obs, task_z], dim=1))
        else:
            q1_high_pred = self._qf_high_1(torch.cat([obs, skill_embed], dim=1), task_z)
            q2_high_pred = self._qf_high_2(torch.cat([obs, skill_embed], dim=1), task_z)
        v_high_pred = self._vf_high(obs, task_z.detach())

        with torch.no_grad():
            target_v_low_values = self.target_vf_low(torch.cat([next_obs, skill_embed], dim=1), task_z)
            target_v_high_values = self.target_vf_high(next_obs, task_z)

        # KL constraint on z if probabilistic
        zero_optim_grads(self.context_optimizer)
        if self._use_information_bottleneck:
            kl_div = self._policy.compute_kl_div()
            kl_loss = self._kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # print(f'q1_low_pred shape : {q1_low_pred.shape}')
        # print(f'q2_low_pred shape : {q2_low_pred.shape}')
        # print(f'v_low_pred shape : {v_low_pred.shape}')
        # print(f'q1_high_pred shape : {q1_high_pred.shape}')
        # print(f'q2_high_pred shape : {q2_high_pred.shape}')
        # print(f'v_high_pred shape : {v_high_pred.shape}')
        zero_optim_grads(self.qf_low_1_optimizer)
        zero_optim_grads(self.qf_low_2_optimizer)
        zero_optim_grads(self.qf_high_1_optimizer)
        zero_optim_grads(self.qf_high_2_optimizer)

        # compute intrinsic reward
        discriminator_out = self._discriminator(torch.cat([next_obs, new_actions.detach(), task_z.detach()], dim=1))
        if self._dist_class == 'Categorical':
            # print(f'discriminator_out shape : [{discriminator_out.shape}]')
            q_z = F.softmax(discriminator_out, dim=1)
            # print(f'softmax_discriminator_out shape : [{q_z.shape}]')
            p_z = new_skill_dist.probs
            # print(f'skill_dist shape : [{skill_dist.shape}]')
            # print(f'p_z_ shape : [{p_z.shape}]')
            log_q_z = torch.log((q_z * new_skill_embed.detach()).sum(dim=1, keepdim=True) + 1e-8)# D.kl_divergence(p_z, q_z).sum()
            log_p_z = -new_skill_dist.entropy()
        else:
            q_z_mean = discriminator_out[:, :self._num_skills]
            q_z_sigma_squared = F.softplus(discriminator_out[:, self._num_skills:])
            q_z_std = torch.sqrt(q_z_sigma_squared)
            q_z = D.Normal(q_z_mean, q_z_std)
            log_q_z = q_z.log_prob(new_skill_embed.detach()).sum(dim=1, keepdim=True)
            log_p_z = new_skill_dist.log_prob(new_skill_embed).sum(dim=1, keepdim=True)

        
        # if self._dist_class == 'Categorical':
        #     label = torch.where(new_skill_embed==1)[1]
        #     discriminator_loss = torch.nn.CrossEntropyLoss()(discriminator_out, label)
        # else:
        #     discriminator_loss = -log_q_z.mean()
        discriminator_loss = -log_q_z.mean()
        zero_optim_grads(self.discriminator_optimizer)
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # print(f'q_z shape : {q_z}')
        # # print(f'p_z shape : {p_z}')
        # print(f'log_q_z shape : {log_q_z.shape}')
        # print(f'log_p_z shape : {log_p_z.shape}')

        rewards_high = rewards.view(self._batch_size * num_tasks, -1)
        # print(f'rewards_flat shape : [{rewards_flat.shape}]')
        
        # final reward = env_reward + intrinsic_reward
        # rewards_high += log_q_z.detach() #- log_p_z.detach()
        rewards_high = rewards_high * self._reward_scale

        rewards_low = log_q_z.detach()
        rewards_low = rewards_low * self._reward_scale

        # print(f'rewards_high shape : {rewards_high.shape}')
        # print(f'rewards_low shape : {rewards_low.shape}')
        terms_flat = terms.view(self._batch_size * num_tasks, -1)
        q_high_target = rewards_high + (
            1. - terms_flat) * self._discount * target_v_high_values

        if self._dist_class == 'Categorical':
            q1_high_pred = (q1_high_pred * skill_embed).sum(dim=-1, keepdim=True)
            q2_high_pred = (q2_high_pred * skill_embed).sum(dim=-1, keepdim=True)

        # print(f'q_high_target shape : {q_high_target.shape}')
        # print(f'q1_high_pred shape : {q1_high_pred.shape}')
        # print(f'q2_high_pred shape : {q2_high_pred.shape}')
        qf_high_loss = torch.mean((q1_high_pred - q_high_target)**2) + torch.mean(
            (q2_high_pred - q_high_target)**2)
        
        q_low_target = rewards_low + (
            1. - terms_flat) * self._discount * target_v_low_values
        # print(f'q_low_target shape : {q_low_target.shape}')
        qf_low_loss = torch.mean((q1_low_pred - q_low_target)**2) + torch.mean(
            (q2_low_pred - q_low_target)**2)
        total_qf_loss = qf_high_loss + qf_low_loss
        total_qf_loss.backward()

        self.qf_low_1_optimizer.step()
        self.qf_low_2_optimizer.step()
        self.qf_high_1_optimizer.step()
        self.qf_high_2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        q_low_1 = self._qf_low_1(torch.cat([obs, new_actions, new_skill_embed.detach()], dim=1), task_z.detach())
        q_low_2 = self._qf_low_2(torch.cat([obs, new_actions, new_skill_embed.detach()], dim=1), task_z.detach())
        if self._dist_class == 'Categorical':
            q_high_1 = self._qf_high_1(torch.cat([obs, task_z.detach()], dim=1))
            q_high_2 = self._qf_high_2(torch.cat([obs, task_z.detach()], dim=1))
        else:
            q_high_1 = self._qf_high_1(torch.cat([obs, new_skill_embed], dim=1), task_z.detach())
            q_high_2 = self._qf_high_2(torch.cat([obs, new_skill_embed], dim=1), task_z.detach())
        min_q_low = torch.min(q_low_1, q_low_2)
        min_q_high = torch.min(q_high_1, q_high_2)

        # optimize vf
        v_low_target = min_q_low - log_pi
        vf_low_loss = self.vf_criterion(v_low_pred, v_low_target.detach())
        zero_optim_grads(self.vf_low_optimizer)
        vf_low_loss.backward()
        self.vf_low_optimizer.step()

        if self._dist_class == 'Categorical':
            v_high_target = (p_z * min_q_high).sum(dim=-1, keepdim=True) - log_p_z.unsqueeze(-1)
        else:
            v_high_target = min_q_high - log_p_z
        vf_high_loss = self.vf_criterion(v_high_pred, v_high_target.detach())
        zero_optim_grads(self.vf_high_optimizer)
        vf_high_loss.backward()
        self.vf_high_optimizer.step()
        # print(f'q_low_1 shape : {q_low_1.shape}')
        # print(f'q_low_2 shape : {q_low_2.shape}')
        # print(f'q_high_1 shape : {q_high_1.shape}')
        # print(f'q_high_2 shape : {q_high_2.shape}')
        # print(f'v_low_target shape : {v_low_target.shape}')
        # print(f'v_high_target shape : {v_high_target.shape}')
        # print(f'log_pi shape : {log_pi.shape}')
        self._update_target_network()
        
        # optimize policy
        log_policy_target_low = min_q_low
        policy_loss_low = (log_pi - log_policy_target_low).mean()

        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean**2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self._policy_pre_activation_coeff * (
            (pre_tanh_value**2).sum(dim=1).mean())
        policy_reg_loss = (mean_reg_loss + std_reg_loss +
                           pre_activation_reg_loss)
        
        policy_loss_low = policy_loss_low + policy_reg_loss
        zero_optim_grads(self._policy_optimizer)
        policy_loss_low.backward()
        self._policy_optimizer.step()

        log_skill_target = min_q_high
        if self._dist_class == 'Categorical':
            skill_loss = (log_p_z - (p_z * log_skill_target).sum(dim=-1)).mean()
        else:
            skill_loss = (log_p_z - log_skill_target).mean()
            mean_reg_loss = self._policy_mean_reg_coeff * (new_skill_dist.loc**2).mean()
            std_reg_loss = self._policy_std_reg_coeff * (new_skill_dist.scale**2).mean()
            skill_loss += mean_reg_loss + std_reg_loss
        zero_optim_grads(self.skill_optimizer)
        skill_loss.backward()
        self.skill_optimizer.step()

        train_loss['Kl_loss'] += kl_loss.item()
        train_loss['2qf_low_loss'] += qf_low_loss.item()
        train_loss['2qf_high_loss'] += qf_high_loss.item()
        train_loss['Vf_low_loss'] += vf_low_loss.item()
        train_loss['Vf_high_loss'] += vf_high_loss.item()
        train_loss['Policy_loss'] += policy_loss_low.item()
        train_loss['Discriminator_loss'] += discriminator_loss.item()
        train_loss['Skill_loss'] += skill_loss.item()

    def _obtain_samples(self,
                        trainer,
                        itr,
                        num_samples,
                        update_posterior_rate,
                        add_to_enc_buffer=True):
        """Obtain samples.

        Args:
            trainer (Trainer): Trainer.
            itr (int): Index of iteration (epoch).
            num_samples (int): Number of samples to obtain.
            update_posterior_rate (int): How often (in episodes) to infer
                posterior of policy.
            add_to_enc_buffer (bool): Whether or not to add samples to encoder
                buffer.

        """
        self._policy.reset_belief()
        total_samples = 0

        if update_posterior_rate != np.inf:
            num_samples_per_batch = (update_posterior_rate *
                                     self.max_episode_length)
        else:
            num_samples_per_batch = num_samples

        while total_samples < num_samples:
            paths = trainer.obtain_samples(itr, num_samples_per_batch,
                                           self._policy,
                                           self._env[self._task_idx])
            total_samples += sum([len(path['rewards']) for path in paths])

            for path in paths:
                p = {
                    'observations':
                    path['observations'],
                    'actions':
                    path['actions'],
                    'rewards':
                    path['rewards'].reshape(-1, 1),
                    'skill_embed':
                    path['agent_infos']['skill_embed'],
                    'next_observations':
                    path['next_observations'],
                    'dones':
                    np.array([
                        step_type == StepType.TERMINAL
                        for step_type in path['step_types']
                    ]).reshape(-1, 1)
                }
                # print(p)
                # print(p['actions'].shape)
                # print(p['skill_embed'].shape)
                self._replay_buffers[self._task_idx].add_path(p)

                if add_to_enc_buffer:
                    self._context_replay_buffers[self._task_idx].add_path(p)

            if update_posterior_rate != np.inf:
                context = self._sample_context(self._task_idx)
                self._policy.infer_posterior(context)

    def _sample_data(self, indices):
        """Sample batch of training data from a list of tasks.

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Obervations, with shape :math:`(X, N, O^*)` where X
                is the number of tasks. N is batch size.
            torch.Tensor: Actions, with shape :math:`(X, N, A^*)`.
            torch.Tensor: Rewards, with shape :math:`(X, N, 1)`.
            torch.Tensor: Next obervations, with shape :math:`(X, N, O^*)`.
            torch.Tensor: Dones, with shape :math:`(X, N, 1)`.

        """
        # transitions sampled randomly from replay buffer
        initialized = False
        for idx in indices:
            batch = self._replay_buffers[idx].sample_transitions(
                self._batch_size)
            if not initialized:
                o = batch['observations'][np.newaxis]
                a = batch['actions'][np.newaxis]
                r = batch['rewards'][np.newaxis]
                no = batch['next_observations'][np.newaxis]
                d = batch['dones'][np.newaxis]
                s = batch['skill_embed'][np.newaxis]
                initialized = True
            else:
                o = np.vstack((o, batch['observations'][np.newaxis]))
                a = np.vstack((a, batch['actions'][np.newaxis]))
                r = np.vstack((r, batch['rewards'][np.newaxis]))
                no = np.vstack((no, batch['next_observations'][np.newaxis]))
                d = np.vstack((d, batch['dones'][np.newaxis]))
                s = np.vstack((s, batch['skill_embed'][np.newaxis]))

        o = np_to_torch(o)
        a = np_to_torch(a)
        r = np_to_torch(r)
        no = np_to_torch(no)
        d = np_to_torch(d)
        s = np_to_torch(s)
        return o, a, r, no, d, s

    def _sample_context(self, indices):
        """Sample batch of context from a list of tasks.

        Args:
            indices (list): List of task indices to sample from.

        Returns:
            torch.Tensor: Context data, with shape :math:`(X, N, C)`. X is the
                number of tasks. N is batch size. C is the combined size of
                observation, action, reward, and next observation if next
                observation is used in context. Otherwise, C is the combined
                size of observation, action, and reward.

        """
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        initialized = False
        for idx in indices:
            batch = self._context_replay_buffers[idx].sample_transitions(
                self._embedding_batch_size)
            o = batch['observations']
            a = batch['actions']
            r = batch['rewards']
            context = np.hstack((np.hstack((o, a)), r))
            if self._use_next_obs_in_context:
                context = np.hstack((context, batch['next_observations']))

            if not initialized:
                final_context = context[np.newaxis]
                initialized = True
            else:
                final_context = np.vstack((final_context, context[np.newaxis]))

        final_context = np_to_torch(final_context)

        if len(indices) == 1:
            final_context = final_context.unsqueeze(0)

        return final_context

    def _update_target_network(self):
        """Update parameters in the target vf network."""
        for target_param, param in zip(self.target_vf_low.parameters(),
                                       self._vf_low.parameters()):
            target_param.data.copy_(target_param.data *
                                    (1.0 - self._soft_target_tau) +
                                    param.data * self._soft_target_tau)

        for target_param, param in zip(self.target_vf_high.parameters(),
                                       self._vf_high.parameters()):
            target_param.data.copy_(target_param.data *
                                    (1.0 - self._soft_target_tau) +
                                    param.data * self._soft_target_tau)

    @property
    def policy(self):
        """Return all the policy within the model.

        Returns:
            garage.torch.policies.Policy: Policy within the model.

        """
        return self._policy

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return self._policy.networks + [self._policy] + [
            self._qf_low_1, self._qf_low_2, self._vf_low, self.target_vf_low, self._discriminator,
            self._qf_high_1, self._qf_high_2, self._vf_high, self.target_vf_high
        ]

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            Policy: The policy used to obtain samples that are later used for
                meta-RL adaptation.

        """
        return self._policy

    def adapt_policy(self, exploration_policy, exploration_episodes):
        """Produce a policy adapted for a task.

        Args:
            exploration_policy (Policy): A policy which was returned from
                get_exploration_policy(), and which generated
                exploration_episodes by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_episodes (EpisodeBatch): Episodes to which to adapt,
                generated by exploration_policy exploring the
                environment.

        Returns:
            Policy: A policy adapted to the task represented by the
                exploration_episodes.

        """
        total_steps = sum(exploration_episodes.lengths)
        o = exploration_episodes.observations
        a = exploration_episodes.actions
        r = exploration_episodes.rewards.reshape(total_steps, 1)
        ctxt = np.hstack((o, a, r)).reshape(1, total_steps, -1)
        context = np_to_torch(ctxt)
        self._policy.infer_posterior(context)

        return self._policy

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)

    @classmethod
    def augment_env_spec(cls, env_spec, latent_dim, action_dim=None):
        """Augment environment by a size of latent dimension.

        Args:
            env_spec (EnvSpec): Environment specs to be augmented.
            latent_dim (int): Latent dimension.

        Returns:
            EnvSpec: Augmented environment specs.

        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape)) if action_dim is None else action_dim
        aug_obs = akro.Box(low=-1,
                           high=1,
                           shape=(obs_dim + latent_dim, ),
                           dtype=np.float32)
        aug_act = akro.Box(low=-1,
                           high=1,
                           shape=(action_dim, ),
                           dtype=np.float32)
        return EnvSpec(aug_obs, aug_act)

    @classmethod
    def get_env_spec(cls, env_spec, latent_dim, module):
        """Get environment specs of encoder with latent dimension.

        Args:
            env_spec (EnvSpec): Environment specification.
            latent_dim (int): Latent dimension.
            module (str): Module to get environment specs for.

        Returns:
            InOutSpec: Module environment specs with latent dimension.

        """
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        if module == 'encoder':
            in_dim = obs_dim + action_dim + 1
            out_dim = latent_dim * 2
        elif module == 'vf':
            in_dim = obs_dim
            out_dim = latent_dim
        in_space = akro.Box(low=-1, high=1, shape=(in_dim, ), dtype=np.float32)
        out_space = akro.Box(low=-1,
                             high=1,
                             shape=(out_dim, ),
                             dtype=np.float32)
        if module == 'encoder':
            spec = InOutSpec(in_space, out_space)
        elif module == 'vf':
            spec = EnvSpec(in_space, out_space)

        return spec


class SRMRL_hWorker(DefaultWorker):
    """A worker class used in sampling for PEARL.

    It stores context and resample belief in the policy every step.

    Args:
        seed (int): The seed to use to intialize random number generators.
        max_episode_length(int or float): The maximum length of episodes which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
        deterministic (bool): If True, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
        accum_context (bool): If True, update context of the agent.

    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.

    """

    def __init__(self,
                 *,
                 seed,
                 max_episode_length,
                 worker_number,
                 deterministic=True,
                 accum_context=False):
        self._deterministic = deterministic
        self._accum_context = accum_context
        self._episode_info = None
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)

    def start_episode(self):
        """Begin a new episode."""
        self._eps_length = 0
        self._prev_obs, self._episode_info = self.env.reset()

    def step_episode(self):
        """Take a single time-step in the current episode.

        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            if self._deterministic:
                a = agent_info['mean']
            # a, agent_info = self.agent.get_action(self._prev_obs)
            es = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if self._accum_context:
                s = TimeStep.from_env_step(env_step=es,
                                           last_observation=self._prev_obs,
                                           agent_info=agent_info,
                                           episode_info=self._episode_info)
                self.agent.update_context(s)
            if not es.last:
                self._prev_obs = es.observation
                return False
        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)
        return True

    def rollout(self):
        """Sample a single episode of the agent in the environment.

        Returns:
            EpisodeBatch: The collected episode.

        """
        self.agent.sample_from_belief()
        self.start_episode()
        while not self.step_episode():
            pass
        return self.collect_episode()
