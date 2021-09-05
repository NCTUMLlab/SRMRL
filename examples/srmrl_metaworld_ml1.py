#!/usr/bin/env python3
"""SRMRL ML1 example."""
import click
import metaworld

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SRMRL
from garage.torch.algos.srmrl import SRMRLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (SkillConditionedPolicy,
                                   TanhGaussianMLPPolicy)
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
import os

@click.command()
@click.option('--seed', default=1)
@click.option('--num_epochs', default=1000)
@click.option('--num_train_tasks', default=50)
@click.option('--num_test_tasks', default=10)
@click.option('--num_skills', default=10)
@click.option('--n_test_episodes', default=5)
@click.option('--encoder_hidden_size', default=200)
@click.option('--net_size', default=300)
@click.option('--num_steps_per_epoch', default=4000)
@click.option('--num_initial_steps', default=4000)
@click.option('--num_steps_prior', default=750)
@click.option('--num_extra_rl_steps_posterior', default=750)
@click.option('--batch_size', default=256)
@click.option('--embedding_batch_size', default=64)
@click.option('--embedding_mini_batch_size', default=64)
@click.option('--dist_class', default='Categorical')
@click.option('--env_name', default='push-v1')
@click.option('--soft/--hard', default=False)
@click.option('--v_contain_skill/--v_not_contain_skill', default=True)
@wrap_experiment
def SRMRL_metaworld_ml1(ctxt=None,
                             seed=1,
                             num_epochs=1000,
                             num_train_tasks=50,
                             num_test_tasks=10,
                             num_skills=10,
                             n_test_episodes=10,
                             latent_size=7,
                             encoder_hidden_size=200,
                             net_size=300,
                             meta_batch_size=16,
                             num_steps_per_epoch=4000,
                             num_initial_steps=4000,
                             num_tasks_sample=15,
                             num_steps_prior=750,
                             num_extra_rl_steps_posterior=750,
                             batch_size=256,
                             embedding_batch_size=64,
                             embedding_mini_batch_size=64,
                             reward_scale=10.,
                             dist_class='Categorical',
                             env_name='push-v1',
                             use_gpu=True,
                             normalize_reward=True,
                             soft=False,
                             disc_contain_obs=True,
                             v_contain_skill=True):
    """Train SRMRL with ML1 environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        num_epochs (int): Number of training epochs.
        num_train_tasks (int): Number of tasks for training.
        latent_size (int): Size of latent context vector.
        encoder_hidden_size (int): Output dimension of dense layer of the
            context encoder.
        net_size (int): Output dimension of a dense layer of Q-function and
            value function.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.
        normalize_reward(bool): Whether or not to normalize reward
        soft(bool): Whether or not to use soft skill embedding
        gmm(bool): Whether or not to use gmm policy
        disc_contain_obs(bool): Whether or not to contain obs in discriminator input
        v_contain_skill(bool): Whether or not to contain skill embedding in V function input

    """
    print(f'Run {env_name} seed : {seed} soft : {soft} v_contain_skill : {v_contain_skill}')
    set_seed(seed)
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)
    # create multi-task environment and sample tasks
    ml1 = metaworld.ML1(env_name)
    train_env = MetaWorldSetTaskEnv(ml1, 'train')
    env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                 env=train_env,
                                 wrapper=lambda env, _: normalize(env, normalize_reward=normalize_reward))
    env = env_sampler.sample(num_train_tasks)
    test_env = MetaWorldSetTaskEnv(ml1, 'test')
    test_env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                      env=test_env,
                                      wrapper=lambda env, _: normalize(env, normalize_reward=normalize_reward))

    trainer = Trainer(ctxt)

    # instantiate networks
    augmented_env = SRMRL.augment_env_spec(env[0](), latent_size + num_skills)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    if v_contain_skill:
        vf_env = SRMRL.get_env_spec(env[0](), latent_size + num_skills, 'vf')
    else:
        vf_env = SRMRL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    sampler = LocalSampler(agents=None,
                           envs=env[0](),
                           max_episode_length=env[0]().spec.max_episode_length,
                           n_workers=1,
                           worker_class=SRMRLWorker)

    srmrl = SRMRL(
        env=env,
        policy_class=SkillConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        sampler=sampler,
        dist_class=dist_class,
        soft=soft,
        disc_contain_obs=disc_contain_obs,
        v_contain_skill=v_contain_skill,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        num_skills=num_skills,
        n_test_episodes=n_test_episodes,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        reward_scale=reward_scale,
    )

    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        srmrl.to()

    trainer.setup(algo=srmrl, env=env[0]())

    trainer.train(n_epochs=num_epochs, batch_size=batch_size)


SRMRL_metaworld_ml1()
