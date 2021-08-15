import metaworld
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment.task_sampler import SetTaskSampler
from garage.experiment.deterministic import set_seed
from garage.torch import set_gpu_mode
from garage.experiment import Snapshotter
from garage.sampler import DefaultWorker, LocalSampler, WorkerFactory
from garage.torch.algos.srmrl import SRMRLWorker
from garage import EpisodeBatch, TimeStep
from PIL import Image
import numpy as np
import mujoco_py
import click
import os

def sim_policy(model_dir, output_dir, deterministic=True, n_test_tasks=10, n_exploration_eps=10):
    set_gpu_mode(True, gpu_id=0)
    snapshotter = Snapshotter()
    data = snapshotter.load(model_dir)
    algo = data['algo']
    policy = algo.policy
    # the env is the normalized env
    env = data['env']
    seed = data['seed']
    set_seed(seed)
    task_name = env._env._env_list[0]
    normalize_reward = env._normalize_reward

    ml1 = metaworld.ML1(task_name)
    test_env = MetaWorldSetTaskEnv(ml1, 'test')
    test_env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                      env=test_env,
                                      wrapper=lambda env, _: normalize(env, normalize_reward=True))

    print('Sampling for adapation and meta-testing...')
    env_updates = test_env_sampler.sample(n_test_tasks)
    env = env_updates[0]()
    # collect adaptation trajectories
    print('Collecting context for adapatation ...')
    test_sampler = LocalSampler.from_worker_factory(
                WorkerFactory(seed=seed,
                              max_episode_length=env.spec.max_episode_length,
                              n_workers=1,
                              worker_class=SRMRLWorker,
                              worker_args=dict(deterministic=True, accum_context=True)),
                agents=policy,
                envs=env)

    adapted_episodes = []
    for i, env_up in enumerate(env_updates):
        print(f'In {i}th env ...')
        eps = EpisodeBatch.concatenate(*[
                    test_sampler.obtain_samples(i, 1, policy,
                                                    env_up)
                    for _ in range(n_exploration_eps)
                ])
        policy = algo.adapt_policy(policy, eps)
        
        for _ in range(5):
            # env_up() return a constructed env
            env = env_up()
            under_env = env._env._current_env._env._env
            # env = e._current_env._env._env
            obs, info = env.reset()
            under_env.viewer = mujoco_py.MjRenderContextOffscreen(under_env.sim, -1) 
            under_env.viewer.cam.azimuth = -70
            under_env.viewer.cam.elevation = -10
            under_env.viewer.cam.distance = 1.5
            under_env.viewer.cam.lookat[0] = 0.2
            under_env.viewer.cam.lookat[1] = 0
            under_env.viewer.cam.lookat[2] = 0

            print('Begin roll out ...')
            total_steps = 0
            frames = []
            success = []
            while total_steps < env.spec.max_episode_length:
                a, agent_info = policy.get_action(obs)
                if deterministic:
                    a = agent_info['mean']
                es = env.step(a)
                success.append(es.env_info['success'])

                s = TimeStep.from_env_step(
                    env_step=es,
                    last_observation=obs,
                    agent_info=agent_info,
                    episode_info=None)
                policy.update_context(s)

                frame = under_env.sim.render(
                        680, 480, mode='offscreen'
                    )
                image = Image.fromarray(np.flipud(frame))
                frames.append(image)
                total_steps += 1

                if es.last: #or es.env_info['success']:
                    break

                obs = es.observation
                
            print(f'End at : {total_steps}')
            success = np.array(success)
            print(f"Sucess at last : {es.env_info['success']}'")
            print(f"Sucess in mid : {success.any()}")
            print(f"Success at steps : {np.where(success==1.0)}")

            if success.any():
                temp_dir = os.path.join(output_dir, task_name, str(i))
                frame_dir = os.path.join(temp_dir, 'frames')
                os.makedirs(frame_dir, exist_ok=True)

                print('Save frames')
                for j, frame in enumerate(frames):
                    frame.save(os.path.join(frame_dir, '%06d.jpg' % j))
                os.system('ffmpeg -r {} -i {}/%06d.jpg {}/{}.gif'.format(30, frame_dir, temp_dir, task_name + '-' + str(i)))

                break

@click.command()
@click.option('--model_dir', required=True)
@click.option('--output_dir', default='.')
def main(model_dir, output_dir):
    sim_policy(model_dir, output_dir)

if __name__ == '__main__':
    main()

