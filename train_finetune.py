import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from typing import Tuple
import cloudpickle as pickle

import datetime
import json
import time
import gym
import numpy as np
import sys
import tqdm
import absl
from absl import app, flags
from ml_collections import config_flags
from ml_collections.config_dict.config_dict import ConfigDict
from tensorboardX import SummaryWriter
from dataclasses import dataclass
from matplotlib import pyplot as plt

import wrappers
from dataset_utils import (Batch, D4RLDataset, ReplayBuffer, BinaryDataset,
                           split_into_trajectories, load_dataset_h5py)
from evaluation import evaluate
from learner import Learner
import wandb

from functools import singledispatch

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-medium-replay-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('offline_dataset_fp', None, 'Path to offline dataset.')
flags.DEFINE_string('off_policy_algo', 'sac', 'sac or td3')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
#int(1e6)
flags.DEFINE_integer('max_steps', int(1e4), 'Number of training steps.')
#int(2e5)
flags.DEFINE_integer('num_pretraining_steps', int(5e3),
                     'Number of pretraining steps.')
flags.DEFINE_integer('offline_validation_budget', int(5e4), 'Number of offline validation steps.')
# 20, 5
flags.DEFINE_integer('eval_episodes_per_evaluation', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_float('temp', 2, 'Loss temperature')
flags.DEFINE_float('tau_actor', 0.005, 'actor moving average')
flags.DEFINE_float('min_temp_online', 0., 'min online temp')
flags.DEFINE_boolean('load_dataset_from_path', False, 'Load dataset from path')
flags.DEFINE_boolean('ablation', False, 'For experiments management')
flags.DEFINE_boolean('bc_pretrain', False, 'reward-free pretraining')
flags.DEFINE_boolean('double', True, 'Use double q-learning when offline pretrain')
flags.DEFINE_boolean('double_online', True, 'Use double q-learning when online finetune')
flags.DEFINE_boolean('offline_dataset_use_buffer', True, 'Use replay buffer as the offline dataset')
flags.DEFINE_integer('replay_buffer_size', None,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', None,
                     'Offline data size (uses all data if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('vanilla', False, 'Use vanilla RL training')
flags.DEFINE_boolean('auto_alpha', True, 'SAC temperature auto adjustment')
flags.DEFINE_boolean('symmetric', True, 'symmetric sampling trick')
flags.DEFINE_boolean('entropy_backup', True, 'entropy backup when update critic')
flags.DEFINE_integer('sample_random_times', 0, 'Number of random actions to add to smooth dataset')
flags.DEFINE_integer('utd', 1, 'Number of gradient updates per online sample')
flags.DEFINE_float('decay_speed', 10, 'online temperature decay speed')
flags.DEFINE_boolean('grad_pen', False, 'Add a gradient penalty to critic network')
flags.DEFINE_float('lambda_gp', 1, 'Gradient penalty coefficient')
flags.DEFINE_float('max_clip', 7., 'Loss clip value')
flags.DEFINE_boolean('log_loss', False, 'Use log gumbel loss')
flags.DEFINE_boolean('noise', False, 'Add noise to actions')

flags.DEFINE_string('CUDA_id', '2', 'CUDA_VISIBLE_DEVICES')


# if ('pen' in FLAGS.env_name) or ('hammer' in FLAGS.env_name) or ('door' in FLAGS.env_name) or ('relocate' in FLAGS.env_name) or ('kitchen' in FLAGS.env_name):
#     
# else:
# config_path = 'configs/adroit_config.py'
config_path = 'configs/mujoco_config_finetune.py'

config_flags.DEFINE_config_file(
    'config',
    config_path,
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def save_pickle(obj, filepath):
    with open(filepath, 'wb') as fout:
        pickle.dump(obj, fout)

def load_pickle(filepath):
    with open(filepath, 'rb') as fin:
        data = pickle.load(fin)
    return data

@dataclass(frozen=True)
class ConfigArgs:
    sample_random_times: int
    grad_pen: bool
    noise: bool
    lambda_gp: int
    max_clip: float
    utd: int
    sac: bool
    auto_alpha: bool
    entropy_backup: bool
    bc_pretrain: bool
    # dual: bool
    log_loss: bool
    # log_grad: bool
    # euler_bias: bool
    # mod: bool


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0
    return 1000/(compute_returns(trajs[-1]) - compute_returns(trajs[0]))


def make_env_and_dataset(env_name: str,
                         seed: int,
                         load_dataset_from_path=False) -> Tuple[gym.Env, gym.Env, D4RLDataset, float]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    env_eval = gym.make(env_name)

    env_eval = wrappers.EpisodeMonitor(env_eval)
    env_eval = wrappers.SinglePrecision(env_eval)

    env_eval.seed(seed)
    env_eval.action_space.seed(seed)
    env_eval.observation_space.seed(seed)

    if load_dataset_from_path:
        dataset, metadata = load_dataset_h5py(env_name)
    else:
        if 'binary' in env_name:
            dataset = BinaryDataset(env)
        else:
            dataset = D4RLDataset(env)

    env_name_lower = env_name.lower()
    if ('halfcheetah' in env_name_lower or 'walker2d' in env_name_lower
          or 'hopper' in env_name_lower):
        normalize_factor = normalize(dataset)
    else:
        normalize_factor = 1.
    return env, env_eval, dataset, normalize_factor


def symmetric_sample(replay_buffer, replay_buffer_online, batch_size):
    indx_off = np.random.randint(replay_buffer.size, size=int(batch_size/2))
    indx_on = np.random.randint(replay_buffer_online.size, size=int(batch_size/2))
    
    return Batch(observations=np.concatenate([replay_buffer.observations[indx_off], replay_buffer_online.observations[indx_on]], axis=0),
                    actions=np.concatenate([replay_buffer.actions[indx_off], replay_buffer_online.actions[indx_on]], axis=0),
                    rewards=np.concatenate([replay_buffer.rewards[indx_off], replay_buffer_online.rewards[indx_on]], axis=0),
                    masks=np.concatenate([replay_buffer.masks[indx_off], replay_buffer_online.masks[indx_on]], axis=0),
                    next_observations=np.concatenate([replay_buffer.next_observations[indx_off], replay_buffer_online.next_observations[indx_on]], axis=0))

max_steps_per_episode_dict = {
    "kitchen-complete-v0": 280,
    "kitchen-mixed-v0": 280,
    "kitchen-new-v0": 280,
    "pen-expert-v0": 100,
    "door-expert-v0": 200,
    "hammer-expert-v0": 200,
    "Ant-v2": 1000,
    "Hopper-v2": 1000,
    "Walker2d-v2": 1000,
    "maze2d-umaze-v1": 300,
    "maze2d-medium-v1": 600,
    "maze2d-large-v1": 800,
}

def main(_):
    if 'halfcheetah' in FLAGS.env_name:
        FLAGS.config.layernorm = False
    
    symmetric = FLAGS.symmetric
    np.random.seed(FLAGS.seed)
    
    offline_dataset_tag = "UseBuffer" if FLAGS.offline_dataset_use_buffer else "UseNewData"
    
    wandb.init(project='Proto',
               sync_tensorboard=True, reinit=True,  settings=wandb.Settings(_disable_stats=True))
    wandb.config.update(flags.FLAGS)
    wandb.run.name = f"{FLAGS.env_name}_{offline_dataset_tag}_{FLAGS.seed}_{FLAGS.temp}"

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(FLAGS.save_dir, f"{wandb.run.name}_{ts_str})

    hparam_str_dict = dict(seed=FLAGS.seed, env=FLAGS.env_name)
    hparam_str = ','.join([
        '%s=%s' % (k, str(hparam_str_dict[k]))
        for k in sorted(hparam_str_dict.keys())
    ])

    summary_writer = SummaryWriter(os.path.join(save_dir, 'tb', hparam_str), write_to_disk=True)
    os.makedirs(save_dir, exist_ok=True)

    # save experiment configuration to file
    flags_dict = absl.flags.FLAGS.flags_by_module_dict()
    flags_dict = {k: {v.name: v.value for v in vs} for k, vs in flags_dict.items()}
    expr_config_filepath = f"{save_dir}/expr_config.json"
    expr_config_dict = {k:(v.to_dict() if isinstance(v, ConfigDict) else v) for k, v in flags_dict.items()}
    with open(expr_config_filepath, "w") as f:
        json.dump(expr_config_dict, f, indent=4, default=to_serializable)

    # load offline dataset
    offline_dataset, _ = load_dataset_h5py(FLAGS.offline_dataset_fp)

    # offline RL validation setup: similar to the setup for online RL validation
    offline_eval_episodes = int(FLAGS.offline_validation_budget / max_steps_per_episode_dict[FLAGS.env_name])
    offline_evals = offline_eval_episodes//FLAGS.eval_episodes_per_evaluation
    offline_eval_epochs = list(
        range(FLAGS.num_pretraining_steps, 0, -1*(FLAGS.num_pretraining_steps//(offline_evals-1)))
        )
    if (len(offline_eval_epochs) + 1) == offline_evals:
        # the agent at the step 0 definitely has a randomly initialized policy. So, do not evaluate here
        offline_eval_epochs.append(offline_eval_epochs[-1]//2) # the step at the middle of step 0 and the the currently earliest evaluation step

    # help save the best offline agent and its performance
    offline_eval_filepath = f"{save_dir}/eval_ave_episode_return_offline{offline_dataset_tag}.txt"
    with open(offline_eval_filepath, "w") as f:
        f.write(f"Experiment Time\tEpoch\tReturn\n")
    offline_eval_file = open(offline_eval_filepath, "a")
    best_offline_agent_epoch = -1
    best_offline_agent_perf = -np.inf
    best_offline_agent_filename = f"best_offline_agent_{offline_dataset_tag}.pkl"
    best_offline_agent_filepath = f"{save_dir}/{best_offline_agent_filename}"

    # help log the evaluation of the fine-tuning process
    ft_eval_filepath = f"{save_dir}/eval_ave_episode_return_ft.txt"
    with open(ft_eval_filepath, "w") as f:
        f.write(f"Experiment Time\tEpoch\tReturn\n")
    ft_eval_file = open(ft_eval_filepath, "a")


    env, env_eval, dataset, normalize_factor = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, FLAGS.load_dataset_from_path)

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 np.maximum(int(2e+6), len(dataset.observations)))
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    # symmetric sampling
    if symmetric:
        replay_buffer_online = ReplayBuffer(env.observation_space, action_dim, FLAGS.max_steps)
        replay_buffer_online.initialize_with_dataset(dataset, 10000)
    
    kwargs = dict(FLAGS.config)
    wandb.config.update(kwargs)

    args = ConfigArgs(sample_random_times=FLAGS.sample_random_times,
                      grad_pen=FLAGS.grad_pen,
                      lambda_gp=FLAGS.lambda_gp,
                      noise=FLAGS.noise,
                      max_clip=FLAGS.max_clip,
                      utd=FLAGS.utd,
                      log_loss=FLAGS.log_loss,
                      sac=True,
                      auto_alpha=FLAGS.auto_alpha,
                      entropy_backup=FLAGS.entropy_backup,
                      bc_pretrain=FLAGS.bc_pretrain)
    
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    loss_temp=FLAGS.temp,
                    double_q=FLAGS.double,
                    double_q_online=FLAGS.double_online,
                    vanilla=FLAGS.vanilla,
                    auto_alpha=FLAGS.auto_alpha,
                    tau_actor=FLAGS.tau_actor,
                    args=args,
                    **kwargs)

    best_eval_returns = -np.inf
    eval_returns = []
    observation, done = env.reset(), False

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1 - FLAGS.num_pretraining_steps,
                             FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i >= 1:
            action = agent.sample_actions(observation, offline=False)
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            
            # symmetric sampling
            if symmetric:
                replay_buffer_online.insert(observation, action, reward * normalize_factor, mask, float(done), next_observation)
            else:
                replay_buffer.insert(observation, action, reward * normalize_factor, mask, float(done), next_observation)                
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                summary_writer.add_scalar(f'steps', i, info['total']['timesteps'])
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'online_samples/{k}', v, info['total']['timesteps'])
        else:
            info = {}
            info['total'] = {'timesteps': i}

        if i >= 1 and symmetric:
            # online symmetric sampling
            batch = symmetric_sample(replay_buffer, replay_buffer_online, FLAGS.batch_size * FLAGS.utd)
        elif i < 1:
            # offline sampling
            batch = replay_buffer.sample(FLAGS.batch_size)
        else:
            # online sampling
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd)
            
        if 'antmaze' in FLAGS.env_name:
            batch = Batch(observations=batch.observations,
                          actions=batch.actions,
                          rewards=batch.rewards - 1,
                          masks=batch.masks,
                          next_observations=batch.next_observations)
        if i < 0:
            update_info = agent.update(batch, offline=True)  # offline
        elif i == 0:
            update_info = agent.update(batch, offline=True)  # offline
            offline = True
            eval_stats = evaluate(agent, env_eval, FLAGS.eval_episodes_per_evaluation, offline)
            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            offline_eval_file.write(f"{expr_time_now_str}\t{epoch}\t{eval_stats['return']}\n")
            offline_eval_file.flush()

            if best_offline_agent_perf < eval_stats['return']:
                best_offline_agent_epoch = i
                best_offline_agent_perf = eval_stats['return']
                save_data = {'agent': agent, 'epoch': i, 'return': eval_stats['return']}
                save_pickle(save_data, best_offline_agent_filename)
            offline_eval_file.write(f"**Best Policy**\t{best_offline_agent_epoch}\t{best_offline_agent_perf}\n")
            offline_eval_file.flush()
            offline_eval_file.close()
            # since the best pooling is used in the offline validation, the best offline-optimized agent is used as the initial agent of the fine-tuning process
            best_offline_agent = load_pickle(best_offline_agent_filepath)
            if "agent" in best_offline_agent:
                agent = best_offline_agent['agent']

            # offline2online transfer
            agent.offline2online()
            # Free the offline replay buffer, use online buffer to better boost the performance
            if not symmetric:
                replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                            np.maximum(int(2e+6), len(dataset.observations)))
                replay_buffer.initialize_with_dataset(dataset, 25000)
        else:
            update_info = agent.update(batch, offline=False)  # online
            agent.loss_temp_online = np.maximum(agent.loss_temp_online - FLAGS.temp / (FLAGS.max_steps/FLAGS.decay_speed), FLAGS.min_temp_online)  # temp annealing

        if i % FLAGS.log_interval == 0:
            summary_writer.add_scalar(f'hyperparameter/temperature', agent.loss_temp, i)
            summary_writer.add_scalar(f'hyperparameter/tau_actor', agent.tau_actor, i)
            summary_writer.add_scalar(f'hyperparameter/buffer_size', replay_buffer.size, i)
            summary_writer.add_scalar(f'hyperparameter/temperature_online', agent.loss_temp_online, i)
            summary_writer.add_scalar(f'hyperparameter/insert_index', replay_buffer.insert_index, i)
            try:
               summary_writer.add_scalar(f'hyperparameter/online_insert_index', replay_buffer_online.insert_index, i)
               summary_writer.add_scalar(f'hyperparameter/online_buffer_size', replay_buffer_online.size, i) 
            except:
                pass
            # summary_writer.add_scalar(f'hyperparameter/ratio', agent.ratio, i)
            for k, v in update_info.items():
                summary_writer.add_scalar(f'steps', i, i)
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i, max_bins=512)
            summary_writer.flush()

        if (i<0 and i in offline_eval_epochs) or (i % FLAGS.eval_interval == 0):
            offline = True if i < 1 else False
            eval_episodes = FLAGS.eval_episodes_per_evaluation if offline else FLAGS.eval_episodes
            eval_stats = evaluate(agent, env_eval, eval_episodes, offline)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()
            print('reward:', eval_stats['return'])
            if eval_stats['return'] > best_eval_returns:
                # Store best eval returns
                best_eval_returns = eval_stats['return']
                
            summary_writer.add_scalar(f'evaluation/best_returns', best_eval_returns, i)
            wandb.run.summary["best_returns"] = best_eval_returns
            
            # if 'antmaze' in FLAGS.env_name:
            #     fig = plt.figure()
            #     batch = replay_buffer.sample(100000)
            #     s = batch.observations
            #     a = batch.actions
            #     qs = agent.critic(s, a)
            #     x = s[:, 0]
            #     y = s[:, 1]
            #     plt.scatter(x, y, c=qs[0, :])
            #     plt.colorbar()
            #     # plt.savefig('q_map.png')
            #     # wandb.log({'q_map': wandb.Image("q_map.png"), 'steps': i})
            #     summary_writer.add_figure(f'evaluation/q_map', fig, i)
                    
            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
            
            if offline:
                # logging the offline evaluation results
                expr_now = datetime.now()
                expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
                offline_eval_file.write(f"{expr_time_now_str}\t{epoch}\t{eval_stats['return']}\n")
                offline_eval_file.flush()

                if best_offline_agent_perf < eval_stats['return']:
                    best_offline_agent_epoch = i
                    best_offline_agent_perf = eval_stats['return']
                    save_data = {'agent': agent, 'epoch': i, 'return': eval_stats['return']}
                    save_pickle(save_data, best_offline_agent_filename)



    wandb.finish()
    sys.exit(0)


if __name__ == '__main__':
    app.run(main)
