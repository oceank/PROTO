"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import flax
import jax.numpy as jnp
import numpy as np
import optax
import os

# NAN debug
# from jax.config import config
# config.update("jax_debug_nans", True)

import policy
import value_net
from actor import update as awr_update_actor
from actor import update_online as sac_update_actor
from actor import update_alpha
from actor import update_mu
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v, update_q_online
from ensemble import Ensemble, subsample_ensemble

from functools import partial


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=['double', 'vanilla', 'args', 'offline'])
def _update_jit(
    rng: PRNGKey, offline_actor: Model, critic: Model, value: Model,
    target_critic: Model, behavior: Model, batch: Batch, discount: float, tau: float,
    expectile: float, loss_temp: float, double: bool, vanilla: bool, offline: bool, args,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    # for _ in range(args.num_v_updates):
    new_value, value_info = update_v(target_critic, value, batch, expectile, loss_temp, double, vanilla, key, args)
    value = new_value

    new_offline_actor, offline_actor_info = awr_update_actor(key, offline_actor, target_critic, new_value, batch, loss_temp, double, args)

    new_critic, critic_info = update_q(critic, new_value, offline_actor, behavior, batch, discount, double, key, loss_temp, offline, args)
    
    new_behavior, behavior_info = update_mu(key, behavior, batch)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_offline_actor, new_critic, new_value, new_target_critic, new_behavior, {
        **critic_info,
        **value_info,
        **offline_actor_info,
        **behavior_info
    }


@partial(jax.jit, static_argnames=['double', 'args', 'discount', 'tau', 'tau_actor', 'ratio', 'target_entropy'])
def _update_jit_online(
    rng: PRNGKey, offline_actor: Model, online_actor: Model, critic: Model, value: Model,
    target_critic: Model, target_online_actor: Model, behavior: Model, batch: Batch, discount: float, tau: float, tau_actor: float, 
    temp: float, temp_online: float, ratio: float, double: bool, log_alpha: float, 
    target_entropy: float, args,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key_critic, key_actor, key_alpha, rng = jax.random.split(rng, num=4)
    # assume that utd is 1 for our experiment, so we don't need to loop
    '''
    for i in range(args.utd): 

    def slice(x):
        assert x.shape[0] % args.utd == 0
        batch_size = x.shape[0] // args.utd
        return x[batch_size * i : batch_size * (i + 1)]
    '''

    mini_batch = batch #jax.tree_util.tree_map(slice, batch)

    new_critic, key_critic, critic_info = update_q_online(critic, target_critic, target_online_actor, online_actor, behavior, offline_actor, mini_batch, discount, double, key_critic, temp, temp_online, log_alpha, args)

    new_target_critic = target_update(new_critic, target_critic, tau)

    critic = new_critic
    target_critic = new_target_critic
    
    new_online_actor, online_actor_info = sac_update_actor(key_actor, online_actor, offline_actor, new_critic, target_online_actor, behavior, mini_batch, temp, temp_online, ratio, double, log_alpha, args)

    # if args.sac and args.auto_alpha:
    new_log_alpha, alpha_info = update_alpha(key_alpha, online_actor_info['entropy'], log_alpha, target_entropy)
    # else:
    #     new_log_alpha = log_alpha
    #     alpha_info = {'target_entropy': target_entropy}

    new_target_actor = target_update(new_online_actor, target_online_actor, tau_actor)

    
    return rng, new_online_actor, new_critic, new_target_critic, new_target_actor, new_log_alpha, {
        **critic_info,
        **online_actor_info,
        **alpha_info
    }



class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 tau_actor: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 layernorm: bool = False,
                 value_dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 loss_temp: float = 1.0,
                 double_q: bool = True,
                 double_q_online: bool = True,
                 vanilla: bool = True,
                 auto_alpha: bool = True,
                 opt_decay_schedule: str = "cosine",
                 args=None):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        self.actions = actions
        self.expectile = expectile
        self.tau = tau
        self.tau_actor = tau_actor
        self.discount = discount
        self.temperature = temperature
        self.loss_temp = loss_temp
        self.loss_temp_online = loss_temp
        self.double_q = double_q
        self.double_q_online = double_q_online
        self.vanilla = vanilla
        self.args = args
        self.target_entropy = actions.shape[1] * -1. / 2
        self.ratio = 1.0
        self.alpha_lr = actor_lr

        rng = jax.random.PRNGKey(seed)
        rng, offline_actor_key, behavior_key, online_actor_key, critic_key, value_key = jax.random.split(rng, 6)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        offline_actor = Model.create(actor_def,
                                     inputs=[offline_actor_key, observations],
                                     tx=optimiser)

        behavior = Model.create(actor_def,
                                inputs=[behavior_key, observations],
                                tx=optimiser)

        online_actor = Model.create(actor_def,
                                    inputs=[online_actor_key, observations],
                                    tx=optimiser)

        target_online_actor = Model.create(actor_def,
                                           inputs=[online_actor_key, observations],
                                           tx=optimiser) 

        critic_cls = partial(value_net.Critic, hidden_dims=hidden_dims, layer_norm=layernorm)
        critic_def = Ensemble(critic_cls, num=2)

        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims,
                                     layer_norm=layernorm,
                                     dropout_rate=value_dropout_rate)
        
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])
        
        init_temperature = 0.01
        log_alpha = Model.create(value_net.Temperature(init_temperature),
                            inputs=[value_key],
                            tx=optax.adam(learning_rate=actor_lr))

        self.offline_actor = offline_actor
        self.online_actor = online_actor
        self.target_online_actor = target_online_actor
        self.behavior = behavior
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.log_alpha = log_alpha
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       offline: bool = True,
                       temperature: float = 1.0) -> jnp.ndarray:
        actor = self.offline_actor if offline else self.online_actor
        rng, actions = policy.sample_actions(self.rng, actor.apply_fn,
                                             actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, offline: bool) -> InfoDict:
        if offline:
            new_rng, new_offline_actor, new_critic, new_value, new_target_critic, new_behavior, info = _update_jit(
            self.rng, self.offline_actor, self.critic, self.value, self.target_critic, self.behavior,
            batch, self.discount, self.tau, self.expectile, self.loss_temp, self.double_q, self.vanilla, offline, self.args)
            
            self.offline_actor = new_offline_actor
            self.behavior = new_behavior
            self.value = new_value
            
            self.rng = new_rng
            self.critic = new_critic
            self.target_critic = new_target_critic
        else:
            new_rng, new_online_actor, new_critic, new_target_critic, new_target_actor, new_log_alpha, info = _update_jit_online(
            self.rng, self.offline_actor, self.online_actor, self.critic, self.value, self.target_critic, self.target_online_actor, self.behavior, 
            batch, self.discount, self.tau, self.tau_actor, self.loss_temp, self.loss_temp_online, self.ratio, self.double_q_online,
            self.log_alpha, self.target_entropy, self.args) 
            
            self.online_actor = new_online_actor
            self.target_online_actor = new_target_actor
            self.log_alpha = new_log_alpha
            
            self.rng = new_rng
            self.critic = new_critic
            self.target_critic = new_target_critic

        return info

    
    
    # offline2online transfer
    def offline2online(self):
        # offline actor -> online actor
        new_online_actor = target_update(self.offline_actor, self.online_actor, tau=1.)
        self.online_actor = new_online_actor
        
        # online actor -> target online actor
        new_target_online_actor = target_update(self.online_actor, self.target_online_actor, tau=1.)
        self.target_online_actor = new_target_online_actor
       
    
    def load(self, save_dir: str, offline: bool):
        self.offline_actor = self.offline_actor.load(os.path.join(save_dir, 'offline_actor'))
        self.behavior = self.behavior.load(os.path.join(save_dir, 'behavior'))
        self.online_actor = self.online_actor.load(os.path.join(save_dir, 'online_actor'))
        self.target_online_actor = self.target_online_actor.load(os.path.join(save_dir, 'target_online_actor'))
        
        self.actor = self.actor.load(os.path.join(save_dir, 'actor'))
        self.critic = self.critic.load(os.path.join(save_dir, 'critic'))
        self.value = self.value.load(os.path.join(save_dir, 'value'))
        self.target_critic = self.target_critic.load(os.path.join(save_dir, 'critic'))

    def save(self, save_dir: str, offline: bool):
        self.actor.save(os.path.join(save_dir, 'actor'))
        self.critic.save(os.path.join(save_dir, 'critic'))
        self.value.save(os.path.join(save_dir, 'value'))
