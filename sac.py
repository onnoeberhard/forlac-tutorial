from collections import namedtuple
from copy import deepcopy
import sys

import distrax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from oetils import JaxTqdm
import optax
import wandb


class Actor(nnx.Module):
    def __init__(self, env, rngs):
        self.u_min = env.u_min
        self.u_max = env.u_max
        self.net = nnx.Sequential(
            nnx.Linear(env.dy, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 256, rngs=rngs),
            nnx.relu
        )
        self.mean = nnx.Linear(256, env.du, rngs=rngs)
        self.lstd = nnx.Linear(256, env.du, rngs=rngs)

    def __call__(self, y):
        z = self.net(y)
        mu = self.mean(z)
        ls = -10 + 12 * nnx.sigmoid(self.lstd(z))  # Soft clamp in [-10, 2]
        dist = distrax.MultivariateNormalDiag(mu, jnp.exp(ls))
        bijector = distrax.Block(distrax.Chain([
            distrax.ScalarAffine(self.u_min, self.u_max - self.u_min),
            distrax.Sigmoid()]), 1)
        return distrax.Transformed(dist, bijector)

    def act(self, y, rng):
        return self(y).sample(seed=rng)


class Critic(nnx.Module):
    def __init__(self, env, rngs):
        arch = lambda: nnx.Sequential(
            nnx.Linear(env.dy + env.du, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 1, rngs=rngs)
        )
        self.q1 = arch()
        self.q2 = arch()

    def __call__(self, y, u):
        yu = jnp.concat([y, u], axis=-1)
        return self.q1(yu), self.q2(yu)


class EntropyCoef(nnx.Module):
    def __init__(self):
        self.log_alpha = nnx.Param(0.)

    def __call__(self):
        return jnp.exp(self.log_alpha.get_value())


def sac(env, rng, n_steps=1_000_000, init_steps=100, lr_actor=0.0003,
        lr_critic=0.001, n_buf=100_000, n_batch=256, gamma=0.99, tau=0.005,
        n_monitor=1000, wablog=False):
    '''Soft actor-critic with automatic entropy tuning'''
    pbar = JaxTqdm(n_steps, n_monitor)
    env.pbar = pbar

    def loss_critic(critic, bat, actor, target, alpha, rng):
        u, lp = actor(bat.y_).sample_and_log_prob(seed=rng)
        qt = jnp.minimum(*target(bat.y_, u)).squeeze()
        qt = bat.r + gamma * (1 - bat.d) * qt - alpha()*lp
        q = jnp.concat(critic(bat.y, bat.u), axis=-1)
        return ((q - qt[:, None])**2).mean()

    def loss_actor(actor, bat, critic, alpha, rng):
        u, lp = actor(bat.y).sample_and_log_prob(seed=rng)
        q = jnp.minimum(*critic(bat.y, u)).squeeze()
        return (alpha() * lp - q).mean()

    def loss_alpha(alpha, bat, actor, rng):
        _, lp = actor(bat.y).sample_and_log_prob(seed=rng)
        return alpha() * (env.du - lp).mean()

    def update(t, buf, actor, critic, target, alpha, opt_actor, opt_critic,
            opt_alpha, rngs):
        # Sample batch from replay buffer
        lev = jnp.minimum(t, n_buf)
        idx = jax.random.randint(rngs(), (n_batch,), 0, lev)
        bat = jax.tree.map(lambda x: x[idx], buf)

        # Update critic
        critic_loss, grads = nnx.value_and_grad(loss_critic)(
            critic, bat, actor, target, alpha, rngs())
        opt_critic.update(critic, grads)

        # Update actor
        actor_loss, grads = nnx.value_and_grad(loss_actor)(
            actor, bat, critic, alpha, rngs())
        opt_actor.update(actor, grads)

        # Update entropy coefficient
        alpha_loss, grads = nnx.value_and_grad(loss_alpha)(
            alpha, bat, actor, rngs())
        opt_alpha.update(alpha, grads)

        # Update target networks
        state_critic = nnx.state(critic, nnx.Param)
        state_target = nnx.state(target, nnx.Param)
        nnx.update(target, jax.tree.map(
            lambda x, y: tau * x + (1 - tau) * y, state_critic, state_target))

        return {'critic_loss': critic_loss, 'actor_loss': actor_loss,
            'alpha_loss': alpha_loss}

    def ep_end(i, metrics, x, ep_t, ep_r, rng):
        metrics['episodes'] = metrics['episodes'].at[i].add(1)
        metrics['tot_rew'] = metrics['tot_rew'].at[i].add(ep_r)
        metrics['ep_len'] = metrics['ep_len'].at[i].add(ep_t)
        x = env.reset(x, rng)
        return metrics, x, 0, 0.

    def monitor_cb(t, x, metrics):
        i = t // (n_steps // n_monitor)
        pbar.write(f'Steps: {(t + 1):{int(np.log10(n_steps)) + 1}d}. '
            f'Mean total reward: {metrics['tot_rew'][i]:8.6f}, '
            f'mean episode length: {metrics['ep_len'][i]:8.3f}, '
            f'critic loss: {metrics['critic_loss'][i]:8.3f}, '
            f'actor loss: {metrics['actor_loss'][i]:8.3f}.'
            + env.monitor(t, x))
        sys.stdout.flush()
        if wablog:
            wandb.log({f'losses/{k}': metrics[k][i] for k in ['critic_loss',
                'actor_loss', 'alpha_loss']} | {k: metrics[k][i] for k in 
                ['tot_rew', 'ep_len', 'alpha']}, (t + 1))

    def monitor(t, x, metrics):
        n = t % (n_steps // n_monitor)
        i = t // (n_steps // n_monitor)
        n_updates = jnp.minimum(n, t - init_steps) + 1
        for k in ['critic_loss', 'actor_loss', 'alpha_loss']:
            metrics[k] = metrics[k].at[i].divide(n_updates)
        metrics['alpha'] = metrics['alpha'].at[i].divide(n)
        for k in ['tot_rew', 'ep_len']:
            metrics[k] = metrics[k].at[i].set(jax.lax.cond(
                metrics['episodes'][i] > 0, lambda m, e: m / e, 
                lambda *_: jnp.nan, metrics[k][i], metrics['episodes'][i]))
        jax.debug.callback(monitor_cb, t, x, metrics)
        return metrics

    @pbar.loop
    @nnx.jit
    def step(t, x):
        metrics, x, ep_t, ep_r, actor, critic, target, alpha, \
            *opt, buf, rngs = x
        y = x[0].obs

        # Sample action and step environment
        u = nnx.cond(t > init_steps, lambda actor, y, key: actor.act(y, key),
            lambda _, __, key: jax.random.uniform(key, env.du,
            minval=env.u_min, maxval=env.u_max), actor, y, rngs())
        x = env.step(x, u)
        ep_t += 1
        ep_r += x[0].reward

        # Save trasition to replay buffer
        buf = jax.tree.map(lambda x, y: x.at[t % n_buf].set(y), buf,
            Transition(y, u, x[0].obs, x[0].reward, x[0].done))

        # Log episode stats, reset episode if goal reached or limit exceeded
        i = t // (n_steps // n_monitor)
        metrics, x, ep_t, ep_r = jax.lax.cond(
            x[0].done.astype(bool) | ep_t == env.ep_len, ep_end,
            lambda _, m, x, ep_t, ep_r, __: (m, x, ep_t, ep_r),
            i, metrics, x, ep_t, ep_r, rngs())

        # Update parameters
        metrics_ = nnx.cond(t >= init_steps, update, lambda *_: {k: jnp.nan
            for k in ['critic_loss', 'actor_loss', 'alpha_loss']}, t, buf,
            actor, critic, target, alpha, *opt, rngs) | {'alpha': alpha()}

        # Monitoring
        for k, v in metrics_.items():
            metrics[k] = metrics[k].at[i].add(v)
        metrics = jax.lax.cond((t + 1) % (n_steps // n_monitor) == 0,
            monitor, lambda _, __, metrics: metrics, t, x, metrics)

        return metrics, x, ep_t, ep_r, actor, critic, target, \
            alpha, *opt, buf, rngs

    # Initialize actor and critic
    rngs = nnx.Rngs(rng)
    actor = Actor(env, rngs)
    critic = Critic(env, rngs)
    target = deepcopy(critic)

    # Initialize optimizers
    opt_actor = nnx.Optimizer(actor, optax.adam(lr_actor), wrt=nnx.Param)
    opt_critic = nnx.Optimizer(critic, optax.adam(lr_critic), wrt=nnx.Param)

    # Set up automatic entropy tuning
    alpha = EntropyCoef()
    opt_alpha = nnx.Optimizer(alpha, optax.adam(lr_critic), wrt=nnx.Param)

    # Initialize replay buffer
    Transition = namedtuple('Transition', ['y', 'u', 'y_', 'r', 'd'])
    buf = Transition(jnp.zeros((n_buf, env.dy)), jnp.zeros((n_buf, env.du)),
        jnp.zeros((n_buf, env.dy)), jnp.zeros(n_buf), jnp.zeros(n_buf))

    # Initialize environment and start training
    x = env.init(rngs())
    metrics = {k: jnp.zeros(n_monitor) for k in ['episodes', 'tot_rew',
        'ep_len', 'critic_loss', 'actor_loss', 'alpha_loss', 'alpha']}
    metrics, *_ = nnx.fori_loop(0, n_steps, step, (metrics, x, 0, 0., actor,
        critic, target, alpha, opt_actor, opt_critic, opt_alpha, buf, rngs))
    params = nnx.state(actor, nnx.Param)
    return metrics, params

