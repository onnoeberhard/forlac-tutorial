from contextlib import redirect_stdout
from oetils import smooth
from flax import nnx
from tqdm import trange
import numpy as np
from types import SimpleNamespace
import csv
from datetime import datetime
from inspect import getfullargspec
from pathlib import Path
import pickle
import shutil
import sys

from cluster_utils import finalize_job, initialize_job
import jax
import jax.numpy as jnp
from oetils import JaxTqdm
import scipy as sp
from smart_settings.param_classes import recursive_objectify
import sympy as sy
import wandb

from cheetah import MjCheetah
from sac import sac, Actor


class Cheetah:
    u_min = -1
    u_max = 1
    dy = 17
    du = 6
    ep_len = 1000

    def __init__(self, version):
        self.mjc = MjCheetah(version)

    def init(self, rng):
        return self.reset(None, rng)

    def _reset(self, rng):
        return self.mjc.reset(rng)

    def reset(self, x, rng):
        return self._reset(rng),

    def _step(self, x, u):
        return self.mjc.step(x, u)

    def step(self, x, u):
        x, *z = x
        return self._step(x, u), *z

    def monitor(self, t, x):
        return ''


class CheetahK(Cheetah):
    def __init__(self, version, kr, kth, kthd):
        super().__init__(version)
        self.kr = kr
        self.kth = kth
        self.kthd = kthd

    @staticmethod
    def obs(x):
        return jnp.stack([x.data.qpos[3:], x.data.qvel[3:]])

    def init(self, rng):  # type: ignore
        self.K = jnp.repeat(
            jnp.r_[self.kth, self.kthd, self.kr][None], self.du, 0)
        return self.reset((None, None, self.K), rng)

    def reset(self, x, rng):  # type: ignore
        *_, K = x
        x = self._reset(rng)
        xm = self.obs(x)
        return x, xm, K

    def step(self, x, u):  # type: ignore
        x, xm, K = x
        r = 2 * jnp.pi * u
        z = jnp.vstack([self.obs(x), r])
        u = (K.T * z).sum(0)
        u = jax.lax.clamp(-1., u, 1.)
        x = self._step(x, u)
        return x, xm, K

    def monitor(self, t, x):  # type: ignore
        x, xm, K = x
        e = self.obs(x) - xm
        return f', {K.min()}, {K.max()}, {e.min()}, {e.max()}'


class CheetahMRAC(Cheetah):
    def __init__(self, version, kr, kth, kthd, A, B, Gamma):
        super().__init__(version)
        self.kr = kr
        self.kth = kth
        self.kthd = kthd
        self.A = A
        self.B = B

        # Adaptive control setup
        Q = jnp.diag(jnp.r_[1, 1])
        self.P = jnp.zeros((self.du, 2, 2))
        for i in range(self.du):
            self.P = self.P.at[i].set(
                jnp.array(sp.linalg.solve_discrete_lyapunov(A[i].T, Q)))
        self.Gamma = Gamma

    @staticmethod
    def obs(x):
        return jnp.stack([x.data.qpos[3:], x.data.qvel[3:]])

    def init(self, rng):  # type: ignore
        K = jnp.repeat(
            jnp.r_[self.kth, self.kthd, self.kr][None], self.du, 0)
        return self.reset((None, None, K), rng)

    def reset(self, x, rng):  # type: ignore
        *_, K = x
        x = self._reset(rng)
        xm = self.obs(x)
        return x, xm, K

    def step(self, x, u):  # type: ignore
        x, xm, K = x
        r = 2 * jnp.pi * u
        z = jnp.vstack([self.obs(x), r])
        e = self.obs(x) - xm
        for i in range(self.du):
            alpha = self.B[i] @ self.P[i] @ e[:, i]
            # norm = 1 + z[:, i] @ Gamma @ z[:, i]
            K = K.at[i].add(-alpha * self.Gamma @ z[:, i]) # / norm
        lims = jnp.repeat(jnp.r_[1, 0.3, 1][None], self.du, 0)
        K = jax.lax.clamp(-lims, K, lims)
        u = (K.T * z).sum(0)
        u = jax.lax.clamp(-1., u, 1.)
        x_ = self._step(x, u)
        for i in range(self.du):
            xm = xm.at[:, i].set(
                self.A[i] @ self.obs(x)[:, i] + self.B[i] * r[i])
        return x_, xm, K

    def monitor(self, t, x):  # type: ignore
        x, xm, K = x
        e = self.obs(x) - xm
        return f', {K.min()}, {K.max()}, {e.min()}, {e.max()}'


def response(origin, seed, env, T=200_000):
    @jax.jit
    def rollouts(x, t):
        (x, *aux), rng = x
        rng, *keys = jax.random.split(rng, 3)
        x = jax.lax.cond(t % 1000 == 0,
            lambda _, rng: env._reset(rng), lambda x, _: x, x, keys[0])
        u = actor.act(x.obs, keys[1])
        r = 2 * jnp.pi * u
        z = jnp.vstack([env.obs(x), r])
        K = jnp.repeat(jnp.r_[-1, 0, 1][:, None], env.du, 1)
        u = (K * z).sum(0)
        u = jax.lax.clamp(-1., u, 1.)
        x_ = env._step(x, u)
        return ((x_, *aux), rng), (env.obs(x), r, env.obs(x_))

    rngs = nnx.Rngs(seed)
    with open(Path(origin) / 'actor.pkl', 'rb') as f:
        actor_params = pickle.load(f)
    actor = Actor(env, rngs)
    nnx.update(actor, actor_params)
    x = env.init(rngs())
    _, data = jax.lax.scan(rollouts, (x, rngs()), jnp.arange(T))

    A = np.zeros((env.du, 2, 2))
    B = np.zeros((env.du, 2))
    X = np.concatenate([data[0], data[1][:, None, :]], 1)
    y = data[2] - data[0]

    for i in range(env.du):
        w, *_ = np.linalg.lstsq(X[..., i], y[..., i])
        A[i] = np.eye(2) + w[:2].T
        B[i] = w[2]

    return A, B


def change(origin, seed, env0, env1):
    def rollouts(env, actor, x, t):
        x, rng = x
        rng, *keys = jax.random.split(rng, 3)
        x = jax.lax.cond(t % 1000 == 0,
            lambda _, rng: env.reset(x, rng), lambda x, _: x, x, keys[0])
        u = actor.act(x[0].obs, keys[1])
        x = env.step(x, u)
        return (x, rng), x[0].reward

    # Load policy
    rngs = nnx.Rngs(seed)
    with open(origin / 'actor.pkl', 'rb') as f:
        actor_params = pickle.load(f)
    actor = Actor(env0, rngs)
    nnx.update(actor, actor_params)

    # Rollout in env0
    x = env0.init(rngs())
    rs = jax.jit(lambda x, t: rollouts(env0, actor, x, t))
    _, rewards = jax.lax.scan(rs, (x, rngs()), jnp.arange(200_000))
    _, rewards0 = smooth(rewards, 200)

    # Rollout in env1
    x = env1.init(rngs())
    rs = jax.jit(lambda x, t: rollouts(env1, actor, x, t))
    _, rewards = jax.lax.scan(rs, (x, rngs()), jnp.arange(800_000))
    _, rewards1 = smooth(rewards, 800)

    return np.concatenate([rewards0, rewards1]) * 1000


def run_sac(seed, env, wab, params, path):
    if wab.log:
        wandb.login(key=wab.key)
        wandb.init(project=wab.project, group=params.name,
            config=params, dir=str(path))

    match env:
        case 'cheetah': env = Cheetah(0)
        case 'cheetah-mrac':
            env = CheetahMRAC(
                0, params.adapt, params.kr, params.kth, params.kthd)
        case _: raise ValueError('invalid environment')

    rng = jax.random.key(seed)
    metrics, actor = sac(env, rng, wablog=wab.log)

    with open(path / 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    with open(path / 'actor.pkl', 'wb') as f:
        pickle.dump(actor, f)

    if wab.log:
        wandb.finish()


def run_change():
    # Raw SAC
    env0 = Cheetah(0)
    env1 = Cheetah(1)
    for i in trange(20):
        origin = Path(f'dat/jobs/sac/working_directories/{i}')
        data = change(origin, i, env0, env1)
        np.save(origin / 'change.npy', data)

    # Fixed-K SAC
    kr, kth, kthd = 1., -1., 0.
    env0 = CheetahK(0, kr, kth, kthd)
    env1 = CheetahK(1, kr, kth, kthd)
    for i in trange(20):
        origin = Path(f'dat/jobs/sac-K/working_directories/{i}')
        data = change(origin, i, env0, env1)
        np.save(origin / 'change.npy', data)

    # Fixed-K SAC
    env0 = CheetahK(0, kr, kth, -0.3)
    env1 = CheetahK(1, kr, kth, -0.3)
    for i in trange(20):
        origin = Path(f'dat/jobs/sac-K/working_directories/{i}')
        data = change(origin, i, env0, env1)
        np.save(origin / 'change-alt.npy', data)

    # MRAC-SAC
    env = CheetahK(0, 1, -1, 0)
    Gamma = jnp.diag(jnp.r_[0, 1e-7, 0])
    for i in trange(20):
        origin = Path(f'dat/jobs/sac-K/working_directories/{i}')
        A, B = response(origin, i, env)
        env0 = CheetahMRAC(0, kr, kth, kthd, A, B, Gamma)
        env1 = CheetahMRAC(1, kr, kth, kthd, A, B, Gamma)
        data = change(origin, i, env0, env1)
        np.save(origin / 'change-mrac-4.npy', data)


def plot():
    import matplotlib.pyplot as plt
    import oetils
    W = oetils.init_plotting('icml2024', bundle_kwargs={'column': 'full'})
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(W, W/3))
    ax[0].set_title('Training of SAC policy')
    ax[1].set_title('Deployment of SAC policy')
    ax[0].set_xlabel('Time $t$')
    ax[1].set_xlabel('Time $t$')
    ax[0].set_ylabel('Total reward per episode')

    sac = np.zeros((20, 1000))
    for i in range(20):
        with open(f'dat/jobs/sac/working_directories/{i}/metrics.pkl',
                'rb') as f:
            metrics = pickle.load(f)
            sac[i] = metrics['tot_rew']
    sac = sac[[i for i in range(20) if i != 2]]

    sac_k = np.zeros((20, 1000))
    for i in range(20):
        with open(f'dat/jobs/sac-K/working_directories/{i}/metrics.pkl',
                'rb') as f:
            metrics = pickle.load(f)
            sac_k[i] = metrics['tot_rew']

    rng = np.random.default_rng(42)
    for run, label in zip([sac, sac_k],
            ['Pure SAC', 'SAC with fixed low-level controller $K_0$']):
        ts, rs = smooth(run, 200)
        medians, _, lows, highs = oetils.bootstrap_cis(rs.T, rng)
        ax[0].fill_between(ts*1000, lows, highs, alpha=0.3)
        ax[0].plot(ts*1000, medians, label=label)

    ax[0].legend()

    change_sac = np.zeros((20, 1000))
    for i in range(20):
        rewards = np.load(f'dat/jobs/sac/working_directories/{i}/change.npy')
        change_sac[i] = rewards
    change_sac = change_sac[[i for i in range(20) if i != 2]]

    change_sac_k = np.zeros((20, 1000))
    for i in range(20):
        rewards = np.load(f'dat/jobs/sac-K/working_directories/{i}/change.npy')
        change_sac_k[i] = rewards

    change_sac_k_alt = np.zeros((20, 1000))
    for i in range(20):
        rewards = np.load(
            f'dat/jobs/sac-K/working_directories/{i}/change-alt.npy')
        change_sac_k_alt[i] = rewards

    change_sac_mrac = np.zeros((20, 1000))
    for i in range(20):
        rewards = np.load(
            f'dat/jobs/sac-K/working_directories/{i}/change-mrac-4.npy')
        change_sac_mrac[i] = rewards

    for run, label in zip([change_sac, change_sac_k, change_sac_mrac],
            ['', '', 'SAC with low-level MRAC controller']):
        ts, rs = smooth(run, 200)
        medians, _, lows, highs = oetils.bootstrap_cis(rs.T, rng)
        ax[1].fill_between((ts + 1000) * 1000, lows, highs, alpha=0.3)
        ax[1].plot((ts + 1000) * 1000, medians, label=label)

    ax[1].axvline(1_200_000, c='r', ls='--', label='Dynamics change')
    ax[1].legend()

    fig.savefig('etc/sac.pdf')


def run(params, path):
    pass


def main():
    now = datetime.now()

    # Read hyperparameters
    params = initialize_job(sys.argv + ['--parameter-dict', '{}']
        if len(sys.argv) == 1 else [], verbose=False)
    params = recursive_objectify(params, make_immutable=False)
    params.update(dict(params.get('conf') or {}))

    # Configure working directory and job name
    named = 'name' in params
    cluster = 'working_dir' in params
    run_ = str(params.get('run', '')) or now.strftime("%Y%m%d%H%M%S")
    name = (n + '-' if (n := params.get('name')) else '') + run_
    params.name = name
    path = Path(params.working_dir) if cluster else \
        Path.cwd() / 'dat/runs' / name if named else Path.cwd() / 'tmp'
    if named and not cluster:
        if path.exists() and input(f"Path {path} already exists. "
                "Delete everything? (y/N) ").lower() == 'y':
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)
    if named:  # Remove partial checkpoints
        for cp in path.glob('checkpoint.*tmp*'):
            shutil.rmtree(cp)
    if cluster and (metrics := path / 'metrics.csv').exists():
        with open(metrics) as f:
            metrics = next(csv.DictReader(f))
        finalize_job(metrics, params)  # type: ignore
        return

    # Run experiment
    metrics = None
    print(f"Using path {path}.")
    interactive = params.interactive if 'interactive' in params else not named
    with (open(path / 'log.txt', 'a' if named else 'w') if not interactive 
            else sys.stdout) as f, redirect_stdout(f):
        print(now.strftime("%Y-%m-%d %H:%M:%S") + '\n' + str(params),
            flush=True)
        function = globals()[fun] if (fun := params.get('function')) else run
        vars_ = getfullargspec(function)[0]
        metrics = function(**(({'path': path} if 'path' in vars_ else {})
            | ({'params': params} if 'params' in vars_ else {})
            | {var: params[var] for var in vars_ if var in params}))

    if cluster:
        finalize_job(metrics or {}, params)


if __name__ == '__main__':
    import subprocess
    import absl.logging
    absl.logging.set_verbosity(absl.logging.DEBUG)
    jax.config.update('jax_log_compiles', True)
    # jax.config.update("jax_debug_nans", True)
    try:
        smi = subprocess.run(["nvidia-smi"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f'nvidia-smi: return code {smi.returncode}\n'
            + smi.stdout + '\n', flush=True)
    except FileNotFoundError:
        print("nvidia-smi command not found!", flush=True)
    plot()
    # main()
