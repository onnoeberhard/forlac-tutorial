from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import oetils
import scipy as sp
import sympy as sy

@jax.jit
def f(x, rABCKK):
    '''Compute next state and observation'''
    r, A, B, C, Kx, Kr = rABCKK
    u = Kx @ x + Kr * r
    x = A @ x + B * u
    return x, C @ x

def ff(A, B, C, Kx):
    '''Compute feedforward gain'''
    return 1 / (C @ jnp.linalg.solve(jnp.eye(2) - A + jnp.outer(B, -Kx), B))

def fa_(xxKK, rABC, Am, Bm, Gamma, P):
    '''Step system and adaptive controller'''
    x, xm, Kx, Kr = xxKK
    r, A, B, C = rABC
    theta = jnp.concat([Kx, Kr[None]])
    omega = jnp.concat([x, r[None]])
    e = x - xm
    alpha = B @ P @ e
    # norm = 1 + omega @ Gamma @ omega
    theta -= alpha * Gamma @ omega  # / norm
    Kx = theta[:2]
    Kr = theta[2]
    x, y = f(x, (r, A, B, C, Kx, Kr))
    xm = Am @ xm + Bm * r
    return (x, xm, Kx, Kr), y

# Setup
jax.config.update("jax_enable_x64", True)
W = oetils.init_plotting('icml2024', bundle_kwargs={'column': 'full'})

# System and model parameters
zeta, omega, T = sy.symbols('zeta, omega, T')
params = {'T': 0.01}  # Sampling period
params1 = params | {'omega': 1, 'zeta': 10}
params2 = params | {'omega': 1, 'zeta': 0.1}
params_m = params | {'omega': 10, 'zeta': 1}

# General continuous second order system
Ac = sy.Matrix([[0, 1], [-omega**2, -2*zeta*omega]])
Bc = sy.Matrix([0, 1])
Cc = sy.Matrix([1, 0])

# Real (ZOH discretized) systems (from params1 and params2)
A1 = jnp.array(sy.exp((Ac * T).subs(params1)), float)
B1 = jnp.array(Ac.subs(params1).inv() @ (A1 - sy.eye(2)) @ Bc, float).flatten()
A2 = jnp.array(sy.exp((Ac * T).subs(params2)), float)
B2 = jnp.array(Ac.subs(params2).inv() @ (A2 - sy.eye(2)) @ Bc, float).flatten()
C = jnp.array(Cc, float).flatten()

# Reference model (params_m)
Am = jnp.array(sy.exp((Ac * T).subs(params_m)), float)
Bm = jnp.array(Ac.subs(params_m).inv() @ (Am - sy.eye(2)) @ Bc, float).flatten()
Bm = Bm / (C @ jnp.linalg.solve(jnp.eye(2) - Am, Bm))

# Adaptive control setup
Gamma = jnp.diag(jnp.array([100, 100, 100]))
Q = jnp.diag(np.r_[1, 1])
P = jnp.array(sp.linalg.solve_discrete_lyapunov(Am.T, Q))

# Problem setup
t0 = 10_000  # Total time length
p0 = 2000   # Period of square wave
x0 = jnp.zeros(2)
rs = jnp.tile(jnp.array([1]*(p0 // 2) + [0]*(p0 // 2)), t0 // p0)
As = jnp.repeat(jnp.stack([A1, A2]), t0 // 2, 0)
Bs = jnp.repeat(jnp.stack([B1, B2]), t0 // 2, 0)
Cs = jnp.repeat(C[None], t0, 0)
rABC = rs, As, Bs, Cs

# Open-loop behavior
Kx = jnp.zeros(2)
Kr = jnp.ones(1)
Kxs = jnp.repeat(Kx[None], t0, 0)
Krs = jnp.repeat(Kr[None], t0, 0)
# Krs = jnp.repeat(jnp.stack([ff(A1, B1, C, Kx), ff(A2, B2, C, Kx)]), 5000, 0)
_, ys_ol = jax.lax.scan(f, x0, (*rABC, Kxs, Krs))

# Open-loop behavior of reference model
Kx = jnp.zeros(2)
Kr = jnp.ones(1)
Kxs = jnp.repeat(Kx[None], t0, 0)
Krs = jnp.repeat(Kr[None], t0, 0)
Ams = jnp.repeat(Am[None], t0, 0)
Bms = jnp.repeat(Bm[None], t0, 0)
_, ys_m = jax.lax.scan(f, x0, (rs, Ams, Bms, Cs, Kxs, Krs))

# Adaptive controller
Kx = jnp.zeros(2)
Kr = 1
fa = jax.jit(partial(fa_, Am=Am, Bm=Bm, Gamma=Gamma, P=P))
xxKK, ys_a = jax.lax.scan(fa, (x0, x0, Kx, Kr), rABC)

fig, ax = plt.subplots(1, 3, figsize=(W, W/3), sharex=True, sharey=True)
# ax[0].set_title(r'Position of driven harmonic oscillator ($\omega = 1, \zeta_1 = 0.1, \zeta_2 = 10$)')
T = t0 * params['T']
t = np.linspace(0, T, t0)
for i in range(2):
    ax[i].axvline(T / 2, c='r', ls='--', label='Dynamics change')
for i in range(3):
    ax[i].set_xlabel('Time $t$')
    ax[i].plot(t, rs, 'k', label='Reference $r(t)$')
ax[0].plot(t, ys_ol, label='Output $y(t)$')
ax[1].plot(t, ys_a, label=r'Adaptive controller')
ax[2].plot(t, ys_m, label=r'Open-loop behavior of reference model ($\omega = 10, \zeta = 1$)')
ax[0].legend()
ax[0].set_title('Open-loop behavior')
ax[1].set_title('Model-Reference Adaptive Control')
ax[2].set_title('Reference model (open-loop)')
# for i in range(3):
#     ax[i].legend(loc='upper right')
plt.savefig('etc/osc-adaptive-2.pdf')

