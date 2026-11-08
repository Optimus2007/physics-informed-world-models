import sys, time, torch
from src.control.envs import Pendulum
from src.control.eval.data import collect_random, fit
from src.control.eval.rollouts import *
from src.control.models.mlp_ensemble import MLPEnsemble
from src.control.models.neural_ode import NeuralODE
from src.control.models.physics_structured import PhysicsStructured
from src.control.models.neural_hamiltonian import NeuralHamiltonian
env = Pendulum(); data = collect_random(env, 100, 50, seed=0)
ev = eval_data(env)
for M in [PhysicsStructured, NeuralODE, NeuralHamiltonian, MLPEnsemble]:
    torch.manual_seed(0); m = M(env) if M is not PhysicsStructured else M(env, init_params=dict(m=1.4, l=0.8, g=7.0))
    t = time.time(); r = fit(m, data, steps=int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
    print(m.name, round(time.time() - t), 's val', r['best_val'], {h: round(v['median'], 4) for h, v in horizon_error(m, env, *ev).items()},
          'E', {k: round(v, 4) for k, v in energy_drift(m, env).items()}, getattr(m, 'param_dict', lambda: '')(), flush=True)
