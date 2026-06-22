# Foundations of Reinforcement Learning and Control: Connections and New Perspectives

This is the official repository to the paper "Foundations of Reinforcement Learning and Control: Connections and New Perspectives" by Claire Vernade, Onno Eberhard, Martha White, Florian Dörfler, Csaba Szepesvári, Miroslav Krstic, and Michael Muehlebach (to be published in Informs Tutorials in Operations Research 2026).

The contents are as follows.
- `cheetah-0.xml` and `cheetah-1.xml` contain the MuJoCo specifications of the original Half-Cheetah and the modified low-friction Half-Cheetah, respectively. The file `cheetah-0.xml` is taken from the DeepMind Control Suite library.
- `cheetah.py` contains the Python API to the Half-Cheetah systems, adapted from the MuJoCo-Playgruound library.
- `main.py` implements the experiments that we perform on the Half-Cheetah system (SAC, SAC-K_0, and SAC-MRAC).
- `mrac.py` implements the MRAC control algorithm for the simple second-order dynamical system.
- `pyproject.toml` contains dependencies.
- `sac.py` contains our implementation of the SAC algorithm.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{vernade-2026-foundations,
  title = {Foundations of Reinforcement Learning and Control: Connections and New Perspectives},
  author = {Vernade, Claire and Eberhard, Onno and White, Martha and Dörfler, Florian and Szepesvári, Csaba and Krstic, Miroslav and Muehlebach, Michael},
  journal = {INFORMS Tutorials in Operations Research},
  year = {2026}
}
```

If there are any problems, or if you have a question, don't hesitate to open an issue here on GitHub.
