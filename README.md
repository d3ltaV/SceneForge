# SceneForge

A novel automatic training curriculum for training reinforcement learning (RL) driving agents: implementing an extended replay buffer with scene specifications.

Standard RL training can be very inefficient: sampled training scenes are often too easy or difficult for the agent, requiring the agent to learn an optimal policy through pure trial and error. Ultimately, difficult edge cases are underrepresented, and the agent is not able to master them during training.

Previously, a curricula has been designed to address this, where past scenes with regret, or positive value loss (PVL), a heuristic for learning potential, are stored in a buffer to be 'replayed' during training. PVL is the mean positive component of the Generalized Advantage Estimation (GAE) over an episode. It is high when the critic (in Actor-Critic method) underestimates the return reward of the agent.

However, this could lead to a replay of scenes with similar parameters, where the agent is not able to encounter a variety of challenging edge cases. Our approach enforces parameter diversity during replay by partitioning the buffer into k sub-buffers, each holding scenes with different parameter criteria. Scenes in buffers are mutated and combined through crossover during replay, producing fresh variants of difficult cases. The result is training on a diverse set of previously encountered scenes. Through testing, we show that we reduced collision rates by 60% compared to baselines, highlighting the importance of training environment diversity during reinforcement learning.

SIP 2026 project.

## What it does

Scenes are written in [Scenic](https://scenic-lang.org/), executed in [MetaDrive](https://github.com/metadriverse/metadrive), and the policy is CleanRL PPO. The curriculum sits between the scene generator and the env.

**n × k buffer** ([`policy/custom/buffer.py`](policy/custom/buffer.py)): `k` sub-buffers bucketed by scene complexity (number of distractor cars, in increments `[1, 2, 3, 5, 8, 12, 18]`), each holding up to `n` scenes. When a bucket is full, the lowest-PVL incumbent is evicted.

**PVL scoring**: every finished episode produces a PVL score (positive GAE advantage, averaged over the episode). High PVL ≈ high regret ≈ the agent still has something to learn here. Scoring happens per-bucket so hard tiers don't drown out easy ones.

**Genetic operators**: after a warmup period, roughly half of episodes come from the buffer via one of three operations.
- **crossover**: pick two scenes from the same bucket (PVL-weighted), swap parameter *groups* (ego, distractors, brake-checker, cluster, etc.).
- **mutation**: pick one scene, resample one parameter group.
- **replay**: rerun a stored scene, update its PVL.

Bucket selection and within-bucket sampling are both PVL-weighted, so high-regret regions get more draws without ever starving the easier ones.

The Scenic programs under [`policy/scenarios/`](policy/scenarios/) parameterize things like intersection occupancy, cluster size, brake-checker distance, and nearest-distractor spacing. [`driver.scenic`](policy/scenarios/driver.scenic) is the main one.

## Results

Four seeds, 1M timesteps each on `driver.scenic`.

| metric | Baseline | ACL | N×K Buffer |
|---|---|---|---|
| Mean Reward | 41.3 ± 17.8 | 43.5 ± 11.9 | **70.2 ± 19.1** |
| Mean Episode Length | 150.8 ± 101.4 | 80.9 ± 15.7 | **496.7 ± 104.9** |
| Mean Difficulty | 10.8 ± 0.8 | **11.6 ± 1.2** | 11.1 ± 1.5 |
| Crash Vehicle % | 60.9 | 58.3 | **21.2** |

Baseline is random scene sampling. ACL is the ablation. N×K Buffer is the full method, giving ~65% drop in collision rate against both references, episodes that last ~3.3× longer, and mean reward up 70%.

## Setup

Three source dependencies. The project pulls from forks of Scenic and MetaDrive (`Kv139`) that add the hooks the buffer needs, plus upstream VerifAI for sampling. Clone them as siblings of this repo and install editable.

```
D:/SIP2026/
├── ACL-experiments/      # this repo
├── Scenic/               # https://github.com/Kv139/Scenic
├── metadrive/            # https://github.com/Kv139/metadrive
└── VerifAI/              # https://github.com/BerkeleyLearnVerify/VerifAI
```

```bash
git clone https://github.com/Kv139/Scenic.git
git clone https://github.com/Kv139/metadrive.git
git clone https://github.com/BerkeleyLearnVerify/VerifAI.git

pip install -e ./Scenic
pip install -e ./metadrive
pip install -e ./VerifAI
pip install torch tyro tensorboard gymnasium numpy
```

## Running

Everything runs from `policy/`.

```bash
cd policy
python ppo.py --capacity 100 --seed 1                                    # baseline
python ppo.py --apply_genetic_ops 1 --seed 1                             # ACL ablation
python ppo.py --nk_buffer 1 --apply_genetic_ops 1 --seed 1               # full N×K
```

Full sweep is in [`policy/run_exp.sh`](policy/run_exp.sh). Episode-level CSVs land in `policy/logs/`, TensorBoard runs in `policy/runs/`.

Key flags:
- `--scenic_file ./scenarios/driver.scenic` picks which Scenic program to sample from
- `--capacity 100` sets scenes per bucket
- `--start_genetic 25` sets the warmup before genetic ops kick in
- `--total_timesteps 1000000`

## Layout

```
policy/
  ppo.py                      # training entry (CleanRL PPO, lightly modified)
  custom/
    buffer.py                 # Scene, Category, Buffer
    gym_w_buffer.py           # Gym env that owns the buffer and runs the operators
    custom_simulator.py       # MetaDrive + Scenic glue
  scenarios/
    driver.scenic             # main scenario (intersections, clusters, brake-checkers)
    braking.scenic, etc.      # other test scenarios
  logs/                       # per-episode CSVs
  runs/                       # TensorBoard
CARLA/                        # SUMO + OpenDRIVE maps (Town01 through Town10)
```

The `*_cop.py` and `buffer2.py` files are old iterations kept around for diffs. The live code is [`buffer.py`](policy/custom/buffer.py) and [`gym_w_buffer.py`](policy/custom/gym_w_buffer.py).
