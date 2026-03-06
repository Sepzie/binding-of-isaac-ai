# Binding of Isaac RL Agent — Project Plan

## Overview

Train a reinforcement learning agent to play The Binding of Isaac: Afterbirth+,
progressing from single-room combat to full runs. Uses the game's Lua modding API
for state extraction and input injection, with a Python RL training pipeline
communicating over TCP sockets.

Inspired by OpenAI Five's approach to Dota 2: game API (not vision), PPO, self-play,
and curriculum learning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Host (Python)                │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  PPO Agent   │  │  Env Wrapper │  │  Reward Shaper │ │
│  │  (SB3 /      │  │  (Gymnasium) │  │                │ │
│  │   CleanRL)   │  │              │  │                │ │
│  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘ │
│         │                 │                  │          │
│         └────────┬────────┘──────────────────┘          │
│                  │                                      │
│           ┌──────▼───────┐                              │
│           │  TCP Client  │                              │
│           └──────┬───────┘                              │
└──────────────────┼──────────────────────────────────────┘
                   │ TCP Socket (localhost)
┌──────────────────┼──────────────────────────────────────┐
│           ┌──────▼───────┐                              │
│           │  TCP Server  │    Binding of Isaac Process   │
│           └──────┬───────┘                              │
│                  │                                      │
│  ┌───────────────▼──────────────────────────────────┐   │
│  │              Lua Mod (Environment)               │   │
│  │                                                  │   │
│  │  ┌──────────────┐  ┌───────────┐  ┌───────────┐ │   │
│  │  │ State        │  │  Action   │  │  Game     │ │   │
│  │  │ Serializer   │  │  Injector │  │  Control  │ │   │
│  │  │ (obs space)  │  │  (inputs) │  │  (reset)  │ │   │
│  │  └──────────────┘  └───────────┘  └───────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**Lua Mod (runs inside Isaac)**
- `state_serializer.lua` — Reads game state each tick, encodes as JSON, sends over TCP
- `action_injector.lua` — Receives action from Python, maps to Isaac input callbacks
- `game_control.lua` — Handles episode resets (restart run), spawning specific enemies, speed control
- `tcp_server.lua` — Listens for Python client, handles request/response per game tick

**Python Training Pipeline**
- `isaac_env.py` — Gymnasium-compatible environment wrapper. Connects to Lua mod via TCP. Defines observation space, action space, step/reset interface.
- `reward.py` — Reward shaping logic. Computes reward from state diffs (damage dealt, damage taken, pickups, room cleared, etc.)
- `train.py` — Training script. Configures PPO hyperparameters, logging, checkpointing.
- `config.py` — Hyperparameters, observation space dimensions, reward weights.
- `evaluate.py` — Runs trained model in inference mode for evaluation/recording.

---

## Observation Space

### Room State (Grid-Based)
Encode the current room as a multi-channel 2D grid.

Isaac rooms are 13x7 tiles (standard room). Each tile gets multiple channels:

| Channel | Description | Values |
|---|---|---|
| walls | Impassable terrain | 0/1 |
| obstacles | Rocks, pots, poop, etc. | 0/1 |
| pits | Holes in the floor | 0/1 |
| player | Player position | 0/1 |
| enemies | Enemy presence | 0-1 (normalized HP) |
| projectiles | Enemy projectiles | 0/1 (+ velocity encoding?) |
| pickups | Hearts, keys, bombs, coins | categorical |
| doors | Door positions and states | categorical |

Grid shape: `(C, 13, 7)` where C = number of channels (~8-10).

### Player State (Vector)
Appended as a flat vector alongside the grid:

| Feature | Type |
|---|---|
| HP (red hearts) | float |
| HP (soul hearts) | float |
| HP (black hearts) | float |
| Speed | float |
| Damage | float |
| Range | float |
| Fire rate | float |
| Shot speed | float |
| Luck | float |
| Num bombs | int |
| Num keys | int |
| Num coins | int |
| Has active item | bool |
| Active item charge | float |

Vector length: ~14 features.

### Network Architecture
```
Grid obs (C, 13, 7) ──► CNN (Conv2d layers) ──► flatten ──┐
                                                           ├──► FC layers ──► Policy + Value heads
Player state (14,) ──► small MLP ─────────────────────────┘
```

The grid goes through a small CNN (2-3 conv layers), gets flattened, concatenated
with the player state MLP output, then fed through shared FC layers to produce
policy (action probabilities) and value (state value estimate) outputs.

**Later phases** can extend this by adding:
- Floor-level context (minimap: which rooms exist, which visited, door states)
- Item inventory as a binary vector (have/don't have each item)
- The combat CNN weights transfer directly — just grow the network around it.

---

## Action Space

**Multi-Discrete** with two independent heads:

| Head | Options | Description |
|---|---|---|
| Movement | 9 | 8 directions + stand still |
| Shooting | 5 | 4 directions + don't shoot |

Total: 9 x 5 = 45 possible action combinations per tick.

Maps directly to Isaac's digital input system (`ACTION_LEFT`, `ACTION_RIGHT`,
`ACTION_UP`, `ACTION_DOWN`, `ACTION_SHOOTLEFT`, etc.).

---

## Reward Design (Dense)

Starting reward signals — weights will be tuned iteratively:

| Signal | Reward | Notes |
|---|---|---|
| Deal damage to enemy | +1.0 per hit | Encourages aggression |
| Kill enemy | +5.0 | Bonus on top of damage |
| Take damage (lose HP) | -10.0 | Strong penalty |
| Room cleared | +20.0 | Major milestone |
| Pickup collected | +2.0 | Hearts, keys, bombs, coins |
| Death | -50.0 | Episode over |
| Time penalty | -0.1 per tick | Prevents stalling |
| Floor cleared | +100.0 | Phase 3+ |

**Reward shaping principles:**
- Compute from state diffs (current_state - previous_state)
- All rewards normalized relative to episode length
- Track reward component breakdowns in logs for debugging
- Expect significant tuning — these initial values are starting points

---

## Training Phases

### Phase 1: Single Room Combat

**Goal:** Agent clears a single room of enemies reliably.

**Sub-phases:**
1. **1a — Stationary enemy:** Single Gaper (walks toward player). Learn to shoot and kite.
2. **1b — Projectile enemy:** Single Monstro or Leaper. Learn to dodge while attacking.
3. **1c — Multiple enemies:** 3-5 mixed enemies. Learn target prioritization and positioning.

**Setup:**
- Lua mod spawns specific enemies via debug commands
- Fixed room layout (no obstacles initially, add later)
- Episode = one room. Success = all enemies dead. Failure = player death.
- No items, no pickups, base stats only.

**Curriculum:** Train on 1a until >80% win rate, then mix in 1b, then 1c.
Keep earlier scenarios in the mix (don't forget how to fight easy enemies).

### Phase 2: Room Combat + Pickups

**Goal:** Agent collects pickups while fighting.

**Changes from Phase 1:**
- Spawn hearts, bombs, keys, coins alongside enemies
- Add pickup channels to observation grid
- Reward for collection
- Agent should learn to grab hearts when low HP

**Transfer:** Load Phase 1 model weights directly. Same obs shape (just add channels),
same action space.

### Phase 3: Single Floor Navigation

**Goal:** Agent navigates an entire floor — enters rooms, clears them, finds the boss, beats it.

**Changes:**
- Add floor-level observations (minimap: which rooms exist, which visited, door states)
- Add "enter door" as implicit action (walk to door edge)
- New rewards: room discovery, boss kill, floor completion
- Episode = one floor

**Transfer:** Combat CNN weights from Phase 2 frozen initially, train navigation
layers on top. Then unfreeze and fine-tune everything.

### Phase 4: Multi-Floor Runs

**Goal:** Agent completes a full run (Basement through Mom).

**Changes:**
- Episode = full run (multiple floors)
- Add trapdoor navigation
- Increasing difficulty across floors
- Long time horizons — may need LSTM or attention mechanism

**Transfer:** Load Phase 3 model, extend with sequence memory (LSTM layer).

### Phase 5: Item Strategy

**Goal:** Agent makes strategic item choices (pedestal items, devil deals, shop purchases).

**Changes:**
- Remove clean-save constraint, allow item accumulation
- Add item inventory to observation space
- Reward for synergistic item combos (hard — may need manual reward engineering or learned value)
- Item choice = new action head or implicit (walk to item vs. skip)

---

## Directory Structure

```
binding-of-isaac-ai/
├── PLAN.md                     # This document
├── README.md                   # Project overview and setup instructions
│
├── mod/                        # Lua mod (installed into Isaac's mod directory)
│   ├── main.lua                # Mod entry point, callback registration
│   ├── metadata.xml            # Isaac mod metadata
│   ├── tcp_server.lua          # TCP socket server for Python communication
│   ├── state_serializer.lua    # Game state → JSON encoding
│   ├── action_injector.lua     # Action decoding → Isaac input
│   ├── game_control.lua        # Episode reset, enemy spawning, speed control
│   └── config.lua              # Mod configuration (port, tick rate, etc.)
│
├── python/                     # Python training pipeline
│   ├── requirements.txt        # Dependencies (gymnasium, stable-baselines3, torch, etc.)
│   ├── isaac_env.py            # Gymnasium environment wrapper
│   ├── reward.py               # Reward computation from state diffs
│   ├── network.py              # Custom CNN + MLP network architecture
│   ├── train.py                # Training entrypoint
│   ├── evaluate.py             # Evaluation / inference script
│   ├── config.py               # Hyperparameters and constants
│   └── utils.py                # Logging, checkpointing helpers
│
├── configs/                    # Training configs for each phase
│   ├── phase1a.yaml
│   ├── phase1b.yaml
│   ├── phase1c.yaml
│   ├── phase2.yaml
│   ├── phase3.yaml
│   └── phase4.yaml
│
├── checkpoints/                # Saved model weights (gitignored)
├── logs/                       # Training logs / TensorBoard (gitignored)
└── scripts/                    # Utility scripts
    ├── install_mod.sh          # Symlink mod/ into Isaac's mod directory
    └── launch_training.sh      # Start Isaac + training pipeline
```

---

## TCP Protocol

Simple JSON-based request/response over TCP, one exchange per game tick.

### Python → Lua (Action)
```json
{
  "action": {
    "move": 3,
    "shoot": 1
  },
  "command": "step"
}
```

Commands: `step` (normal tick), `reset` (restart episode), `configure` (change settings).

### Lua → Python (Observation)
```json
{
  "grid": [[...], ...],
  "player": {
    "hp_red": 6,
    "hp_soul": 0,
    "speed": 1.0,
    "damage": 3.5,
    "position": [320, 280]
  },
  "enemies": [
    {"type": 10, "hp": 15, "position": [200, 150]}
  ],
  "pickups": [],
  "room_cleared": false,
  "player_dead": false,
  "tick": 142
}
```

### Tick Timing
- Isaac runs at 30 logic ticks/sec (MC_POST_UPDATE)
- Each tick: Lua sends state → Python computes action → Python sends action → Lua applies
- Action applied on the next tick (1-tick delay, acceptable)
- For faster training: skip frames (act every N ticks, repeat last action)

---

## Key Dependencies

| Package | Purpose |
|---|---|
| Python 3.10+ | Training pipeline |
| PyTorch | Neural network backend |
| Stable-Baselines3 | PPO implementation |
| Gymnasium | Environment interface standard |
| TensorBoard | Training visualization |
| luasocket | TCP networking in Lua mod |

---

## Open Questions / Risks

1. **Game speed** — Can we run Isaac faster than real-time? Investigate debug console
   commands, frame-skip, or engine modifications. This is the #1 throughput bottleneck.

2. **Parallel instances** — Can we run multiple Isaac processes simultaneously for
   parallel rollout collection? Each would need its own TCP port. May need virtual
   displays (Xvfb) on Linux.

3. **Projectile encoding** — Projectiles move fast and matter a lot. Encoding just
   position may not be enough; velocity/direction channels may be needed. TBD.

4. **Randomness** — Isaac is heavily randomized (room layouts, enemy spawns, items).
   The agent must generalize across this variance. May need large training budgets.

5. **luasocket availability** — Isaac's Lua environment is sandboxed. Need to confirm
   luasocket works, or find an alternative (file-based IPC as fallback).

6. **Reward hacking** — Dense rewards risk the agent finding degenerate strategies
   (e.g., farming time penalty by dying fast). Monitor reward component breakdowns.
