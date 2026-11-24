"""Array-based fast path for value-function runs (speed review item 4).

EXPERIMENTAL and OPT-IN: llm_runner uses it only when VF_FAST_PATH=1 is set;
the default is the object-grid path in base_simulation.Simulation.update_agents.

What it reproduces, decision for decision, of the object-grid path:

* the initial grid: built by the unchanged Simulation.__init__ (same numpy
  shuffles, same agent_id per agent), then read once into arrays;
* the visiting order: np.random.shuffle on a list of the occupied cells in
  row-major order, the same call on a list of the same length as
  update_agents makes (the permutation depends only on the length);
* sequential decisions: agents still decide and move one at a time, in that
  order, and neighbour counts are updated after every move. Nothing is
  decided from the start-of-step grid;
* the random numbers: keyed_uniform(run_id, step, agent_id, salt) for every
  agent at once, as uint64 splitmix64 in numpy. The four chained mixes are
  the same 64-bit operations (numpy uint64 wraps mod 2**64 like the & mask),
  and (h >> 11) * 2**-53 is exact in float64 on both paths;
* the move rule: MOVE iff u_accept < P[type, n_similar, n_occupied], with the
  same float P taken from the same policy dict; the destination is
  empty_cells[int(u_dest * len(empty_cells))] with the empty cells kept as a
  sorted list of flat indices (flat order == row-major (r, c) order);
* the record: one int8 frame per step, the same metrics, the same step-log
  row (decisions, moved, successful_move, chose_to_stay).

Only the keyed RNG scheme and the per-step record are supported; anything
else (VF_RNG_SCHEME=shared, FULL_MOVE_LOG=1, live-LLM runs) raises rather
than silently falling back.
"""

import bisect

import numpy as np

import config as cfg
import run_files
import llm_runner

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = (np.uint64(s) for s in (30, 27, 31, 11))
_U53 = 1.0 / (1 << 53)


def splitmix64_np(x):
    """llm_runner._splitmix64 on a uint64 array (wrapping arithmetic)."""
    with np.errstate(over='ignore'):
        x = x + _GOLDEN
        x = (x ^ (x >> _S30)) * _M1
        x = (x ^ (x >> _S27)) * _M2
    return x ^ (x >> _S31)


def keyed_uniforms_np(run_id, step, agent_ids, salts=(0, 1)):
    """[keyed_uniform(run_id, step, a, salt) for a in agent_ids] per salt, as float64 arrays.

    The first three splitmix rounds (run, step, agent) are shared by the salts.
    """
    h = llm_runner._splitmix64(int(run_id) & llm_runner._MASK64)
    h = llm_runner._splitmix64(h ^ (int(step) & llm_runner._MASK64))
    h = splitmix64_np(np.uint64(h) ^ agent_ids)
    return [(splitmix64_np(h ^ np.uint64(salt)) >> _S11).astype(np.float64) * _U53
            for salt in salts]


def policy_table(policy):
    """Flat list P[t * 81 + n_sim * 9 + n_occ] of the policy's MOVE probabilities.

    Same floats as the dict; a (type, comp) cell the policy lacks is None, so a
    lookup of it fails as the dict lookup would (KeyError there, TypeError here).
    """
    table = [None] * (2 * 81)
    for t, key in ((0, 'type_a'), (1, 'type_b')):
        for n_occ in range(9):
            for n_sim in range(n_occ + 1):
                p = policy.get((key, f"comp:{n_sim},{n_occ}"))
                table[t * 81 + n_sim * 9 + n_occ] = p
    return table


class FastVFSimulation(llm_runner.LLMSimulation):
    """LLMSimulation whose step works on arrays instead of Agent objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.value_function_policy:
            raise ValueError("the fast path is for value-function runs only")
        if llm_runner.VF_RNG_SCHEME != 'keyed':
            raise ValueError("the fast path implements VF_RNG_SCHEME=keyed only")
        if self.full_move_log:
            raise ValueError("the fast path writes the per-step record only")
        G = cfg.GRID_SIZE
        self._G = G
        n_cells = G * G
        # Cell state, read once from the object grid built by Simulation.__init__.
        self._type = [-1] * n_cells          # -1 empty, else type_id
        self._aid = [-1] * n_cells           # agent_id of the agent in the cell
        for r in range(G):
            for c in range(G):
                agent = self.grid[r][c]
                if agent is not None:
                    self._type[r * G + c] = int(agent.type_id)
                    self._aid[r * G + c] = int(agent.agent_id)
        if any(t not in (-1, 0, 1) for t in self._type):
            raise ValueError("the fast path supports type ids 0 and 1 only")
        self._n_agents = sum(1 for t in self._type if t >= 0)
        # Neighbour lists (8-neighbourhood, walls excluded), as flat indices.
        self._nbrs = []
        for r in range(G):
            for c in range(G):
                self._nbrs.append([(r + dr) * G + (c + dc)
                                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                   if (dr or dc) and 0 <= r + dr < G and 0 <= c + dc < G])
        # Per cell: occupied neighbours, and type-0 neighbours.
        self._occ = [0] * n_cells
        self._n0 = [0] * n_cells
        for cell in range(n_cells):
            for nb in self._nbrs[cell]:
                if self._type[nb] >= 0:
                    self._occ[cell] += 1
                    if self._type[nb] == 0:
                        self._n0[cell] += 1
        self._P = policy_table(self.value_function_policy)
        self._decisions = 0
        self._agent_ids = None               # uint64 0..max_id, built lazily

    # --- the step -----------------------------------------------------------
    def update_agents(self, verbose_move_log=False):
        G = self._G
        ttype, aid, occ, n0, nbrs, P = self._type, self._aid, self._occ, self._n0, self._nbrs, self._P
        # Same call as update_agents: shuffle the row-major list of occupied
        # cells (the permutation depends only on the list's length).
        order = [cell for cell in range(G * G) if ttype[cell] >= 0]
        np.random.shuffle(order)
        empty = [cell for cell in range(G * G) if ttype[cell] < 0]   # sorted

        if self._agent_ids is None:
            self._agent_ids = np.arange(max(aid) + 1, dtype=np.uint64)
        u_acc, u_dst = (u.tolist() for u in keyed_uniforms_np(self.run_id, self.step, self._agent_ids))

        moves = 0
        for cell in order:
            t = ttype[cell]
            a = aid[cell]
            o = occ[cell]
            s = n0[cell] if t == 0 else o - n0[cell]
            if u_acc[a] < P[t * 81 + s * 9 + o] and empty:
                k = int(u_dst[a] * len(empty))
                dest = empty[k]
                # Move: dest becomes occupied, cell becomes empty.
                ttype[dest] = t
                aid[dest] = a
                ttype[cell] = -1
                aid[cell] = -1
                del empty[k]
                bisect.insort(empty, cell)
                for nb in nbrs[cell]:
                    occ[nb] -= 1
                for nb in nbrs[dest]:
                    occ[nb] += 1
                if t == 0:
                    for nb in nbrs[cell]:
                        n0[nb] -= 1
                    for nb in nbrs[dest]:
                        n0[nb] += 1
                moves += 1

        n = len(order)
        if n:
            row = run_files.new_step_row(self.step)
            row['decisions'] = n
            row['moved'] = moves
            row['successful_move'] = moves
            row['chose_to_stay'] = n - moves
            if self.step_log and self.step_log[-1]['step'] == self.step:
                raise RuntimeError("two update_agents calls for one step")
            self.step_log.append(row)
        self._decisions += n
        return moves > 0

    def _grid_to_int(self):
        if not hasattr(self, '_type'):          # during Simulation.__init__
            return super()._grid_to_int()
        return np.array(self._type, dtype=np.int8).reshape(self._G, self._G)

    def _collect_llm_counters(self):
        # One value-function "call" per decision, as LLMAgent counts them.
        self.total_llm_calls += self._decisions
        self._decisions = 0
