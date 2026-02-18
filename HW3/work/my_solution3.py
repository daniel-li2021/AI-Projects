"""
CSCI 561 Spring 2025 HW3: POMDP Temporal Reasoning
Implements Viterbi algorithm for:
  - Scenario 1: Little Prince (states, observations, actions)
  - Scenario 2: Speech Recognition (graphemes, phonemes, action "N")
"""
import re
from collections import defaultdict

def parse_quoted(s):
    """Extract quoted strings from a line."""
    return re.findall(r'"([^"]*)"', s)

def load_state_weights(path):
    """Parse state_weights.txt -> dict state -> P(s)"""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()][1:]
    states, weights = [], []
    for line in lines:
        parts = parse_quoted(line)
        nums = re.findall(r'\d+', line)
        if len(parts) >= 1:
            states.append(parts[0])
            w = int(nums[-1]) if nums else 1
            weights.append(w)
    total = sum(weights)
    return {s: w/total for s, w in zip(states, weights)}, states

def load_state_action_state(path, states, actions, default_w):
    """Parse state_action_state_weights.txt -> P(s'|s,a)"""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()][1:]
    header = lines[0].split()
    default_w = int(header[-1])
    weight = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: default_w)))
    all_actions = set()
    for line in lines[1:]:
        parts = parse_quoted(line)
        nums = re.findall(r'\d+', line)
        if len(parts) >= 3:
            s, a, s2 = parts[0], parts[1], parts[2]
            all_actions.add(a)
            w = int(nums[-1]) if nums else default_w
            weight[s][a][s2] = w
    act_list = actions if actions else list(all_actions)
    if not act_list:
        act_list = ["N"]
    trans = {}
    for s in states:
        trans[s] = {}
        for a in act_list:
            row = {s2: weight[s][a][s2] for s2 in states}
            total = sum(row.values())
            trans[s][a] = {s2: row[s2]/total if total > 0 else 1.0/len(states) for s2 in states}
    return trans

def load_state_observation(path, states):
    """Parse state_observation_weights.txt -> P(o|s)"""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()][1:]
    header = lines[0].split()
    default_w = int(header[-1])
    weight = defaultdict(lambda: defaultdict(lambda: default_w))
    for line in lines[1:]:
        parts = parse_quoted(line)
        nums = re.findall(r'\d+', line)
        if len(parts) >= 2:
            s, o = parts[0], parts[1]
            w = int(nums[-1]) if nums else default_w
            weight[s][o] = w
    obs = {}
    for s in states:
        row = dict(weight[s])
        total = sum(row.values())
        obs[s] = {o: row[o]/total if total > 0 else 0 for o in row}
    return obs

def load_observation_actions(path):
    """Parse observation_actions.txt -> list of (obs, action)"""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()][1:]
    pairs = []
    for line in lines:
        parts = parse_quoted(line)
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
        elif len(parts) == 1:
            pairs.append((parts[0], "N"))
    return pairs

def viterbi(init, trans, obs_prob, obs_actions, states):
    """Viterbi: most probable state sequence.
    Joint: P(s0)*P(o0|s0) * P(s1|s0,a0)*P(o1|s1) * ... * P(sT|s_{T-1},a_{T-1})*P(oT|sT)
    """
    T = len(obs_actions)
    if T == 0:
        return []
    dp = [{} for _ in range(T)]
    bp = [{} for _ in range(T)]
    o0, a0 = obs_actions[0]
    for s in states:
        p_obs = obs_prob.get(s, {}).get(o0, 1e-20)
        dp[0][s] = init.get(s, 1e-20) * p_obs
        bp[0][s] = None
    for t in range(1, T):
        ot, at = obs_actions[t]
        prev_act = obs_actions[t-1][1]
        for s in states:
            best, best_prev = -1, None
            p_o = obs_prob.get(s, {}).get(ot, 1e-20)
            for s_prev in states:
                p_trans = trans.get(s_prev, {}).get(prev_act, {}).get(s, 1e-20)
                cand = dp[t-1][s_prev] * p_trans * p_o
                if cand > best:
                    best, best_prev = cand, s_prev
            dp[t][s] = best
            bp[t][s] = best_prev
    last = max(states, key=lambda s: dp[T-1].get(s, 0))
    seq = [last]
    for t in range(T-1, 0, -1):
        seq.append(bp[t][seq[-1]])
    seq.reverse()
    return seq

def main():
    init, states = load_state_weights("state_weights.txt")
    trans = load_state_action_state("state_action_state_weights.txt", states, None, 0)
    obs_prob = load_state_observation("state_observation_weights.txt", states)
    obs_actions = load_observation_actions("observation_actions.txt")
    seq = viterbi(init, trans, obs_prob, obs_actions, states)
    with open("states.txt", "w") as f:
        f.write("states\n")
        f.write(str(len(seq)) + "\n")
        for s in seq:
            f.write('"' + s + '"\n')

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()
