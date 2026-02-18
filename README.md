# AI Projects — USC CSCI 561

Three projects from **USC CSCI 561: Foundations of Artificial Intelligence**, implementing core AI algorithms from scratch.

**Technologies:** Python, NumPy — no ML frameworks (implementations from first principles)

---

## Projects Overview

### HW1 — Genetic Algorithm for 3D Traveling Salesman Problem

- **Topic:** Genetic algorithms for combinatorial optimization  
- **Approach:** Tournament selection, order crossover, 2-opt local search, swap mutation  
- **Input:** `input.txt` — number of cities and 3D coordinates  
- **Output:** `output.txt` — total distance and tour  

**Run:**
```bash
cd HW1
python homework1.py
```

---

### HW2 — Game Playing Agents

#### Task 1: Go Agent (5×5 board)
- **Method:** Minimax with alpha–beta pruning  
- **Heuristics:** Liberty count, Euler number, capture moves  
- **Input:** `input.txt` — color and board states  
- **Output:** `output.txt` — move or PASS  

**Run:** (requires `input.txt` with board state; sample included)
```bash
cd HW2
python go_agent.py
```

#### Task 2: Q-Learning for Tic-Tac-Toe
- **Method:** Q-learning with tabular Q-values  
- **Training:** 100K games vs random players  
- **Evaluation:** Win/draw rates vs Random, Smart, and Perfect players  

**Run:**
```bash
cd HW2
python TicTacToe.py
```

---

### HW3 — POMDP Temporal Reasoning (Viterbi)

- **Topic:** Partially Observable Markov Decision Process (POMDP)  
- **Method:** Viterbi algorithm for most probable hidden state sequence  
- **Scenarios:**
  1. **Little Prince** — states, observations (Volcano/Grass/Apple), actions (Forward/Backward/Turnaround)
  2. **Speech Recognition** — graphemes (states), phonemes (observations), action "N"  

**Input files (provided by Vocareum):**
- `state_weights.txt` — initial P(s)
- `state_action_state_weights.txt` — transition P(s'|s,a)
- `state_observation_weights.txt` — emission P(o|s)
- `observation_actions.txt` — sequence of (observation, action) pairs  

**Output:** `states.txt` — predicted state sequence  

**Run (from `HW3/work` with input files in same directory):**
```bash
cd HW3/work
python3 my_solution3.py
```

---

## Course

- **Course:** CSCI 561, Foundations of Artificial Intelligence (USC)  
- **Submission:** Vocareum  

---

## Repository

**[github.com/daniel-li2021/AI-Projects](https://github.com/daniel-li2021/AI-Projects)**
