import numpy as np
WIN_REWARD, DRAW_REWARD, LOSS_REWARD = 1.0, 0.5, 0.0

class QLearner:
    GAME_NUM = 100000

    def __init__(self, alpha=.7, gamma=.9, initial_value=0.5, side=None):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        self.side, self.alpha, self.gamma = side, alpha, gamma
        self.q_values, self.history_states = {}, []
        self.initial_value = initial_value

    def set_side(self, side):
        self.side = side

    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((3, 3))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def _find_max(self, q_values):
        curr_max, row, col = -np.inf, 0, 0
        for i in range(3):
            for j in range(3):
                if q_values[i][j] > curr_max:
                    curr_max, row, col = q_values[i][j], i, j
        return row, col

    def _select_best_move(self, board):
        q_values = self.Q(board.encode_state())
        while True:
            i, j = self._find_max(q_values)
            if board.is_valid_move(i, j):
                return i, j
            q_values[i][j] = -1.0

    def move(self, board):
        if board.game_over():
            return
        row, col = self._select_best_move(board)
        self.history_states.append((board.encode_state(), (row, col)))
        return board.move(row, col, self.side)

    def learn(self, board):
        reward = DRAW_REWARD if board.game_result==0 else (WIN_REWARD if board.game_result==self.side else LOSS_REWARD)
        self.history_states.reverse()
        max_q = -1.0
        for state, move in self.history_states:
            q = self.Q(state)
            if max_q < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]]*(1-self.alpha) + self.alpha*self.gamma*max_q
            max_q = np.max(q)
        self.history_states = []
