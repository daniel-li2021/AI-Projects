import numpy as np
BOARD_SIZE = 3
ONGOING, DRAW, X_WIN, O_WIN = -1, 0, 1, 2

class Board:
    def __init__(self, state=None, show_board=False, show_result=False):
        self.state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int) if state is None else state.copy()
        self.game_result = ONGOING
        self.show_board, self.show_result = show_board, show_result

    def encode_state(self):
        return ''.join(str(self.state[i][j]) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE))

    def reset(self):
        self.state.fill(0)
        self.game_result = ONGOING

    def is_valid_move(self, row, col):
        return 0<=row<BOARD_SIZE and 0<=col<BOARD_SIZE and self.state[row][col]==0

    def move(self, row, col, player):
        if not self.is_valid_move(row, col):
            raise ValueError("Invalid Move")
        self.state[row][col] = player
        self.game_result = self._check_winner()
        return self.state, self.game_result

    def game_over(self):
        return self.game_result != ONGOING

    def _check_winner(self):
        for i in range(3):
            if self.state[i][0]>0 and self.state[i][0]==self.state[i][1]==self.state[i][2]:
                return X_WIN if self.state[i][0]==1 else O_WIN
            if self.state[0][i]>0 and self.state[0][i]==self.state[1][i]==self.state[2][i]:
                return X_WIN if self.state[0][i]==1 else O_WIN
        if self.state[1][1]>0 and self.state[0][0]==self.state[1][1]==self.state[2][2]:
            return X_WIN if self.state[1][1]==1 else O_WIN
        if self.state[1][1]>0 and self.state[2][0]==self.state[1][1]==self.state[0][2]:
            return X_WIN if self.state[1][1]==1 else O_WIN
        return DRAW if (self.state==0).sum()==0 else ONGOING
