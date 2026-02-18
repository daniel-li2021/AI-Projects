"""Minimax optimal player for Tic-Tac-Toe."""

class PerfectPlayer:
    def set_side(self, side):
        self.side = side

    def _minimax(self, board, player, depth):
        r = board._check_winner()
        if r != -1:
            return (10-depth if r==self.side else -10+depth if r==3-self.side else 0)
        best = -1000 if player==self.side else 1000
        for i in range(3):
            for j in range(3):
                if board.is_valid_move(i, j):
                    board.state[i][j] = player
                    score = self._minimax(board, 3-player, depth+1)
                    board.state[i][j] = 0
                    best = max(best, score) if player==self.side else min(best, score)
        return best

    def move(self, board):
        if board.game_over():
            return
        best_move, best_score = None, -1000
        for i in range(3):
            for j in range(3):
                if board.is_valid_move(i, j):
                    board.state[i][j] = self.side
                    s = self._minimax(board, 3-self.side, 0)
                    board.state[i][j] = 0
                    if s > best_score:
                        best_score, best_move = s, (i, j)
        return board.move(best_move[0], best_move[1], self.side)

    def learn(self, board):
        pass
