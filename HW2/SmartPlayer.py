"""Heuristic player: block wins, take wins, prefer center."""

class SmartPlayer:
    def set_side(self, side):
        self.side = side

    def _winning_move(self, board, player):
        for i in range(3):
            for j in range(3):
                if board.is_valid_move(i, j):
                    board.state[i][j] = player
                    if board._check_winner() != -1:
                        board.state[i][j] = 0
                        return (i, j)
                    board.state[i][j] = 0
        return None

    def move(self, board):
        if board.game_over():
            return
        opp = 3 - self.side
        m = self._winning_move(board, self.side)
        if m:
            return board.move(m[0], m[1], self.side)
        m = self._winning_move(board, opp)
        if m:
            return board.move(m[0], m[1], self.side)
        if board.is_valid_move(1, 1):
            return board.move(1, 1, self.side)
        corners = [(0,0),(0,2),(2,0),(2,2)]
        for i, j in corners:
            if board.is_valid_move(i, j):
                return board.move(i, j, self.side)
        for i in range(3):
            for j in range(3):
                if board.is_valid_move(i, j):
                    return board.move(i, j, self.side)

    def learn(self, board):
        pass
