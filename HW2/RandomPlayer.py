import random

class RandomPlayer:
    def set_side(self, side):
        self.side = side

    def move(self, board):
        if board.game_over():
            return
        valid = [(i,j) for i in range(3) for j in range(3) if board.is_valid_move(i,j)]
        row, col = random.choice(valid)
        return board.move(row, col, self.side)

    def learn(self, board):
        pass
