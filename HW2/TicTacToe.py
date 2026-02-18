"""HW2 Task 2: Q-Learning for Tic-Tac-Toe"""
from Board import Board
from RandomPlayer import RandomPlayer
from QLearner import QLearner
from PerfectPlayer import PerfectPlayer
from SmartPlayer import SmartPlayer

PLAYER_X, PLAYER_O = 1, 2

def play(board, p1, p2, learn):
    p1.set_side(PLAYER_X)
    p2.set_side(PLAYER_O)
    while not board.game_over():
        p1.move(board)
        if not board.game_over():
            p2.move(board)
    if learn:
        p1.learn(board)
        p2.learn(board)
    return board.game_result

def battle(board, p1, p2, n, learn=False, show=True):
    stats = [0, 0, 0]
    for _ in range(n):
        stats[play(board, p1, p2, learn)] += 1
        board.reset()
    stats = [round(x/n*100, 1) for x in stats]
    if show:
        print('_'*60)
        print(f'{p1.__class__.__name__}(X) | Wins:{stats[1]}% Draws:{stats[0]}% Losses:{stats[2]}%')
        print(f'{p2.__class__.__name__}(O) | Wins:{stats[2]}% Draws:{stats[0]}% Losses:{stats[1]}%')
        print('_'*60)
    return stats

if __name__ == "__main__":
    qlearner = QLearner()
    board = Board()
    print(f'Training QLearner for {qlearner.GAME_NUM} games...')
    battle(board, RandomPlayer(), qlearner, qlearner.GAME_NUM, learn=True, show=False)
    battle(board, qlearner, RandomPlayer(), qlearner.GAME_NUM, learn=True, show=False)
    print('Evaluation:')
    q_rand = battle(board, qlearner, RandomPlayer(), 500)
    rand_q = battle(board, RandomPlayer(), qlearner, 500)
    q_smart = battle(board, qlearner, SmartPlayer(), 500)
    smart_q = battle(board, SmartPlayer(), qlearner, 500)
    q_perfect = battle(board, qlearner, PerfectPlayer(), 500)
    perfect_q = battle(board, PerfectPlayer(), qlearner, 500)
    wr_rand = round(100 - (q_rand[2]+rand_q[1])/2, 2)
    wr_smart = round(100 - (q_smart[2]+smart_q[1])/2, 2)
    wr_perfect = round(100 - (q_perfect[2]+perfect_q[1])/2, 2)
    print("QLearner VS RandomPlayer | Win/Draw Rate = {}%".format(wr_rand))
    print("QLearner VS SmartPlayer  | Win/Draw Rate = {}%".format(wr_smart))
    print("QLearner VS PerfectPlayer| Win/Draw Rate = {}%".format(wr_perfect))
