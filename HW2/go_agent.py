"""
HW2 Task 1: Go Game Agent
5x5 board, minimax with alpha-beta pruning
"""
import copy
import numpy as np

BOARD_SIZE = 5
DIRECTIONS = [[1,0],[0,1],[-1,0],[0,-1]]

def parse_input():
    with open("input.txt", 'r') as f:
        lines = f.readlines()
        color = int(lines[0])
        prev = [[int(c) for c in line.strip()] for line in lines[1:1+BOARD_SIZE]]
        curr = [[int(c) for c in line.strip()] for line in lines[1+BOARD_SIZE:1+2*BOARD_SIZE]]
    return color, prev, curr

def output_result(move):
    with open("output.txt", 'w') as f:
        if move is None or move == ("r", "r"):
            f.write("PASS")
        else:
            f.write(f"{move[0]},{move[1]}")

class GoAgent:
    def __init__(self, color, prev_board, curr_board):
        self.color = color
        self.prev_board = prev_board
        self.curr_board = curr_board

    def get_best_move(self, depth, total_moves):
        move, _ = self._maximize(0, float('-inf'), float('inf'), self.color, depth, 0, self.curr_board, total_moves, None)
        output_result(move)

    def _maximize(self, is_over, alpha, beta, player, max_depth, curr_depth, board, move_count, last_move):
        best_score = float('-inf')
        if curr_depth == max_depth or move_count + curr_depth >= 24 or is_over:
            return None, self.evaluate_board(self.color, board)
        next_moves = self.get_valid_moves(player, board) + [("r", "r")]
        if last_move == ("r", "r"):
            is_over = 1
        best_action = None
        for move in next_moves[:20]:
            next_board = copy.deepcopy(board) if move == ("r", "r") else self.simulate_move(player, board, move)
            _, score = self._minimize(is_over, alpha, beta, 3 - player, max_depth, curr_depth + 1, next_board, move_count, move)
            if score > best_score:
                best_score, best_action = score, move
            if best_score >= beta:
                return (best_action, best_score) if curr_depth == 0 else (None, best_score)
            alpha = max(alpha, best_score)
        return (best_action, best_score) if curr_depth == 0 else (None, best_score)

    def _minimize(self, is_over, alpha, beta, player, max_depth, curr_depth, board, move_count, last_move):
        worst_score = float('inf')
        if curr_depth == max_depth or move_count + curr_depth >= 24 or is_over:
            return None, self.evaluate_board(self.color, board)
        next_moves = self.get_valid_moves(player, board) + [("r", "r")]
        if last_move == ("r", "r"):
            is_over = 1
        for move in next_moves[:20]:
            next_board = copy.deepcopy(board) if move == ("r", "r") else self.simulate_move(player, board, move)
            _, score = self._maximize(is_over, alpha, beta, 3 - player, max_depth, curr_depth + 1, next_board, move_count, move)
            worst_score = min(worst_score, score)
            if worst_score <= alpha:
                return None, worst_score
            beta = min(beta, worst_score)
        return None, worst_score

    def _count_q1(self, player, block):
        bl, br, tl, tr = block[0][0], block[0][1], block[1][0], block[1][1]
        return int(any([bl==player and br!=player and tl!=player and tr!=player,
            bl!=player and br==player and tl!=player and tr!=player,
            bl!=player and br!=player and tl==player and tr!=player,
            bl!=player and br!=player and tl!=player and tr==player]))

    def _count_q3(self, player, block):
        bl, br, tl, tr = block[0][0], block[0][1], block[1][0], block[1][1]
        return int(any([bl==player and br!=player and tl!=player and tr==player,
            bl!=player and br==player and tl==player and tr!=player]))

    def _count_qd(self, player, block):
        bl, br, tl, tr = block[0][0], block[0][1], block[1][0], block[1][1]
        return int(any([bl==player and br==player and tl==player and tr!=player,
            bl!=player and br==player and tl==player and tr==player,
            bl==player and br!=player and tl==player and tr==player,
            bl!=player and br==player and tl==player and tr==player]))

    def euler_number(self, player, board):
        padded = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                padded[i + 1][j + 1] = board[i][j]
        q1_self = q3_self = qd_self = q1_opp = q3_opp = qd_opp = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                block = padded[i:i+2, j:j+2]
                q1_self += self._count_q1(player, block)
                q3_self += self._count_q3(player, block)
                qd_self += self._count_qd(player, block)
                opp = 3 - player
                q1_opp += self._count_q1(opp, block)
                q3_opp += self._count_q3(opp, block)
                qd_opp += self._count_qd(opp, block)
        return (2 * q3_self - (q1_opp - qd_opp + 2 * q3_opp) + q1_self - qd_self) / 4

    def evaluate_board(self, player, board):
        own_stones = sum(row.count(player) for row in board)
        opp_stones = sum(row.count(3 - player) for row in board)
        own_libs = set()
        opp_libs = set()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == 0:
                    for dx, dy in DIRECTIONS:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                            if board[ni][nj] == player:
                                own_libs.add((i, j))
                            elif board[ni][nj] == 3 - player:
                                opp_libs.add((i, j))
        edge_penalty = sum(board[0][j]==player or board[BOARD_SIZE-1][j]==player for j in range(BOARD_SIZE))
        edge_penalty += sum(board[i][0]==player or board[i][BOARD_SIZE-1]==player for i in range(1, BOARD_SIZE-1))
        euler_val = self.euler_number(player, board)
        score = min(max(len(own_libs)-len(opp_libs),-4),4) - edge_penalty + (-4*euler_val) + 5*(own_stones-opp_stones)
        return score + (2.5 if self.color == 2 else 0)

    def simulate_move(self, player, board, move):
        new_board = copy.deepcopy(board)
        new_board[move[0]][move[1]] = player
        for dx, dy in DIRECTIONS:
            x, y = move[0]+dx, move[1]+dy
            if 0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE and new_board[x][y]==3-player:
                visited, stack = set(), [(x,y)]
                captured = True
                while stack:
                    cx, cy = stack.pop()
                    visited.add((cx,cy))
                    for ddx, ddy in DIRECTIONS:
                        nx, ny = cx+ddx, cy+ddy
                        if 0<=nx<BOARD_SIZE and 0<=ny<BOARD_SIZE:
                            if (nx,ny) in visited: continue
                            if new_board[nx][ny]==0: captured=False; break
                            elif new_board[nx][ny]==3-player: stack.append((nx,ny))
                    if not captured: break
                if captured:
                    for (cx,cy) in visited: new_board[cx][cy]=0
        return new_board

    def get_valid_moves(self, player, board):
        top, mid, low = [], [], []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j]!=0: continue
                test = copy.deepcopy(board)
                test[i][j]=player
                for dx,dy in DIRECTIONS:
                    ni,nj=i+dx,j+dy
                    if 0<=ni<BOARD_SIZE and 0<=nj<BOARD_SIZE and test[ni][nj]==3-player:
                        if self.check_liberty(ni,nj,test,3-player)==2:
                            self.remove_group(ni,nj,test,3-player)
                if self.check_liberty(i,j,test,player)==1:
                    if self.check_ko(i,j)==2:
                        (low if i==0 or j==0 or i==BOARD_SIZE-1 or j==BOARD_SIZE-1 else mid).append((i,j))
                else:
                    for dx,dy in DIRECTIONS:
                        ni,nj=i+dx,j+dy
                        if 0<=ni<BOARD_SIZE and 0<=nj<BOARD_SIZE and board[ni][nj]==3-player:
                            cap=copy.deepcopy(board)
                            cap[i][j]=player
                            if self.check_liberty(ni,nj,cap,3-player)==2 and self.check_ko(i,j)==2:
                                top.append((i,j))
                            break
        return top+mid+low

    def check_liberty(self,x,y,board,player):
        visited, stack = set(), [(x,y)]
        while stack:
            i,j = stack.pop()
            visited.add((i,j))
            for dx,dy in DIRECTIONS:
                ni,nj=i+dx,j+dy
                if 0<=ni<BOARD_SIZE and 0<=nj<BOARD_SIZE:
                    if (ni,nj) in visited: continue
                    if board[ni][nj]==0: return 1
                    if board[ni][nj]==player: stack.append((ni,nj))
        return 2

    def check_ko(self,x,y):
        if self.prev_board[x][y]!=self.color: return 2
        new=copy.deepcopy(self.curr_board)
        new[x][y]=self.color
        for dx,dy in DIRECTIONS:
            nx,ny=x+dx,y+dy
            if 0<=nx<BOARD_SIZE and 0<=ny<BOARD_SIZE and new[nx][ny]==3-self.color:
                if self.check_liberty(nx,ny,new,3-self.color)==2:
                    self.remove_group(nx,ny,new,3-self.color)
        return 1 if np.array_equal(new,self.prev_board) else 2

    def remove_group(self,x,y,board,player):
        stack, visited = [(x,y)], set()
        while stack:
            i,j = stack.pop()
            visited.add((i,j))
            board[i][j]=0
            for dx,dy in DIRECTIONS:
                ni,nj=i+dx,j+dy
                if 0<=ni<BOARD_SIZE and 0<=nj<BOARD_SIZE and (ni,nj) not in visited and board[ni][nj]==player:
                    stack.append((ni,nj))
        return board

def main():
    color, prev_board, curr_board = parse_input()
    move_num = 0
    try:
        with open("num.txt",'r') as f:
            move_num = int(f.read()) + 2
    except: pass
    with open("num.txt",'w') as f:
        f.write(str(move_num))
    agent = GoAgent(color, prev_board, curr_board)
    agent.get_best_move(depth=4, total_moves=move_num)

if __name__ == '__main__':
    main()
