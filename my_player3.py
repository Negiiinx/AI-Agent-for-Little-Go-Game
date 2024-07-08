import numpy as np
import random 

def sr(B,x,y,p):
    N = len(B); v = [[False]*N for _ in range(N)]; s = [(x,y)]; sr = True
    if p==1: p1=2
    else: p1=1
    while s:
        i,j = s.pop()
        v[i][j]=True
        if B[i][j]==p1: continue
        if B[i][j]!=p: sr=False
        for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            if 0<=i+di<N and 0<=j+dj<N and not v[i+di][j+dj]: s.append((i+di,j+dj))
    return sr

def remove_surrounded_stones(board, player):
  positions_to_remove = [(i,j) for i in range(len(board)) for j in range(len(board)) if board[i][j]==player and sr(board,i,j,player)]
  return positions_to_remove

def Res(board, player):
  opponent = 2 if player==1 else 1
  player_stones = sum(1 for i in range(5) for j in range(5) if board[i][j]==player)
  opponent_stones = sum(1 for i in range(5) for j in range(5) if board[i][j]==opponent)
  return player_stones - opponent_stones

def Board_state(board, player):
  opponent = 2 if player == 1 else 1
  remove_list = remove_surrounded_stones(board, player)
  for x,y in remove_list:
    board[x][y] = 0
  return board

def numz(board):
  return sum(1 for i in range(5) for j in range(5) if board[i][j] == 0)

def ko(prev_board, curr_board, prop_board):
  cap_count = sum(1 for i in range(5) for j in range(5) if curr_board[i][j] != 0 and prop_board[i][j] == 0)
  if cap_count != 1:
    return False
  for i in range(5):
    for j in range(5):
      if prop_board[i][j] != prev_board[i][j]:
        return False
  return True

def get_group_liberties(x, y, b, v=None):
  if v is None: v=set()
  if (x,y) in v: return set()
  v.add((x,y)); gl=b[x][y]; l=set()
  for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
    if 0<=x+dx<len(b) and 0<=y+dy<len(b[0]):
      if b[x+dx][y+dy]==0: l.add((x+dx,y+dy)) 
      elif b[x+dx][y+dy]==gl: l|=get_group_liberties(x+dx,y+dy,b,v)
  return l

def check_captures(x, y, p, b):
  o = 3-p
  for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
    if 0<=x+dx<len(b) and 0<=y+dy<len(b[0]) and b[x+dx][y+dy]==o:
      if len(get_group_liberties(x+dx,y+dy,b))==0: return True
  return False

def subs(x, y, p, b):
  if b[x][y]!=0:return False
  t = list(map(list,b)) 
  t[x][y]=p 
  if check_captures(x,y,p,t):return True
  if len(get_group_liberties(x,y,t))==0:return False
  return True

def Resb(board, player):
    opponent = 2 if player == 1 else 1

    player_stones = sum(1 for i in range(5) for j in range(5) if board[i][j] == player)
    opponent_stones = sum(1 for i in range(5) for j in range(5) if board[i][j] == opponent)

    
    def get_neighbors(x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  
                nx, ny = x + dx, y + dy
                if 0 <= nx < 5 and 0 <= ny < 5:
                    neighbors.append((nx, ny))
        return neighbors

    bonus_points = 0
    for i in range(5):
        for j in range(5):
            if board[i][j] == player:
                for nx, ny in get_neighbors(i, j):
                    if board[nx][ny] == opponent:
                        bonus_points += -0.2
                        

    return (player_stones + bonus_points) - opponent_stones



def maximize(board, depth, player, alpha, beta, wb, previous=None):
    if depth == 0 or numz(board) == 0:
        if wb == 1:
            return -Resb(board, player), None
        else:
            return -Res(board, player), None

    best_value = -200000
    best_move = None

    possible_moves = [(i, j) for i in range(5) for j in range(5) if board[i][j] == 0]
    random.shuffle(possible_moves)

    for i, j in possible_moves:
        new_board = [row[:] for row in board]  
        new_board1 = [row[:] for row in board]
        
        if subs(i, j, player, new_board):
            new_board1[i][j] = player    
            new_board1 = Board_state(new_board1, 3-player)    
            if ko(previous, board, new_board1):
                continue

            new_board[i][j] = player    
            new_board = Board_state(new_board, 3-player)  
            move_value, _ = minimize(new_board, depth - 1, 3-player, alpha, beta, wb, board)

            if move_value > best_value:
                best_value = move_value
                best_move = (i, j)
            alpha = max(alpha, best_value)

            if beta <= alpha:
                break  # beta cut-off

    return best_value, best_move


def minimize(board, depth, player, alpha, beta, wb, previous=None):
    if depth == 0:
        if wb == 1:
            return -Resb(board, player), None
        else:
            return -Res(board, player), None

    best_value = 200000
    best_move = None

    possible_moves = [(i1, j1) for i1 in range(5) for j1 in range(5) if board[i1][j1] == 0]
    random.shuffle(possible_moves)

    for i1, j1 in possible_moves:
        new_board = [row[:] for row in board]
        new_board1 = [row[:] for row in board]
        
        if subs(i1, j1, player, new_board):
            new_board1[i1][j1] = player    
            new_board1 = Board_state(new_board1, 3-player)    
            if ko(previous, board, new_board1):
                continue

            new_board[i1][j1] = player    
            new_board = Board_state(new_board, 3-player)
            move_value, _ = maximize(new_board, depth - 1, 3-player, alpha, beta, wb, board)

            if move_value < best_value:
                best_value = move_value
                best_move = (i1, j1)
            beta = min(beta, best_value)

            if beta <= alpha:
                break  # alpha cut-off

    return best_value, best_move

    
with open('input.txt', 'r') as file:
    lines = file.read().splitlines()
    


    
player = int(lines[0])
Pr = [list(map(int, line)) for line in lines[1:6]]
C = [list(map(int, line)) for line in lines[6:11]]

counter = 0
for i in range(5):
    for j in range(5):
        if(C[i][j] == player):
            counter = counter + 1

if(counter < 6):
    best_value, optimal_move = maximize(C, 3, player, -200000, 200000, player, Pr)
elif(counter < 11):    
    best_value, optimal_move = maximize(C, 5, player, -200000, 200000, player, Pr)
else:
    best_value, optimal_move = maximize(C, 1, player, -200000, 200000, player, Pr)

with open('output.txt', 'w') as f:
  if optimal_move is None:
    f.write('PASS')
  else:  
    f.write(f'{optimal_move[0]},{optimal_move[1]}')#!/usr/bin/env python
# coding: utf-8

import math
import copy  # for deep copy

BOARD_SIZE = 5

# Players
MAXIMIZER = 1
MINIMIZER = -1

def read_input(input_file):
    """
    Read current board state from input file.

    Returns:
    - board: 2D list representing current board state
    - player: int representing which player we are
    """

    with open(input_file) as f:
        lines = f.readlines()

    player = int(lines[0])

    board = []
    for i in range(1, BOARD_SIZE+1):
        row = list(map(int, lines[i].strip()))
        board.append(row)

    return board, player

def check_win(board, player):
    # Check rows
    for row in board:
        if all(spot == player for spot in row):
            return True
    # Check columns
    for col in range(BOARD_SIZE):
        if all(board[row][col] == player for row in range(BOARD_SIZE)):
            return True
    # Check diagonals
    if all(board[i][i] == player for i in range(BOARD_SIZE)):
        return True
    if all(board[i][BOARD_SIZE - 1 - i] == player for i in range(BOARD_SIZE)):
        return True
    return False

def get_valid_moves(board):
    # Return list of valid moves
    return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]

def make_move(board, move, player):
    # Return new board with made move
    new_board = copy.deepcopy(board)
    i, j = move
    new_board[i][j] = player
    return new_board


def minimax(board, depth, maximizing, player):
  if depth == 0:
    return evaluate(board)

  if maximizing:
    max_eval = -math.inf
    for move in get_valid_moves(board):
      new_board = make_move(board, move, player)  # use 'player' argument here
      ev = minimax(new_board, depth - 1, False, MINIMIZER)  # pass MINIMIZER as next player
      max_eval = max(max_eval, ev)
    return max_eval
  else:
    min_eval = math.inf
    for move in get_valid_moves(board):
      new_board = make_move(board, move, player)  # use 'player' argument here
      ev = minimax(new_board, depth - 1, True, MAXIMIZER)  # pass MAXIMIZER as next player
      min_eval = min(min_eval, ev)
    return min_eval


def evaluate(board):
    # Check if maximizer won
    if check_win(board, MAXIMIZER):
        return 1

    # Check if minimizer won
    if check_win(board, MINIMIZER):
        return -1

    # Otherwise, calculate heuristic value
    maximizer_stones = 0
    minimizer_stones = 0

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == MAXIMIZER:
                maximizer_stones += 1
            elif board[i][j] == MINIMIZER:
                minimizer_stones += 1

    # Positive value if maximizer ahead, negative if behind
    return maximizer_stones - minimizer_stones

def minimax_move(board, player):
    best_move = None
    best_value = -math.inf if player == MAXIMIZER else math.inf
    depth = 3

    for move in get_valid_moves(board):
        new_board = make_move(board, move, player)
        value = minimax(new_board, depth, player != MAXIMIZER, player)


        if player == MAXIMIZER:
            if value > best_value:
                best_value = value
                best_move = move
        else:
            if value < best_value:
                best_value = value
                best_move = move

    return best_move

# Driver code
if __name__ == "__main__":

    # Read board state
    board, player = read_input("input.txt")

    # Calculate next move
    next_move = minimax_move(board, MAXIMIZER)

    # Print output
    print(f"{next_move[0]},{next_move[1]}")
