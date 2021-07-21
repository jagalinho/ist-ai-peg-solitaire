# Project 1 - Group 27
# Joao Andre Galinho - 87667
# Filipe Henriques - 87653

from copy import copy
from utils import count
from search import *


# TAI content
def c_peg():
    return "O"


def c_empty():
    return "_"


def c_blocked():
    return "X"


def is_empty(e):
    return e == c_empty()


def is_peg(e):
    return e == c_peg()


def is_blocked(e):
    return e == c_blocked()


# TAI pos
# Tuple (l, c)
def make_pos(l, c):
    return (l, c)


def pos_l(pos):
    return pos[0]


def pos_c(pos):
    return pos[1]


# TAI move
# List [p_initial, p_final]
def make_move(i, f):
    return [i, f]


def move_initial(move):
    return move[0]


def move_final(move):
    return move[1]


def greedy_search(problem, h=None):
    """f(n) = h(n)"""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


# _______________________________________________________________________________________________________________


def board_moves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if is_empty(board[i][j]):
                if i > 1 and is_peg(board[i - 1][j]) and is_peg(board[i - 2][j]):
                    moves.append(make_move(make_pos(i - 2, j), make_pos(i, j)))
                if i < len(board) - 2 and is_peg(board[i + 1][j]) and is_peg(board[i + 2][j]):
                    moves.append(make_move(make_pos(i + 2, j), make_pos(i, j)))
                if j > 1 and is_peg(board[i][j - 1]) and is_peg(board[i][j - 2]):
                    moves.append(make_move(make_pos(i, j - 2), make_pos(i, j)))
                if j < len(board[i]) - 2 and is_peg(board[i][j + 1]) and is_peg(board[i][j + 2]):
                    moves.append(make_move(make_pos(i, j + 2), make_pos(i, j)))
    return moves


def board_perform_move(board, move):
    line_initial, col_initial = pos_l(move_initial(move)), pos_c(move_initial(move))
    line_final, col_final = pos_l(move_final(move)), pos_c(move_final(move))

    new_board = copy(board)
    step = -1 if line_initial > line_final else 1
    for i in range(line_initial, line_final + step, step):
        new_board[i] = copy(board[i])

    new_board[line_initial][col_initial] = c_empty()
    new_board[line_final][col_final] = c_peg()
    new_board[(line_initial + line_final) // 2][(col_initial + col_final) // 2] = c_empty()

    return new_board


def is_peg_corner(board, pos):
    line, col = pos_l(pos), pos_c(pos)
    return is_peg(board[line][col]) \
        and (line == 0 or line == len(board) - 1 or is_blocked(board[line - 1][col]) or is_blocked(board[line + 1][col])) \
        and (col == 0 or col == len(board[0]) - 1 or is_blocked(board[line][col - 1]) or is_blocked(board[line][col + 1]))


def is_peg_group_pivot(board, pos):
    line, col = pos_l(pos), pos_c(pos)
    return is_peg(board[line][col]) \
           and ((line == 0 or not is_peg(board[line - 1][col])) or (col == 0 or not is_peg(board[line][col - 1])))


def board_yield(board):
    for line in board:
        yield from line


def board_pos_yield(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            yield make_pos(i, j)


class sol_state(object):
    __slots__ = ['board', 'peg_count']

    def __init__(self, board, peg_count=None):
        self.board = board
        self.peg_count = peg_count if peg_count is not None else count(map(is_peg, board_yield(self.board)))

    def __lt__(self, other):
        return self.peg_count > other.peg_count


class solitaire(Problem):
    def __init__(self, board):
        super().__init__(sol_state(board))

    def actions(self, state):
        return board_moves(state.board)

    def result(self, state, action):
        return sol_state(board_perform_move(state.board, action), state.peg_count - 1)

    def goal_test(self, state):
        return state.peg_count == 1

    def h(self, node):
        return node.state.peg_count \
            + count(map(lambda pos: is_peg_corner(node.state.board, pos), board_pos_yield(node.state.board))) \
            + count(map(lambda pos: is_peg_group_pivot(node.state.board, pos), board_pos_yield(node.state.board))) - 2


# _______________________________________________________________________________________________________________


board_1 = solitaire(
    [["O", "O", "O", "X"],
     ["O", "O", "O", "O"],
     ["O", "_", "O", "O"],
     ["O", "O", "O", "O"]]
)

board_2 = solitaire(
    [["O", "O", "O", "X", "X"],
     ["O", "O", "O", "O", "O"],
     ["O", "_", "O", "_", "O"],
     ["O", "O", "O", "O", "O"]]
)

board_3 = solitaire(
    [["O", "O", "O", "X", "X", "X"],
     ["O", "_", "O", "O", "O", "O"],
     ["O", "O", "O", "O", "O", "O"],
     ["O", "O", "O", "O", "O", "O"]]
)

board_4 = solitaire(
    [["_", "O", "O", "O", "_"],
     ["O", "_", "O", "_", "O"],
     ["_", "O", "_", "O", "_"],
     ["O", "_", "O", "_", "_"],
     ["_", "O", "_", "_", "_"]]
)

def compare_searchers(problems, header, searchers = [breadth_first_tree_search,
                                                     breadth_first_search,
                                                     depth_first_graph_search,
                                                     iterative_deepening_search,
                                                     depth_limited_search,
                                                     recursive_best_first_search]):
    from timeit import default_timer as timer
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        start = timer()
        searcher(p)
        return p, timer() - start


    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)

print("Board 4x4")
compare_searchers([board_1], None, [depth_first_tree_search, astar_search, greedy_search])
print("\nBoard 4x5")
compare_searchers([board_2], None, [depth_first_tree_search, astar_search, greedy_search])
print("\nBoard 4x6")
compare_searchers([board_3], None, [depth_first_tree_search, astar_search, greedy_search])
print("\nBoard 5x5")
compare_searchers([board_4], None, [depth_first_tree_search, astar_search, greedy_search])
