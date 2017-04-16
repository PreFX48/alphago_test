import numpy as np
from generate import prepare_board, show_board, get_rollout_result
import sys
from tqdm import *
import time


def get_result(board):  # TODO: optimize it. heavily
    for i in range(2):
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 4):
                black = True
                white = True
                for i in range(5):
                    if board[row, col+i] != 1.0:
                        white = False
                    if board[row, col+i] != -1.0:
                        black = False
                if black:
                    return -1.0
                if white:
                    return 1.0
        board = np.rot90(board)
    for i in range(2):
        for row in range(board.shape[0] - 4):
            for col in range(board.shape[1] - 4):
                black = True
                white = True
                for i in range(5):
                    if board[row+i, col+i] != 1.0:
                        white = False
                    if board[row+i, col+i] != -1.0:
                        black = False
                if black:
                    return -1.0
                if white:
                    return 1.0
        board = np.rot90(board)
    if np.abs(board).sum() == board.size:
        return 0.0
    else:
        return None


def legal_move(board, policy, extensive_features):
    is_black = (int(board.sum()) == 0)
    p_board = prepare_board(board,
                            is_black,
                            extensive_features=extensive_features)
    p_board = p_board.flatten()
    probas = policy.predict_proba(p_board[np.newaxis])[0]
    mask = 1.0 - np.abs(board).flatten()
    probas *= mask
    return np.argmax(probas)


class TreeNode:
    def __init__(self, board, is_black_to_go, sl_policy):
        self.sl_policy = sl_policy
        self.children = {}
        self.board = board.copy()
        self.values_sum = 0.0
        self.visits = 0
        self.prior_proba = 1.0
        self.last_expanded_action = -1
        self.is_black_to_go = is_black_to_go
        self.children_probas = self.sl_policy.predict(prepare_board(board, not is_black_to_go)[np.newaxis])[0]
        self.max_children = round(board.size - np.abs(board).sum())
        self.expand_times = 0

    def expand(self):
        self.expand_times += 1
        action = self.last_expanded_action + 1
        while True:
            if self.board[action // 15, action % 15] == 0.0:
                break
            else:
                action += 1
        self.last_expanded_action = action
        new_board = self.board.copy()
        new_board[action // 15, action % 15] = -1 - 2*new_board.sum()
        new_node = TreeNode(new_board, not self.is_black_to_go, self.sl_policy)
        self.children[(action // 15, action % 15)] = new_node
        return new_node

    def get_interest_value(self):
        return self.get_q_value() + self.prior_proba / (1 + self.visits)

    def get_q_value(self):
        return self.values_sum / max(1, self.visits)

    def get_best_action(self, without_prior_proba=False):
        max_action = None
        max_interest = None
        for action in self.children:
            if without_prior_proba:
                new_interest = self.children[action].get_q_value()
            else:
                new_interest = self.children[action].get_interest_value()
            if max_action is None or new_interest > max_interest:
                max_action = action
                max_interest = new_interest
        return max_action


class MonteCarloTreeSearch:
    def __init__(self, black_starts, root, sl_policy, rollout_policy, value_policy, mixing_param=0.5):
        self.node_list = []
        self.root = TreeNode(root, not black_starts, sl_policy)
        self.sl_policy = sl_policy
        self.value_policy = value_policy
        self.rollout_policy = rollout_policy
        self.mixing_param = mixing_param

    def make_move(self, iterations_count):
        for iteration in trange(iterations_count):
            node = self.select(self.root)
            # TODO: run NN for the whole dihedral group
            value = self.value_policy.predict(prepare_board(node.board,
                                                            node.is_black_to_go)[np.newaxis])[0, 0]
            reward = (1 - self.mixing_param) * value + self.mixing_param * self.rollout(node)
            self.backpropagate(reward)
            self.node_list = []
        action = self.root.get_best_action(without_prior_proba=True)
        self.root = self.root.children[action]
        return action

    def pass_move(self, action):
        if action not in self.root.children:
            board = self.root.board.copy()
            board[action] = -1.0 - 2*self.root.board.sum()
            new_node = TreeNode(board, not self.root.is_black_to_go, self.sl_policy)
            self.root.children[action] = new_node
        self.root = self.root.children[action]

    def select(self, node):
        action = (0, 0)
        while get_rollout_result(node.board, action) is None:
            self.node_list.append(node)
            if node.expand_times < node.max_children:
                new_node = node.expand()
                self.node_list.append(new_node)
                return new_node
            else:
                action = node.get_best_action()
                node = node.children[action]
        self.node_list.append(node)
        return node

    def rollout(self, node):
        board = node.board.copy()
        action = None
        while get_rollout_result(board, action) is None:
            action = legal_move(board, self.rollout_policy, extensive_features=False)
            board[action // 15, action % 15] = -1.0 - 2*board.sum()
            action = (action // 15, action % 15)
        return get_rollout_result(board, action)

    def backpropagate(self, reward):
        for node in self.node_list:
            node.visits += 1
            fixed_reward = reward if node.is_black_to_go else -reward
            node.values_sum += fixed_reward
