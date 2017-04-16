from tree_search import *
from generate import *
from sklearn.externals import joblib

sl_policy = load_model('current_policy_model')
value_policy = load_model('current_value_model')
rollout_policy = joblib.load('logreg_3.pkl')
board = np.zeros((15, 15), dtype='float32')
search = MonteCarloTreeSearch(True, board, sl_policy, rollout_policy, value_policy, 0.5)
turn_number = 0
turns = 5
while get_result(board) is None:
    board = search.run(225)
    turn_number += 1
    if turn_number == turns:
        break
