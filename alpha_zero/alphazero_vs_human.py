"""
Name: alphazero_vs_human.py
Desc: AlphaZero plays against human.
"""

from ast import literal_eval
import torch
import numpy as np
import sys

from games import Connect4, Othello
from algo_components import Node, mcts_one_iter, PolicyValueNet, get_device


# Interactive Game Selection
available_games = [Connect4, Othello]
print("Select game:")
for i, game_class in enumerate(available_games):
    print(f"{i + 1}. {game_class.__name__}")

while True:
    try:
        selection = int(input("Enter choice (index): "))
        if 1 <= selection <= len(available_games):
            game_klass = available_games[selection - 1]
            break
        else:
            print("Invalid index. Please try again.")
    except ValueError:
        print("Invalid input. Please enter a number.")

game_name = game_klass.__name__.lower()
print(f"Selected game: {game_klass.__name__}")

game = game_klass()

policy_value_net = PolicyValueNet(*game.board.shape).to(get_device())

# Auto-detect model path
default_model_path = f"trained_models/{game_name}_pvnet_3000.pth"
# Check if exists? Maybe just try to load, or use specific path if provided via arg (but trying to avoid argparse as requested, though args are useful)
# Let's just try default and warn.

try:
    policy_value_net.load_state_dict(torch.load(default_model_path, map_location=get_device()))
    print(f"Loaded model from {default_model_path}")
except FileNotFoundError:
    print(f"Warning: Trained model not found at {default_model_path}. Using random weights.")

print(game)

while True:

    if game.current_player == -1:  # alphazero is always first-hand

        pi_vec, _ = policy_value_net.policy_value_fn(game.board * game.get_current_player(), game.get_valid_moves(), True)
        pi_vec[pi_vec < 0.01] = 0
        print("@@@@@ Prior move probabilities @@@@@")
        print(pi_vec.reshape(game.board.shape))

        root = Node(parent=None, prior_prob=1.0)

        for _ in range(np.random.randint(50, 1000)):  # introduce some stochasticity here
            mcts_one_iter(game, root, policy_value_fn=policy_value_net.policy_value_fn)

        move = root.get_move(temp=0)

    else:

        move = literal_eval(input("What's your move: "))
        move = (move[0] - 1, move[1] - 1)  # Input as 1-indexed (assuming consistent with previous scripts, though Othello usually is letter-number, but let's stick to tuple input for now as per Connect4)

    done, winner = game.evolve(move)
    if game.get_previous_player() == -1:
        print("@@@@@ AlphaZero just moved @@@@@")
    print(game)

    if done:
        print(game)
        print(f"Winner is player {winner}.")
        break
