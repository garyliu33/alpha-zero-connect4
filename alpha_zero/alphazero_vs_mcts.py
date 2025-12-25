"""
Name: alphazero_vs_mcts.py
Desc: AlphaZero plays against Pure MCTS.
"""

import torch
import numpy as np
import time
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

# Initialize AlphaZero Network
policy_value_net = PolicyValueNet(*game.board.shape).to(get_device())
default_model_path = f"trained_models/{game_name}_pvnet_3000.pth"
try:
    policy_value_net.load_state_dict(torch.load(default_model_path, map_location=get_device()))
    print(f"Loaded AlphaZero model from {default_model_path}")
except FileNotFoundError:
    print(f"Error: AlphaZero model not found at {default_model_path}. Exiting.")
    sys.exit(1)

# Select Start Player
print("\nWho starts first?")
print("1. AlphaZero")
print("2. Pure MCTS")
while True:
    try:
        start_choice = int(input("Enter choice: "))
        if start_choice in [1, 2]:
            break
        else:
            print("Invalid choice. Enter 1 or 2.")
    except ValueError:
        print("Invalid input. Please enter a number.")

if start_choice == 1:
    alphazero_player = -1
    mcts_player = 1
else:
    alphazero_player = 1
    mcts_player = -1

print(f"\nAlphaZero (Player {alphazero_player}) vs Pure MCTS (Player {mcts_player})")
print(game)

while True:

    if game.current_player == alphazero_player:  # AlphaZero
        print("\nAlphaZero is thinking...")
        root = Node(parent=None, prior_prob=1.0)
        
        # MCTS with network
        for _ in range(500): # 500 iterations for AlphaZero
            mcts_one_iter(game, root, policy_value_fn=policy_value_net.policy_value_fn)
            
        move = root.get_move(temp=0)
        print(f"AlphaZero plays: {move}")

    else: # Pure MCTS
        print("\nPure MCTS is thinking...")
        root = Node(parent=None, prior_prob=1.0)
        
        start = time.perf_counter()
        # MCTS without network (pure random rollouts)
        for _ in range(200000): # 200000 iterations for Pure MCTS
             mcts_one_iter(game, root, policy_value_fn=None)
        end = time.perf_counter()
        
        print(f"Decision time: {end - start:.4f}s")
        move = root.get_move(temp=0)
        print(f"Pure MCTS plays: {move}")

    done, winner = game.evolve(move)
    print(game)

    if done:
        print(game)
        if winner == alphazero_player:
             print(f"Winner is AlphaZero (Player {alphazero_player}).")
        elif winner == mcts_player:
             print(f"Winner is Pure MCTS (Player {mcts_player}).")
        else:
             print("Draw.")
        break
