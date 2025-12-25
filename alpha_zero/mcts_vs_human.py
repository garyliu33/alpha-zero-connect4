"""
Name: mcts_vs_human.py
Desc: MCTS plays against human.
"""

from ast import literal_eval
import time

from games import Connect4, Othello
from algo_components import Node, mcts_one_iter

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

print(f"Selected game: {game_klass.__name__}")
game = game_klass()

while True:

    print(game)

    if game.current_player == -1:

        root = Node(parent=None, prior_prob=1.0)

        start = time.perf_counter()
        # For Othello 10000 might be slow or fine, keeping it same as original script for now.
        # Original had 10000.
        for _ in range(10000):
            mcts_one_iter(game, root)
        end = time.perf_counter()

        print('Decision time:', end - start)

        move = root.get_move(temp=0)

    else:

        move = literal_eval(input("What's your move: "))
        move = (move[0] - 1, move[1] - 1)

    done, winner = game.evolve(move)
    if done:
        print(game)
        print(f"Winner is player {winner}.")
        break
