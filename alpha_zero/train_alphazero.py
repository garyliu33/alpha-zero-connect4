"""
Name: train_alphazero.py
Desc: Training script for AlphaZero.
"""

import numpy as np
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
import sys

from games import Connect4, Othello
from algo_components import PolicyValueNet, Buffer, generate_self_play_data, play_one_game_against_pure_mcts, get_device

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

wandb.init(
    project="alphazero",
    entity="garyliu33-amazon",
    settings=wandb.Settings(_disable_stats=True),
    name=f'{game_name}_test'
)

# @@@@@@@@@@ hyper-parameters @@@@@@@@@@

num_games_for_training = 5000  # in total, 5000 self-play games will be played
num_grad_steps = 50
eval_freq = 200  # evaluate alphazero once per 200 self-play games
eval_num_games = 10  # 10 first-hand games, 10 second-hand games
buffer_size = 20000
batch_size = 512
num_mcts_iter_alphazero = 500
num_mcts_iter_pure_mcts = 5000

# @@@@@@@@@@ important objects @@@@@@@@@@

board_width, board_height = game_klass().board.shape
policy_value_net = PolicyValueNet(board_width, board_height).float().to(get_device())
optimizer = optim.Adam(policy_value_net.parameters(), lr=1e-3, weight_decay=1e-4)  # l2 norm
buffer = Buffer(board_width, board_height, buffer_size, batch_size)

# @@@@@@@@@@ training loop @@@@@@@@@@

for game_idx in tqdm(range(num_games_for_training)):  # for each self-play game ...

    # self-play means to let the current alphazero play with itself (with exploration noise added)

    states, mcts_probs, zs = generate_self_play_data(
        game_klass=game_klass,
        num_mcts_iter=num_mcts_iter_alphazero,
        policy_value_fn=policy_value_net.policy_value_fn,
        high_temp_for_first_n=3
    )

    # add the self-play data into a buffer; the buffer also augments the self-play data using geometries
    # we do this because generating self-play data is a time-consuming process, especially when
    # our MCTS is implemented in pure Python code

    buffer.push(states, mcts_probs, zs)

    # update the policy-value network using supervised training

    if buffer.is_ready():

        for n in range(num_grad_steps):

            states_b, mcts_probs_b, zs_b = buffer.sample()
            predicted_log_probs, predicted_zs = policy_value_net(states_b)

            loss_part1 = torch.mean((zs_b - predicted_zs) ** 2)
            loss_part2 = - torch.mean(torch.sum(mcts_probs_b * predicted_log_probs, dim=1))
            loss = loss_part1 + loss_part2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("iteration = ", n, " loss = ", loss.item())

    # evaluate alphazero infrequently against a pure MCTS play
    # this can be very time-consuming, so set eval_freq to a large value and num_mcts_iter_pure_mcts
    # for a small value

    if (game_idx + 1) % eval_freq == 0:

        first_hand_scores = []
        for i in range(eval_num_games):
            score = play_one_game_against_pure_mcts(
                game_klass=game_klass,
                num_mcts_iters_pure=num_mcts_iter_pure_mcts,
                num_mcts_iters_alphazero=num_mcts_iter_alphazero,
                policy_value_fn=policy_value_net.policy_value_fn,
                first_hand="alphazero"
            )
            first_hand_scores.append(score)

        second_hand_scores = []
        for i in range(eval_num_games):
            score = play_one_game_against_pure_mcts(
                game_klass=game_klass,
                num_mcts_iters_pure=num_mcts_iter_pure_mcts,
                num_mcts_iters_alphazero=num_mcts_iter_alphazero,
                policy_value_fn=policy_value_net.policy_value_fn,
                first_hand="pure_mcts"
            )
            second_hand_scores.append(score)

        mean_first_hand_score = float(np.mean(first_hand_scores))
        mean_second_hand_score = float(np.mean(second_hand_scores))

        mean_score = (mean_first_hand_score + mean_second_hand_score) / 2

        print(f"@@@@@ Eval after {game_idx + 1}/{num_games_for_training} "
              f"games against pure-mcts {num_mcts_iter_pure_mcts} @@@@@")

        print(f"Score (first-hand): {round(mean_first_hand_score, 2)}")
        print(f"Score (second-hand): {round(mean_second_hand_score, 2)}")
        print(f"Score (overall): {round(mean_score, 2)}")

        torch.save(policy_value_net.state_dict(), f"{wandb.run.dir}/{game_name}_pvnet_{game_idx+1}.pth")
