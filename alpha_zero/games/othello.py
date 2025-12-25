import numpy as np
from games.abstract_game import Game


class Othello(Game):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((6, 6))
        self.board[2, 2] = 1  # White
        self.board[3, 3] = 1  # White
        self.board[2, 3] = -1 # Black
        self.board[3, 2] = -1 # Black
        self.current_player = -1 # Black starts

    def get_previous_player(self) -> int:
        return self.current_player * -1

    def get_current_player(self) -> int:
        return self.current_player

    def _is_on_board(self, x, y):
        return 0 <= x < 6 and 0 <= y < 6
    
    def _get_flips(self, move, player, board=None):
        if board is None:
            board = self.board
            
        r, c = move
        if board[r, c] != 0:
            return []
        
        flips = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            current_flips = []
            cur_r, cur_c = r + dr, c + dc
            while self._is_on_board(cur_r, cur_c):
                if board[cur_r, cur_c] == -player:
                    current_flips.append((cur_r, cur_c))
                elif board[cur_r, cur_c] == player:
                    if current_flips:
                        flips.extend(current_flips)
                    break
                else: # empty
                    break
                cur_r += dr
                cur_c += dc
                
        return flips

    def get_valid_moves(self, player=None) -> list:
        if player is None:
            player = self.current_player
        
        valid_moves = []
        # Optimization: only check empty squares?
        # Or cleaner to iterate all squares. 6x6 is small enough.
        for r in range(6):
            for c in range(6):
                if self.board[r, c] == 0:
                    if self._get_flips((r, c), player):
                        valid_moves.append((r, c))
        return valid_moves

    def evolve(self, move) -> tuple:
        assert move in self.get_valid_moves(), f"Move {move} is not valid for player {self.current_player}."
        
        # Apply move
        flips = self._get_flips(move, self.current_player)
        self.board[move] = self.current_player
        for r, c in flips:
            self.board[r, c] = self.current_player
            
        # Switch player
        next_player = self.current_player * -1
        
        # Check if next player has moves
        if self.get_valid_moves(next_player):
            self.current_player = next_player
            # Check game over not strictly needed here if next player can play
            # But standard checks usually done at end of loop
        else:
            # Next player has no moves, check if CURRENT player (who just moved) has moves
            # If current player also has no moves -> Game Over
            # But wait, logic is: player moves. Then we update.
            # If next player cannot move, pass turn back to current player.
            # If current player ALSO cannot move (which might happen if board full or no moves), then game over.
            
            # Actually, standard Othello:
            # If one player cannot move, play passes to other.
            # If neither can move, game ends.
            
            if self.get_valid_moves(self.current_player):
                # Next player skips, current player goes again
                # self.current_player remains same
                pass
            else:
                # Neither can move -> Game Over
                return True, self._get_winner()

        # Check if board is full (redundant if neither can move, but good for quick check)
        if not (self.board == 0).any():
             return True, self._get_winner()
        
        return False, None

    def _get_winner(self):
        score = np.sum(self.board)
        if score > 0:
            return 1 # White
        elif score < 0:
            return -1 # Black
        else:
            return 0 # Draw

    def __repr__(self):
        return f"Current Player: {self.current_player}\n" + np.array2string(self.board)
