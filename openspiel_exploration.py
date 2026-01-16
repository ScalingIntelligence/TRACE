import pyspiel

game = pyspiel.load_game("breakthrough")
print("Players:", game.num_players())
print("Actions:", game.num_distinct_actions())
print("Type:", game.get_type())
print("\nInitial state:")
state = game.new_initial_state()
print(state)
print("\nLegal actions:", state.legal_actions())
print("Action strings:", [game.action_to_string(state.current_player(), a) for a in state.legal_actions()])