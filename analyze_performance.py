import time
import pandas as pd
import matplotlib.pyplot as plt
import random

from minimax_agent import MinimaxAgent
from alphabeta_agent import AlphaBetaAgent
from expectimax_agent import ExpectimaxAgent
from game import GameState


class RandomAgent(MinimaxAgent):  # inherit to use metric attributes
    def get_action(self, state, depth=None):
        actions = state.get_legal_actions()
        # minimal metrics
        self.nodes_explored = 1
        self.time_taken = 0.000001
        return random.choice(actions)

def run_match(agent_class_x, agent_class_o, depth, game_id):#run 1 round
    state = GameState()
    agentX = agent_class_x()
    agentO = agent_class_o()

    match_results = []

    while not state.is_terminal():
        current_player = state.to_move
        agent = agentX if current_player == 'X' else agentO

        start_time = time.time()
        action = agent.get_action(state, depth)
        agent.time_taken = time.time() - start_time
        state = state.generate_successor(action)

        match_results.append({# record its metrics in a dictionanary
            "Game_ID": game_id,
            "Agent_Type": agent.__class__.__name__,#name of class the obj belongs to
            "Player": current_player,
            "Depth": depth,
            "Nodes_Explored": agent.nodes_explored,
            "Time_Taken": agent.time_taken,
        })

    winner = state.winner()
    for move in match_results:
        move["Outcome"] = winner

    return match_results

def run_experiments(agent_pairs, depths, num_games=10):#run more rounds with different agents with different depths
    all_results = []
    game_counter = 1

    for agent_x, agent_o in agent_pairs:
        for depth in depths:
            for _ in range(num_games):
                all_results.extend(run_match(agent_x, agent_o, depth, game_counter))
                game_counter += 1

    return pd.DataFrame(all_results)#converts list into data table


def plot_metrics(df):
    # avg metrics per agent type and depth
    summary = df.groupby(["Agent_Type", "Depth"]).agg(
        Avg_Nodes=("Nodes_Explored", "mean"),
        Avg_Time=("Time_Taken", "mean")
    ).reset_index()

    summary.to_csv("performance_metrics.csv", index=False)
    print("Metrics saved to performance_metrics.csv")

    plt.figure(figsize=(8, 5))
    for agent in summary["Agent_Type"].unique():
        subset = summary[summary["Agent_Type"] == agent]
        plt.plot(subset["Depth"], subset["Avg_Nodes"], marker="o", label=agent)
    plt.xlabel("Search Depth")
    plt.ylabel("Average Nodes Explored")
    plt.title("Nodes Explored vs Depth")
    plt.legend()
    plt.grid(True)
    plt.savefig("Nodes_Explored.png")#saves on local place

    plt.figure(figsize=(8, 5))
    for agent in summary["Agent_Type"].unique():
        subset = summary[summary["Agent_Type"] == agent]
        plt.plot(subset["Depth"], subset["Avg_Time"], marker="o", label=agent)
    plt.xlabel("Search Depth")
    plt.ylabel("Average Time per Move (s)")
    plt.title("Time per Move vs Depth")
    plt.legend()
    plt.grid(True)
    plt.savefig("Time_Per_Move.png")

    print("Plots saved: Nodes_Explored.png, Time_Per_Move.png")

def main():
    depths = [3, 5, 7, 9]
    num_games = 10

    agent_pairs = [
        (MinimaxAgent, RandomAgent),      # baseline
        (AlphaBetaAgent, RandomAgent),    # pruning efficiency
        (ExpectimaxAgent, RandomAgent),   # stochastic behavior
        (AlphaBetaAgent, MinimaxAgent),   # accuracy test
    ]

    df = run_experiments(agent_pairs, depths, num_games)
    plot_metrics(df)

if __name__ == "__main__":#this means if the file is executed diretly by directly calling it, the main function runs
    main()
