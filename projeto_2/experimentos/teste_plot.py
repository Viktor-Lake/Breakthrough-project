from plots import Plots

# Para ver os resultados do Q-Learning
viz_q = Plots("q_learning_grid_results.json")
viz_q.load_data()
viz_q.plot_learning_curve()
viz_q.plot_heatmap()

# Para ver os resultados do Bellman
viz_b = Plots("bellman_results.json")
viz_b.load_data()
viz_b.plot_learning_curve() # Mostrará o net_worth_history do backtest