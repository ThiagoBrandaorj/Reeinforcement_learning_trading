from gym_trading_env.renderer import Renderer

renderer = Renderer(render_logs_dir="render_logs")

# Exemplo: adicionar linhas personalizadas (opcional)
renderer.add_line(
    name="sma10",
    function=lambda df: df["close"].rolling(10).mean(),
    line_options={"color": "blue", "width": 1}
)
renderer.add_line(
    name="sma20",
    function=lambda df: df["close"].rolling(20).mean(),
    line_options={"color": "red", "width": 1}
)

# Exemplo: adicionar métricas personalizadas (opcional)
import pandas as pd
renderer.add_metric(
    name="Annual Market Return",
    function=lambda df: f"{((df['close'].iloc[-1] / df['close'].iloc[0]) ** (1 / ((df.index[-1] - df.index[0]).days / 365)) - 1) * 100:.2f}%"
)
renderer.add_metric(
    name="Annual Portfolio Return",
    function=lambda df: f"{((df['portfolio_valuation'].iloc[-1] / df['portfolio_valuation'].iloc[0]) ** (1 / ((df.index[-1] - df.index[0]).days / 365)) - 1) * 100:.2f}%"
)

# Rodar visualização web
renderer.run()