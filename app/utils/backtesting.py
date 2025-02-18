import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

class Backtester:
    def __init__(self, df, initial_capital=100000, commission=0.001, slippage=0.0005):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.positions = pd.Series(0, index=df.index)
        self.portfolio = pd.DataFrame(index=df.index, columns=['cash', 'positions', 'total']).fillna(0)
        self.trades = []
    
    def test_strategy(self, signals, stop_loss=0.02, take_profit=0.05, trailing_stop=None):
        capital = self.initial_capital
        position = 0
        entry_price = 0
        self.portfolio.loc[:, 'cash'] = capital
        self.portfolio.loc[:, 'total'] = capital
        
        for i in range(1, len(self.df)):
            current_price = self.df['Close'].iloc[i]
            signal = signals.iloc[i]
            
            if position > 0:
                pnl = (current_price - entry_price) / entry_price
                
                if pnl <= -stop_loss or pnl >= take_profit:
                    capital += position * current_price
                    self.trades.append(('CLOSE', current_price, position, capital))
                    position = 0
                elif trailing_stop:
                    trailing_price = entry_price * (1 + trailing_stop)
                    if current_price <= trailing_price:
                        capital += position * current_price
                        self.trades.append(('CLOSE', current_price, position, capital))
                        position = 0
            
            if signal == 'BUY' and position == 0:
                position = capital // current_price
                entry_price = current_price * (1 + self.slippage)
                cost = position * entry_price * (1 + self.commission)
                capital -= cost
                self.trades.append(('BUY', entry_price, position, capital))
            elif signal == 'SELL' and position > 0:
                capital += position * current_price * (1 - self.commission)
                self.trades.append(('SELL', current_price, position, capital))
                position = 0
            
            self.positions.iloc[i] = position
            self.portfolio.loc[self.df.index[i], 'positions'] = position * current_price
            self.portfolio.loc[self.df.index[i], 'cash'] = capital
            self.portfolio.loc[self.df.index[i], 'total'] = capital + position * current_price
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self):
        returns = self.portfolio['total'].pct_change().dropna()
        win_trades = [t for t in self.trades if t[0] == 'SELL' and t[2] > 0]
        loss_trades = [t for t in self.trades if t[0] == 'SELL' and t[2] < 0]
        
        metrics = {
            'total_return': (self.portfolio['total'].iloc[-1] / self.initial_capital - 1) * 100,
            'annualized_return': (((self.portfolio['total'].iloc[-1] / self.initial_capital) ** (252/len(self.df))) - 1) * 100,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / (returns.std() + 1e-8),
            'sortino_ratio': np.sqrt(252) * returns.mean() / (returns[returns < 0].std() + 1e-8),
            'max_drawdown': (self.portfolio['total'] / self.portfolio['total'].cummax() - 1).min() * 100,
            'win_rate': len(win_trades) / max(1, len(win_trades) + len(loss_trades)) * 100,
            'profit_factor': sum(t[2] for t in win_trades) / (abs(sum(t[2] for t in loss_trades)) + 1e-8),
            'volatility': returns.std() * np.sqrt(252)
        }
        return metrics