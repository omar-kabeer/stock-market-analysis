import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import linregress
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import plotly.graph_objects as go

class PatternType(Enum):
    HEAD_SHOULDERS = "head_shoulders"
    INV_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASC_TRIANGLE = "ascending_triangle"
    DESC_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bullish_flag"
    BEAR_FLAG = "bearish_flag"
    RECTANGLE = "rectangle"

@dataclass
class Pattern:
    type: PatternType
    start_idx: pd.Timestamp
    end_idx: pd.Timestamp
    confidence: float
    support_price: Optional[float] = None
    resistance_price: Optional[float] = None
    target_price: Optional[float] = None

class PatternRecognition:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.patterns: Dict[PatternType, List[Pattern]] = {pattern: [] for pattern in PatternType}
        self.peak_params = {'distance': 5, 'prominence': 0.005, 'width': 3}

    def find_all_patterns(self) -> Dict[PatternType, List[Pattern]]:
        """Find all supported chart patterns in the data."""
        self._prepare_data()
        self.find_head_shoulders()
        self.find_double_patterns()
        self.find_triangles()
        return self.patterns

    def _prepare_data(self) -> None:
        """Preprocess data by detecting peaks, troughs, and key indicators."""
        high_peaks, _ = find_peaks(self.df['High'], distance=self.peak_params['distance'])
        low_peaks, _ = find_peaks(-self.df['Low'], distance=self.peak_params['distance'])
        
        self.df['High_idx'] = np.nan
        self.df['Low_idx'] = np.nan
        self.df.iloc[high_peaks, self.df.columns.get_loc('High_idx')] = self.df['High'].iloc[high_peaks]
        self.df.iloc[low_peaks, self.df.columns.get_loc('Low_idx')] = self.df['Low'].iloc[low_peaks]

        # Add RSI for momentum confirmation
        delta = self.df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(14)/14, mode='same')
        avg_loss = np.convolve(loss, np.ones(14)/14, mode='same')
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def find_head_shoulders(self, tolerance: float = 0.02) -> None:
        """Identify Head & Shoulders patterns."""
        highs = self.df.dropna(subset=['High_idx'])
        for i in range(len(highs) - 4):
            window = highs.iloc[i:i+5]
            if len(window) < 5:
                continue

            left_shoulder, head, right_shoulder = window.iloc[0]['High'], window.iloc[2]['High'], window.iloc[4]['High']
            
            if (abs(left_shoulder - right_shoulder) < tolerance * left_shoulder and 
                head > left_shoulder * (1 + tolerance) and head > right_shoulder * (1 + tolerance)):

                neckline = self._calculate_neckline(window)
                target = neckline - (head - neckline)

                self.patterns[PatternType.HEAD_SHOULDERS].append(
                    Pattern(PatternType.HEAD_SHOULDERS, window.index[0], window.index[-1], 0.8, neckline, None, target)
                )

    def find_double_patterns(self, tolerance: float = 0.02) -> None:
        """Identify Double Tops and Bottoms."""
        highs, lows = self.df.dropna(subset=['High_idx']), self.df.dropna(subset=['Low_idx'])
        
        for i in range(len(highs) - 1):
            peak1, peak2 = highs.iloc[i]['High'], highs.iloc[i + 1]['High']
            if abs(peak1 - peak2) < tolerance * peak1:
                self.patterns[PatternType.DOUBLE_TOP].append(
                    Pattern(PatternType.DOUBLE_TOP, highs.index[i], highs.index[i + 1], 0.85, None, max(peak1, peak2), min(peak1, peak2) - (max(peak1, peak2) - min(peak1, peak2)))
                )

        for i in range(len(lows) - 1):
            trough1, trough2 = lows.iloc[i]['Low'], lows.iloc[i + 1]['Low']
            if abs(trough1 - trough2) < tolerance * trough1:
                self.patterns[PatternType.DOUBLE_BOTTOM].append(
                    Pattern(PatternType.DOUBLE_BOTTOM, lows.index[i], lows.index[i + 1], 0.85, min(trough1, trough2), None, max(trough1, trough2) + (max(trough1, trough2) - min(trough1, trough2)))
                )

    def find_triangles(self, window_size: int = 20, min_slope: float = 0.05) -> None:
        """Identify Triangle patterns."""
        for i in range(len(self.df) - window_size):
            window = self.df.iloc[i:i+window_size]
            highs, lows = window['High'], window['Low']
            
            high_slope, _, _, _, _ = linregress(range(len(highs)), highs)
            low_slope, _, _, _, _ = linregress(range(len(lows)), lows)

            if abs(high_slope) < min_slope and low_slope > min_slope:
                pattern_type = PatternType.ASC_TRIANGLE
            elif abs(low_slope) < min_slope and high_slope < -min_slope:
                pattern_type = PatternType.DESC_TRIANGLE
            elif high_slope < -min_slope and low_slope > min_slope:
                pattern_type = PatternType.SYMMETRICAL_TRIANGLE
            else:
                continue

            height = highs.iloc[0] - lows.iloc[0]
            target = window['Close'].iloc[-1] + (height if low_slope > 0 else -height)

            self.patterns[pattern_type].append(
                Pattern(pattern_type, window.index[0], window.index[-1], 0.9, None, None, target)
            )

    def _calculate_neckline(self, window: pd.DataFrame) -> float:
        """Calculate neckline price."""
        lows = window.dropna(subset=['Low_idx'])['Low']
        if len(lows) < 2:
            return np.nan
        return np.mean(lows)

    def plot_patterns(self) -> go.Figure:
        """Plot detected patterns."""
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], low=self.df['Low'], close=self.df['Close'], name='Price'))
        
        for pattern_list in self.patterns.values():
            for pattern in pattern_list:
                fig.add_shape(type="rect", x0=pattern.start_idx, x1=pattern.end_idx, y0=self.df['Low'].min(), y1=self.df['High'].max(), opacity=0.2, fillcolor="gray")

        return fig
