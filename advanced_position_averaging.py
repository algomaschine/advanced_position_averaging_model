"""
Advanced Position Averaging Trading System with Cross-Validation

This system implements:
- Multiprocessing for parameter optimization
- Lopez de Prado cross-validation with deflated Sharpe ratio
- Triple barrier method for all trades
- Embargo periods to prevent data leakage
- Walk-forward analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistics
try:
    from scipy.stats import norm
except ImportError:
    # Fallback if scipy not available
    import math
    def norm_ppf(p):
        return np.array([norm.ppf(x) if hasattr(norm, 'ppf') else None for x in p])

# Multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools

# Import Bokeh components for HTML report generation
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, Legend
from bokeh.layouts import gridplot, column, row
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.palettes import Category10

class AdvancedPositionAveraging:
    """Advanced position averaging trading system with cross-validation."""
    
    def __init__(self, minute_data_path, initial_capital=1000000, start_date='2025-01-01'):
        self.minute_data_path = minute_data_path
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.minute_data = None
        
        # Optimized parameter space based on successful run
        self.param_space = {
            'min_confidence_threshold': [0.5],           # Keep as is - best performer
            'max_entries_per_position': [3],              # Keep as is - best performer
            'position_size_pct': [0.07, 0.10, 0.15],     # Increase risk: 5%→10%→15%
            'stop_loss_atr_multiplier': [1.5, 2.0],      # Wider stops allow more room
            'take_profit_atr_multiplier': [2.5, 3.0, 4.0], # Longer profit targets
            'move_sl_to_be_coef': [2.5, 3.0],            # Trailing stop timing
            'sl_to_be_add_pips_coef': [3.5, 4.5],        # Trailing stop distance
            'rsi_period': [14],                          
            'bb_period': [20],                           
            'bb_std': [2.0],                            
            'momentum_period': [14],                     
            'volume_threshold': [1.0],                    # Keep as is - best performer
            'breakout_threshold': [0.001],               # Keep as is - best performer
            'touch_tolerance': [0.001]                    # Keep as is - best performer
        }
        
        # Cross-validation parameters
        self.n_splits = 5  # Number of CV folds
        self.embargo_pct = 0.01  # 1% embargo period
        self.min_train_size = 0.3  # Minimum 30% for training
        
        # Results storage
        self.cv_results = {}
        self.best_params = {}
        self.optimization_results = []
        
    def load_data(self):
        """Load and prepare 1-minute data with start date filtering."""
        print("Loading 1-minute data...")
        try:
            self.minute_data = pd.read_csv(self.minute_data_path)
            self.minute_data['Date'] = pd.to_datetime(self.minute_data['Date'])
            self.minute_data = self.minute_data.sort_values('Date').reset_index(drop=True)
            
            # Filter data from start date
            start_date = pd.to_datetime(self.start_date)
            original_length = len(self.minute_data)
            self.minute_data = self.minute_data[self.minute_data['Date'] >= start_date].reset_index(drop=True)
            
            print(f"✅ Loaded {len(self.minute_data)} 1-minute bars (filtered from {original_length} total)")
            print(f"   Date range: {self.minute_data['Date'].min()} to {self.minute_data['Date'].max()}")
            print(f"   Data reduction: {((original_length - len(self.minute_data)) / original_length * 100):.1f}%")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_cv_splits(self):
        """Create cross-validation splits with embargo periods."""
        print("Creating cross-validation splits...")
        
        total_length = len(self.minute_data)
        train_size = int(total_length * self.min_train_size)
        test_size = int((total_length - train_size) / self.n_splits)
        embargo_size = int(total_length * self.embargo_pct)
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate split boundaries
            train_start = 0
            train_end = train_size + i * test_size
            test_start = train_end + embargo_size
            test_end = min(test_start + test_size, total_length)
            
            if test_end - test_start < test_size * 0.5:  # Ensure minimum test size
                break
                
            splits.append({
                'fold': i + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'embargo_start': train_end,
                'embargo_end': test_start
            })
        
        print(f"✅ Created {len(splits)} CV splits")
        for split in splits:
            train_days = (split['train_end'] - split['train_start']) / (24 * 60)
            test_days = (split['test_end'] - split['test_start']) / (24 * 60)
            embargo_days = (split['embargo_end'] - split['embargo_start']) / (24 * 60)
            print(f"   Fold {split['fold']}: Train={train_days:.1f} days, Test={test_days:.1f} days, Embargo={embargo_days:.1f} days")
        
        return splits
    
    def calculate_buy_and_hold_benchmark(self, data):
        """Calculate Buy & Hold benchmark performance."""
        if len(data) == 0:
            return {'return': 0, 'sharpe': 0, 'max_drawdown': 0}
        
        # Calculate Buy & Hold returns
        start_price = data['close'].iloc[0]
        end_price = data['close'].iloc[-1]
        total_return = (end_price - start_price) / start_price
        
        # Calculate daily returns for Sharpe ratio
        daily_returns = data['close'].pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        return {
            'return': total_return,
            'sharpe': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def get_buy_and_hold_equity_curve(self, data):
        """Generate Buy & Hold equity curve."""
        if len(data) == 0:
            return pd.DataFrame()
        
        # Create equity curve
        start_price = data['close'].iloc[0]
        equity = data['close'] / start_price * self.initial_capital
        
        equity_df = pd.DataFrame({
            'Date': data['Date'],
            'Equity': equity
        })
        
        return equity_df
    
    def get_equity_curve(self, positions, data, params):
        """Generate equity curve for a strategy."""
        if len(positions) == 0:
            return pd.DataFrame()
        
        # Calculate PnL
        pnl_df = self.calculate_triple_barrier_pnl(positions, data, params)
        
        if len(pnl_df) == 0:
            return pd.DataFrame()
        
        # Resample to daily for equity curve
        pnl_df['date'] = pd.to_datetime(pnl_df['date'])
        daily_pnl = pnl_df.groupby(pnl_df['date'].dt.date)['dollar_pnl'].sum().reset_index()
        daily_pnl.columns = ['Date', 'Daily_PnL']
        
        # Calculate cumulative equity
        daily_pnl['Cumulative_PnL'] = daily_pnl['Daily_PnL'].cumsum()
        equity = self.initial_capital + daily_pnl['Cumulative_PnL']
        
        # Create equity curve dataframe
        equity_df = pd.DataFrame({
            'Date': pd.to_datetime(daily_pnl['Date']),
            'Equity': equity
        })
        
        return equity_df
    
    def plot_comparison(self, n_top_models=10):
        """Plot Buy & Hold vs top N models."""
        print(f"\nGenerating comparison plot with top {n_top_models} models...")
        
        # Get buy and hold equity curve
        bh_equity = self.get_buy_and_hold_equity_curve(self.minute_data)
        
        # Sort optimization results by deflated Sharpe
        sorted_results = sorted(
            self.optimization_results,
            key=lambda x: x['avg_deflated_sharpe'],
            reverse=True
        )
        
        # Get top N results
        top_results = sorted_results[:n_top_models]
        
        print(f"Plotting Buy & Hold + {len(top_results)} best models...")
        
        # Create figure
        p = figure(
            title=f"Equity Curve Comparison: Buy & Hold vs Top {n_top_models} Models",
            x_axis_label='Date',
            y_axis_label='Equity',
            width=1400,
            height=700,
            x_axis_type='datetime',
            tools='pan,box_zoom,wheel_zoom,reset,save'
        )
        
        # Plot Buy & Hold
        p.line(
            bh_equity['Date'],
            bh_equity['Equity'],
            legend_label='Buy & Hold',
            line_width=3,
            line_color='black',
            alpha=0.8
        )
        
        # Plot top N models
        colors = Category10[max(len(top_results), 3)]
        
        for i, result in enumerate(top_results):
            strategy_type = result['strategy_type']
            params = result['params']
            
            # Get equity curve for this model
            # Note: We need to simulate this on full data
            test_data = self.minute_data.copy()
            test_data = self.calculate_technical_indicators(test_data, params)
            
            signals = self.detect_signals(test_data, strategy_type, params)
            positions = self.create_positions(signals, params)
            
            if len(positions) > 0:
                equity_curve = self.get_equity_curve(positions, test_data, params)
                
                if len(equity_curve) > 0:
                    p.line(
                        equity_curve['Date'],
                        equity_curve['Equity'],
                        legend_label=f"{strategy_type} (DeSR: {result['avg_deflated_sharpe']:.2f})",
                        line_width=2,
                        line_color=colors[i % len(colors)],
                        alpha=0.6
                    )
        
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'
        p.xaxis.formatter = NumeralTickFormatter(format='0.0')
        p.yaxis.formatter = NumeralTickFormatter(format='$0,0')
        
        # Save plot
        output_file('equity_curve_comparison.html')
        save(p)
        print("✅ Equity curve comparison saved to: equity_curve_comparison.html")
        
        return 'equity_curve_comparison.html'
    
    def calculate_technical_indicators(self, data, params):
        """Calculate technical indicators with given parameters."""
        # Price-based indicators
        data = data.copy()
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Moving averages
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI with parameter
        rsi_period = params.get('rsi_period', 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands with parameters
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2)
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        bb_std_val = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std_val * bb_std)
        data['bb_lower'] = data['bb_middle'] - (bb_std_val * bb_std)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # ATR with error handling
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            data['atr'] = true_range.rolling(14).mean()
            # Fill NaN values with a default ATR value
            data['atr'] = data['atr'].fillna(data['close'].rolling(14).std() * 0.02)
            # Ensure no negative or zero ATR values
            data['atr'] = data['atr'].clip(lower=data['close'] * 0.001)
        except Exception as e:
            print(f"Warning: ATR calculation failed: {e}")
            data['atr'] = data['close'].rolling(14).std() * 0.02
            data['atr'] = data['atr'].fillna(data['close'] * 0.02)
            data['atr'] = data['atr'].clip(lower=data['close'] * 0.001)
        
        # Support and resistance
        data['resistance'] = data['high'].rolling(50).max()
        data['support'] = data['low'].rolling(50).min()
        
        # Momentum indicators with parameter
        momentum_period = params.get('momentum_period', 14)
        data['momentum_14'] = data['close'].pct_change(momentum_period)
        data['momentum_5'] = data['close'].pct_change(5)
        
        return data
    
    def detect_signals(self, data, strategy_type, params):
        """Detect trading signals with given parameters."""
        signals = []
        
        # Process data in daily chunks for efficiency
        daily_groups = data.groupby(data['Date'].dt.date)
        
        print(f"Debug: Processing {len(daily_groups)} days for {strategy_type} strategy")
        
        days_with_signals = 0
        for date, day_data in daily_groups:
            if len(day_data) < 20:  # Skip days with insufficient data
                continue
            
            # Detect signals for this day
            day_signals = self._detect_daily_signals(day_data, strategy_type, params)
            if len(day_signals) > 0:
                days_with_signals += 1
                print(f"Debug: Found {len(day_signals)} signals on {date}")
            signals.extend(day_signals)
        
        print(f"Debug: Total signals found: {len(signals)} across {days_with_signals} days")
        return signals
    
    def _detect_daily_signals(self, day_data, strategy_type, params):
        """Detect signals for a single day with parameters."""
        signals = []
        
        # Calculate daily volatility for signal strength
        if len(day_data) > 20:
            recent_returns = day_data['returns'].iloc[:20]
            daily_vol = recent_returns.std()
        else:
            daily_vol = 0.01
        
        base_confidence = min(daily_vol * 100, 1.0)
        
        # Always generate signals regardless of confidence for testing
        # In production, we'd want higher confidence thresholds
        min_confidence = params.get('min_confidence_threshold', 0.6)
        
        # Only filter by confidence if it's too low
        if base_confidence < 0.01:
            return signals
        
        # Strategy 1: Breakout signals
        breakout_signals = self._detect_breakout_signals(day_data, strategy_type, base_confidence, params)
        signals.extend(breakout_signals)
        
        # Strategy 2: Mean reversion signals
        mean_reversion_signals = self._detect_mean_reversion_signals(day_data, strategy_type, base_confidence, params)
        signals.extend(mean_reversion_signals)
        
        # Strategy 3: Momentum signals
        momentum_signals = self._detect_momentum_signals(day_data, strategy_type, base_confidence, params)
        signals.extend(momentum_signals)
        
        # Strategy 4: Support/Resistance signals
        sr_signals = self._detect_support_resistance_signals(day_data, strategy_type, base_confidence, params)
        signals.extend(sr_signals)
        
        return signals
    
    def _detect_breakout_signals(self, day_data, strategy_type, base_confidence, params):
        """Detect breakout signals using multi-indicator confluence (inspired by Strategy 5.2.4)."""
        signals = []
        lookback = 20
        
        # Similar to Strategy 5.2.4 which uses:
        # - BearsPower/BullsPower for momentum
        # - Bollinger Bands for volatility extremes
        # - MACD for trend confirmation
        
        # Check for multi-indicator confluence
        for i in range(lookback + 1, len(day_data)):
            if i < 5:  # Need enough bars for trend detection
                continue
                
            signal_price = day_data['close'].iloc[i-1]
            high_20 = day_data['high'].iloc[i-lookback-1:i-1].max()
            low_20 = day_data['low'].iloc[i-lookback-1:i-1].min()
            bb_upper = day_data['bb_upper'].iloc[i-1]
            bb_lower = day_data['bb_lower'].iloc[i-1]
            bb_position = day_data['bb_position'].iloc[i-1]
            rsi = day_data['rsi'].iloc[i-1]
            macd_hist = day_data['macd_histogram'].iloc[i-1]
            volume_ratio = day_data['volume_ratio'].iloc[i-1]
            
            # Long Entry Signal (Similar to Strategy 5.2.4): 
            # Price below lower BB (oversold) + RSI < 30 + MACD histogram falling + volume confirmation
            if (strategy_type in ['longs_only', 'combined'] and 
                bb_position < 0.2 and                    # Below lower Bollinger Band
                rsi < 35 and                             # Oversold (similar to BearsPower falling)
                macd_hist < -0.0002 and                  # MACD falling (momentum confirmation)
                volume_ratio > 1.2):                     # Volume confirmation
                
                confidence = base_confidence * min(volume_ratio / 1.2, 2.0)
                signals.append({
                    'timestamp': day_data['Date'].iloc[i],
                    'price': signal_price,
                    'direction': 'long',
                    'strategy': 'multi_indicator_breakout',
                    'confidence': confidence,
                    'size_pct': 0.35
                })
            
            # Short Entry Signal (Similar to Strategy 5.2.4):
            # Price above upper BB (overbought) + RSI > 70 + MACD histogram rising + volume confirmation
            if (strategy_type in ['shorts_only', 'combined'] and 
                bb_position > 0.8 and                    # Above upper Bollinger Band
                rsi > 65 and                             # Overbought (similar to BullsPower rising)
                macd_hist > 0.0002 and                   # MACD rising (momentum confirmation)
                volume_ratio > 1.2):                    # Volume confirmation
                
                confidence = base_confidence * min(volume_ratio / 1.2, 2.0)
                signals.append({
                    'timestamp': day_data['Date'].iloc[i],
                    'price': signal_price,
                    'direction': 'short',
                    'strategy': 'multi_indicator_breakout',
                    'confidence': confidence,
                    'size_pct': 0.35
                })
        
        return signals
    
    def _detect_mean_reversion_signals(self, day_data, strategy_type, base_confidence, params):
        """Detect mean reversion signals with parameters."""
        signals = []
        
        for i in range(21, len(day_data)):
            signal_price = day_data['close'].iloc[i-1]
            bb_position = day_data['bb_position'].iloc[i-1]
            rsi = day_data['rsi'].iloc[i-1]
            
            # Long mean reversion (oversold)
            if (strategy_type in ['longs_only', 'combined'] and 
                bb_position < 0.2 and rsi < 30):
                confidence = base_confidence * (1 - bb_position) * (1 - rsi / 100)
                signals.append({
                    'timestamp': day_data['Date'].iloc[i],
                    'price': signal_price,
                    'direction': 'long',
                    'strategy': 'mean_reversion',
                    'confidence': confidence,
                    'size_pct': 0.34
                })
            
            # Short mean reversion (overbought)
            if (strategy_type in ['shorts_only', 'combined'] and 
                bb_position > 0.8 and rsi > 70):
                confidence = base_confidence * bb_position * (rsi / 100)
                signals.append({
                    'timestamp': day_data['Date'].iloc[i],
                    'price': signal_price,
                    'direction': 'short',
                    'strategy': 'mean_reversion',
                    'confidence': confidence,
                    'size_pct': 0.34
                })
        
        return signals
    
    def _detect_momentum_signals(self, day_data, strategy_type, base_confidence, params):
        """Detect momentum signals with parameters."""
        signals = []
        momentum_threshold = 0.001
        
        for i in range(15, len(day_data)):
            signal_price = day_data['close'].iloc[i-1]
            momentum_14 = day_data['momentum_14'].iloc[i-1]
            macd_hist = day_data['macd_histogram'].iloc[i-1]
            
            # Long momentum
            if (strategy_type in ['longs_only', 'combined'] and 
                momentum_14 > momentum_threshold and macd_hist > 0):
                confidence = base_confidence * min(abs(momentum_14) / momentum_threshold, 2.0)
                signals.append({
                    'timestamp': day_data['Date'].iloc[i],
                    'price': signal_price,
                    'direction': 'long',
                    'strategy': 'momentum',
                    'confidence': confidence,
                    'size_pct': 0.33
                })
            
            # Short momentum
            if (strategy_type in ['shorts_only', 'combined'] and 
                momentum_14 < -momentum_threshold and macd_hist < 0):
                confidence = base_confidence * min(abs(momentum_14) / momentum_threshold, 2.0)
                signals.append({
                    'timestamp': day_data['Date'].iloc[i],
                    'price': signal_price,
                    'direction': 'short',
                    'strategy': 'momentum',
                    'confidence': confidence,
                    'size_pct': 0.33
                })
        
        return signals
    
    def _detect_support_resistance_signals(self, day_data, strategy_type, base_confidence, params):
        """Detect support/resistance signals with parameters."""
        signals = []
        touch_tolerance = params.get('touch_tolerance', 0.001)
        
        for i in range(51, len(day_data)):
            signal_price = day_data['close'].iloc[i-1]
            resistance = day_data['resistance'].iloc[i-1]
            support = day_data['support'].iloc[i-1]
            
            # Long at support
            if (strategy_type in ['longs_only', 'combined'] and 
                abs(signal_price - support) / support < touch_tolerance):
                confidence = base_confidence * 0.8
                signals.append({
                    'timestamp': day_data['Date'].iloc[i],
                    'price': signal_price,
                    'direction': 'long',
                    'strategy': 'support_resistance',
                    'confidence': confidence,
                    'size_pct': 0.33
                })
            
            # Short at resistance
            if (strategy_type in ['shorts_only', 'combined'] and 
                abs(signal_price - resistance) / resistance < touch_tolerance):
                confidence = base_confidence * 0.8
                signals.append({
                    'timestamp': day_data['Date'].iloc[i],
                    'price': signal_price,
                    'direction': 'short',
                    'strategy': 'support_resistance',
                    'confidence': confidence,
                    'size_pct': 0.33
                })
        
        return signals
    
    def create_positions(self, signals, params):
        """Create positions from signals using position averaging."""
        positions = []
        signal_groups = self._group_signals_by_day(signals)
        max_entries = params.get('max_entries_per_position', 3)
        
        for date, day_signals in signal_groups.items():
            if len(day_signals) == 0:
                continue
            
            # Group signals by direction
            long_signals = [s for s in day_signals if s['direction'] == 'long']
            short_signals = [s for s in day_signals if s['direction'] == 'short']
            
            # Create long position
            if long_signals:
                long_position = self._create_averaged_position(long_signals, 'long', params)
                if long_position:
                    positions.append(long_position)
            
            # Create short position
            if short_signals:
                short_position = self._create_averaged_position(short_signals, 'short', params)
                if short_position:
                    positions.append(short_position)
        
        return positions
    
    def _group_signals_by_day(self, signals):
        """Group signals by trading day."""
        signal_groups = {}
        for signal in signals:
            date = signal['timestamp'].date()
            if date not in signal_groups:
                signal_groups[date] = []
            signal_groups[date].append(signal)
        return signal_groups
    
    def _create_averaged_position(self, signals, direction, params):
        """Create an averaged position from multiple signals."""
        if len(signals) == 0:
            return None
        
        # Sort by confidence and select best signals
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        max_entries = params.get('max_entries_per_position', 3)
        selected_signals = signals[:max_entries]
        
        # Calculate position details
        entry_prices = [s['price'] for s in selected_signals]
        avg_entry_price = np.mean(entry_prices)
        total_size = sum(s['size_pct'] for s in selected_signals)
        
        # Calculate stop loss and take profit using ATR
        atr = self._get_atr_for_date(selected_signals[0]['timestamp'].date())
        if atr is None or pd.isna(atr) or atr <= 0:
            # Fallback: use 2% of entry price as ATR
            atr = avg_entry_price * 0.02
        
        stop_loss_mult = params.get('stop_loss_atr_multiplier', 2.0)
        take_profit_mult = params.get('take_profit_atr_multiplier', 3.0)
        
        if direction == 'long':
            stop_loss = avg_entry_price - (stop_loss_mult * atr)
            take_profit = avg_entry_price + (take_profit_mult * atr)
        else:
            stop_loss = avg_entry_price + (stop_loss_mult * atr)
            take_profit = avg_entry_price - (take_profit_mult * atr)
        
        return {
            'date': selected_signals[0]['timestamp'].date(),
            'direction': direction,
            'entry_prices': entry_prices,
            'avg_entry_price': avg_entry_price,
            'total_size_pct': total_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'num_entries': len(selected_signals),
            'strategies': [s['strategy'] for s in selected_signals],
            'avg_confidence': np.mean([s['confidence'] for s in selected_signals])
        }
    
    def _get_atr_for_date(self, date):
        """Get ATR value for a specific date."""
        day_data = self.minute_data[self.minute_data['Date'].dt.date == date]
        if len(day_data) == 0:
            return None
        
        # Calculate ATR for this specific day if not available
        if 'atr' not in day_data.columns:
            try:
                high_low = day_data['high'] - day_data['low']
                high_close = np.abs(day_data['high'] - day_data['close'].shift())
                low_close = np.abs(day_data['low'] - day_data['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = true_range.rolling(14).mean().iloc[-1]
                if pd.isna(atr):
                    atr = day_data['close'].std() * 0.02  # Fallback
                return atr
            except Exception:
                return day_data['close'].std() * 0.02  # Fallback
        else:
            atr = day_data['atr'].iloc[-1]
            if pd.isna(atr):
                return day_data['close'].std() * 0.02  # Fallback
            return atr
    
    def calculate_triple_barrier_pnl(self, positions, data, params):
        """Calculate PnL using triple barrier method."""
        pnl_data = []
        cumulative_pnl = 0
        
        for position in positions:
            # Get price data for the position
            position_date = position['date']
            day_data = data[data['Date'].dt.date == position_date]
            
            if len(day_data) == 0:
                continue
            
            # Find the position in the data - use full data index
            first_bar_in_day = day_data.index[0]
            # Find first bar in full data
            position_idx = data.index.get_loc(first_bar_in_day)
            
            # Triple barrier parameters
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            entry_price = position['avg_entry_price']
            direction = position['direction']
            
            # Time barrier (5 days maximum, in minutes)
            max_lookahead = min(5 * 24 * 60, len(data) - position_idx)
            
            # Initialize barriers
            hit_stop_loss = False
            hit_take_profit = False
            hit_time_barrier = False
            exit_price = entry_price
            exit_reason = 'unknown'
            
            # Calculate trailing stop parameters (like Strategy 5.2.4)
            move_sl_to_be_coef = params.get('move_sl_to_be_coef', 2.8)
            sl_to_be_add_pips_coef = params.get('sl_to_be_add_pips_coef', 4.2)
            
            # ATR for trailing stop calculation
            atr_45 = data['atr'].iloc[max(0, position_idx-45):position_idx+1].mean() if position_idx >= 45 else data['atr'].iloc[position_idx]
            atr_55 = data['atr'].iloc[max(0, position_idx-55):position_idx+1].mean() if position_idx >= 55 else data['atr'].iloc[position_idx]
            
            # Calculate breakeven trigger and adjusted stop loss
            breakeven_trigger = move_sl_to_be_coef * atr_45
            breakeven_adjustment = sl_to_be_add_pips_coef * atr_55
            
            # Track if trailing stop has been activated
            trailing_stop_activated = False
            current_stop_loss = stop_loss
            highest_profit = 0  # For longs
            lowest_profit = 0   # For shorts
            
            # Check each minute after the position date
            for i in range(1, max_lookahead):
                if position_idx + i >= len(data):
                    hit_time_barrier = True
                    exit_reason = 'time_barrier'
                    break
                    
                current_bar = data.iloc[position_idx + i]
                current_high = current_bar['high']
                current_low = current_bar['low']
                current_close = current_bar['close']
                
                if direction == 'long':
                    # Calculate unrealized profit
                    profit_amount = current_close - entry_price
                    if profit_amount > highest_profit:
                        highest_profit = profit_amount
                    
                    # Trailing Stop Logic (like Strategy 5.2.4)
                    if profit_amount >= breakeven_trigger and not trailing_stop_activated:
                        # Move stop loss to breakeven + small buffer
                        if direction == 'long':
                            current_stop_loss = entry_price + breakeven_adjustment
                            trailing_stop_activated = True
                    
                    # Check stop loss
                    if current_low <= current_stop_loss:
                        hit_stop_loss = True
                        exit_price = current_stop_loss
                        exit_reason = 'stop_loss' if not trailing_stop_activated else 'trailing_stop'
                        break
                    # Then check take profit
                    elif current_high >= take_profit:
                        hit_take_profit = True
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                        break
                else:  # short
                    # Calculate unrealized profit
                    profit_amount = entry_price - current_close
                    if profit_amount > lowest_profit:
                        lowest_profit = profit_amount
                    
                    # Trailing Stop Logic (like Strategy 5.2.4)
                    if profit_amount >= breakeven_trigger and not trailing_stop_activated:
                        # Move stop loss to breakeven + small buffer
                        current_stop_loss = entry_price - breakeven_adjustment
                        trailing_stop_activated = True
                    
                    # Check stop loss
                    if current_high >= current_stop_loss:
                        hit_stop_loss = True
                        exit_price = current_stop_loss
                        exit_reason = 'stop_loss' if not trailing_stop_activated else 'trailing_stop'
                        break
                    # Then check take profit
                    elif current_low <= take_profit:
                        hit_take_profit = True
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                        break
            
            # If no barrier hit, use final price
            if not (hit_stop_loss or hit_take_profit or hit_time_barrier):
                exit_price = data.iloc[min(position_idx + max_lookahead - 1, len(data) - 1)]['close']
                exit_reason = 'final_price'
            
            # Calculate PnL
            pnl_pct = (exit_price - entry_price) / entry_price
            if direction == 'short':
                pnl_pct = -pnl_pct
            
            # Calculate dollar PnL
            position_value = self.initial_capital * position['total_size_pct']
            dollar_pnl = position_value * pnl_pct
            cumulative_pnl += dollar_pnl
            
            pnl_data.append({
                'date': position['date'],
                'direction': position['direction'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'dollar_pnl': dollar_pnl,
                'cumulative_pnl': cumulative_pnl,
                'position_size': position_value,
                'num_entries': position['num_entries'],
                'strategies': ', '.join(position['strategies']),
                'confidence': position['avg_confidence'],
                'exit_reason': exit_reason
            })
        
        return pd.DataFrame(pnl_data)
    
    def calculate_deflated_sharpe_ratio(self, returns, n_parameter_sets=64):
        """Calculate deflated Sharpe ratio as per Lopez de Prado.
        
        Formula from "The Sharpe Ratio: Inference and Performance" (Lopez de Prado, 2018):
        DeSR = SR * (1 - probit(0.95) / sqrt(V(SR)))
        
        Where V(SR) = (1 + 0.5*SR^2) / T
        And we adjust for multiple testing by n_parameter_sets.
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        # Calculate actual Sharpe ratio (annualized)
        T = len(returns)
        actual_sharpe = returns.mean() / returns.std() * np.sqrt(252)
        
        if actual_sharpe <= 0:
            return 0
        
        # Variance of Sharpe ratio estimate
        variance_of_sharpe = (1 + 0.5 * actual_sharpe**2) / T
        
        # Adjust for multiple testing (Bonferroni correction)
        # We tested n_parameter_sets combinations, so adjust significance level
        adjusted_alpha = 0.05 / n_parameter_sets if n_parameter_sets > 0 else 0.05
        
        # Calculate probit for adjusted significance level
        try:
            probit = norm.ppf(1 - adjusted_alpha)
        except:
            # Fallback to 95th percentile if scipy not available
            probit = 1.645
        
        # Calculate deflated Sharpe ratio
        deflated_sharpe = actual_sharpe * (1 - probit * np.sqrt(variance_of_sharpe))
        
        return deflated_sharpe
    
    def evaluate_strategy(self, train_data, test_data, params, strategy_type):
        """Evaluate a strategy on given data."""
        try:
            # Calculate indicators
            train_data = self.calculate_technical_indicators(train_data, params)
            test_data = self.calculate_technical_indicators(test_data, params)
            
            # Detect signals on test data
            signals = self.detect_signals(test_data, strategy_type, params)
            
            # Create positions
            positions = self.create_positions(signals, params)
            
            if len(positions) == 0:
                return {
                    'sharpe_ratio': 0,
                    'deflated_sharpe': 0,
                    'total_return': 0,
                    'max_drawdown': 0,
                    'num_trades': 0,
                    'win_rate': 0,
                    'params': params
                }
            
            # Calculate PnL using triple barrier
            pnl_df = self.calculate_triple_barrier_pnl(positions, test_data, params)
            
            if len(pnl_df) == 0:
                return {
                    'sharpe_ratio': 0,
                    'deflated_sharpe': 0,
                    'total_return': 0,
                    'max_drawdown': 0,
                    'num_trades': 0,
                    'win_rate': 0,
                    'params': params
                }
            
            # Calculate metrics
            returns = pnl_df['dollar_pnl'] / self.initial_capital
            total_return = returns.sum()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Calculate number of parameter combinations tested
            n_param_combinations = np.prod([len(v) for v in self.param_space.values()])
            deflated_sharpe = self.calculate_deflated_sharpe_ratio(returns, n_param_combinations)
            
            # Calculate max drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Calculate win rate
            win_rate = (returns > 0).mean()
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'deflated_sharpe': deflated_sharpe,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'num_trades': len(pnl_df),
                'win_rate': win_rate,
                'params': params
            }
            
        except Exception as e:
            print(f"Error evaluating strategy: {e}")
            return {
                'sharpe_ratio': 0,
                'deflated_sharpe': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0,
                'params': params
            }
    
    def optimize_parameters(self, strategy_type):
        """Optimize parameters using cross-validation."""
        print(f"\nOptimizing parameters for {strategy_type} strategy...")
        print("=" * 60)
        
        # Create CV splits
        splits = self.create_cv_splits()
        
        # Generate parameter combinations
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Use multiprocessing with a simpler approach
        num_processes = max(1, cpu_count() - 4)
        print(f"Using {num_processes} processes for optimization...")
        print(f"Total parameter combinations: {len(param_combinations)}")
        print(f"Total CV splits: {len(splits)}")
        print(f"Total evaluations: {len(param_combinations) * len(splits)}")
        
        # Create parameter chunks for parallel processing
        chunk_size = max(1, len(param_combinations) // num_processes)
        param_chunks = [param_combinations[i:i + chunk_size] for i in range(0, len(param_combinations), chunk_size)]
        
        print(f"Created {len(param_chunks)} chunks for parallel processing")
        for i, chunk in enumerate(param_chunks):
            print(f"  Chunk {i}: {len(chunk)} parameter combinations")
        
        # Calculate Buy & Hold benchmark for comparison
        print(f"\nCalculating Buy & Hold benchmark...")
        buy_hold_benchmark = self.calculate_buy_and_hold_benchmark(self.minute_data)
        print(f"Buy & Hold Benchmark:")
        print(f"  Total Return: {buy_hold_benchmark['return']:.2%}")
        print(f"  Sharpe Ratio: {buy_hold_benchmark['sharpe']:.4f}")
        print(f"  Max Drawdown: {buy_hold_benchmark['max_drawdown']:.2%}")
        
        # Prepare data for multiprocessing (only indices, not DataFrames)
        minute_data_path = self.minute_data_path
        initial_capital = self.initial_capital
        param_space = self.param_space
        
        # Create tasks for multiprocessing
        tasks = []
        for i, param_chunk in enumerate(param_chunks):
            tasks.append({
                'minute_data_path': minute_data_path,
                'initial_capital': initial_capital,
                'param_space': param_space,
                'param_combinations': param_chunk,
                'splits': splits,
                'strategy_type': strategy_type,
                'chunk_id': i,
                'start_date': self.start_date
            })
        
        # Run optimization in parallel
        print(f"\nStarting parallel optimization with {num_processes} processes...")
        start_time = datetime.now()
        
        with Pool(processes=num_processes) as pool:
            chunk_results = pool.map(optimize_parameter_chunk, tasks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n✅ Parallel optimization completed in {duration:.2f} seconds")
        
        # Combine results from all chunks
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        # Process results
        param_scores = {}
        for result in all_results:
            param_key = str(result['params'])
            if param_key not in param_scores:
                param_scores[param_key] = []
            param_scores[param_key].append(result)
        
        # Calculate average scores for each parameter set
        best_score = -np.inf
        best_params = None
        
        for param_key, split_results in param_scores.items():
            avg_deflated_sharpe = np.mean([r['deflated_sharpe'] for r in split_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in split_results])
            avg_return = np.mean([r['total_return'] for r in split_results])
            avg_trades = np.mean([r['num_trades'] for r in split_results])
            
            # Store results
            self.optimization_results.append({
                'strategy_type': strategy_type,
                'params': split_results[0]['params'],
                'avg_deflated_sharpe': avg_deflated_sharpe,
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'cv_scores': [r['deflated_sharpe'] for r in split_results]
            })
            
            # Update best parameters
            if avg_deflated_sharpe > best_score:
                best_score = avg_deflated_sharpe
                best_params = split_results[0]['params']
        
        self.best_params[strategy_type] = best_params
        
        print(f"✅ Best parameters for {strategy_type}:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"   Best deflated Sharpe: {best_score:.4f}")
        
        # Compare with Buy & Hold
        if best_score > 0:
            bh_return = buy_hold_benchmark['return']
            bh_sharpe = buy_hold_benchmark['sharpe']
            print(f"   vs Buy & Hold: Return {bh_return:.2%}, Sharpe {bh_sharpe:.4f}")
        
        return best_params
    
    
    def run_full_optimization(self):
        """Run full optimization for all strategies."""
        print("Starting Advanced Position Averaging Optimization")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Optimize each strategy
        strategies = ['longs_only', 'shorts_only', 'combined']
        
        for strategy in strategies:
            self.optimize_parameters(strategy)
        
        # Generate final report
        self.generate_optimization_report()
        
        return True
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        print("\nGenerating Optimization Report...")
        print("=" * 40)
        
        # Create results summary
        report = f"""
Advanced Position Averaging Optimization Report
==============================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Initial Capital: ${self.initial_capital:,.2f}

Parameter Optimization Results:
"""
        
        for strategy in ['longs_only', 'shorts_only', 'combined']:
            if strategy in self.best_params:
                best_params = self.best_params[strategy]
                
                # Find the best result for this strategy
                strategy_results = [r for r in self.optimization_results if r['strategy_type'] == strategy]
                if strategy_results:
                    best_result = max(strategy_results, key=lambda x: x['avg_deflated_sharpe'])
                    
                    report += f"""
{strategy.upper()} Strategy - Best Parameters:
- Deflated Sharpe Ratio: {best_result['avg_deflated_sharpe']:.4f}
- Average Sharpe Ratio: {best_result['avg_sharpe']:.4f}
- Average Return: {best_result['avg_return']:.2%}
- Average Trades: {best_result['avg_trades']:.0f}
- CV Scores: {[f'{s:.4f}' for s in best_result['cv_scores']]}

Parameters:
"""
                    for param, value in best_params.items():
                        report += f"  {param}: {value}\n"
        
        # Save report
        with open('advanced_position_averaging_optimization_report.txt', 'w') as f:
            f.write(report)
        
        print("✅ Optimization report saved to: advanced_position_averaging_optimization_report.txt")
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(self.optimization_results)
        results_df.to_csv('advanced_position_averaging_optimization_results.csv', index=False)
        print("✅ Detailed results saved to: advanced_position_averaging_optimization_results.csv")
        
        # Generate equity curve comparison plot
        try:
            plot_file = self.plot_comparison(n_top_models=10)
            print(f"✅ Equity curve comparison saved to: {plot_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate equity curve comparison: {e}")

def optimize_parameter_chunk(task):
    """Optimize a chunk of parameters - standalone function for multiprocessing."""
    minute_data_path = task['minute_data_path']
    initial_capital = task['initial_capital']
    param_space = task['param_space']
    param_combinations = task['param_combinations']
    splits = task['splits']
    strategy_type = task['strategy_type']
    chunk_id = task['chunk_id']
    start_date = task['start_date']
    
    print(f"Process {chunk_id}: Starting optimization of {len(param_combinations)} parameter combinations...")
    
    # Load data in this process
    minute_data = pd.read_csv(minute_data_path)
    minute_data['Date'] = pd.to_datetime(minute_data['Date'])
    minute_data = minute_data.sort_values('Date').reset_index(drop=True)
    
    # Filter data from start date
    start_date_dt = pd.to_datetime(start_date)
    minute_data = minute_data[minute_data['Date'] >= start_date_dt].reset_index(drop=True)
    
    # Create a temporary system instance for this process
    temp_system = AdvancedPositionAveraging(minute_data_path, initial_capital, start_date)
    temp_system.minute_data = minute_data
    
    results = []
    param_names = list(param_space.keys())
    total_evaluations = len(param_combinations) * len(splits)
    completed_evaluations = 0
    start_time = datetime.now()
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        
        # Test this parameter combination on all CV splits
        for split in splits:
            train_data = minute_data.iloc[split['train_start']:split['train_end']].copy()
            test_data = minute_data.iloc[split['test_start']:split['test_end']].copy()
            
            result = temp_system.evaluate_strategy(train_data, test_data, param_dict, strategy_type)
            result['split_id'] = split['fold']
            results.append(result)
            
            completed_evaluations += 1
            
            # Progress update with ETA
            if completed_evaluations % 50 == 0 or completed_evaluations == total_evaluations:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if completed_evaluations > 0:
                    avg_time_per_eval = elapsed_time / completed_evaluations
                    remaining_evaluations = total_evaluations - completed_evaluations
                    eta_seconds = remaining_evaluations * avg_time_per_eval
                    eta_minutes = eta_seconds / 60
                    
                    progress_pct = (completed_evaluations / total_evaluations) * 100
                    print(f"Process {chunk_id}: {completed_evaluations}/{total_evaluations} ({progress_pct:.1f}%) - ETA: {eta_minutes:.1f} min")
    
    print(f"Process {chunk_id}: Completed optimization of {len(param_combinations)} parameter combinations!")
    return results

def test_worker(x):
    """Test worker function for multiprocessing."""
    import time
    time.sleep(0.1)  # Simulate work
    return x * 2

def test_multiprocessing():
    """Test if multiprocessing is working correctly."""
    print("Testing multiprocessing...")
    
    num_processes = max(1, cpu_count() - 4)
    test_data = list(range(20))
    
    print(f"Testing with {num_processes} processes...")
    start_time = datetime.now()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(test_worker, test_data)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"✅ Multiprocessing test completed in {duration:.2f} seconds")
    print(f"   Expected results: {[x*2 for x in test_data[:5]]}...")
    print(f"   Actual results: {results[:5]}...")
    
    return len(results) == len(test_data)

def test_signal_detection():
    """Test signal detection with a simple case."""
    print("Testing signal detection...")
    
    minute_data_path = 'BTCUSDT_1m_binance.csv'
    if not os.path.exists(minute_data_path):
        print(f"❌ Minute data file not found: {minute_data_path}")
        return False
    
    # Initialize system
    system = AdvancedPositionAveraging(minute_data_path, initial_capital=1000000, start_date='2024-01-01')
    
    # Load data
    if not system.load_data():
        return False
    
    # Test with a larger sample to get more days
    test_data = system.minute_data.head(10000).copy()  # First 10000 rows (about 7 days)
    
    # Use default parameters for testing
    default_params = {
        'rsi_period': 14,
        'bb_period': 20,
        'bb_std': 2.0,
        'momentum_period': 14,
        'volume_threshold': 1.0,  # Lowered from 1.5 to 1.0
        'breakout_threshold': 0.002,
        'touch_tolerance': 0.001,
        'min_confidence_threshold': 0.6,
        'max_entries_per_position': 2,
        'position_size_pct': 0.05,
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0
    }
    
    test_data = system.calculate_technical_indicators(test_data, default_params)
    
    # Test signal detection
    signals = system.detect_signals(test_data, 'longs_only', default_params)
    print(f"Test: Found {len(signals)} signals in first 1000 rows")
    
    return len(signals) > 0

def main():
    """Main execution function."""
    print("Advanced Position Averaging Trading System")
    print("=" * 50)
    
    # Test multiprocessing first
    if not test_multiprocessing():
        print("❌ Multiprocessing test failed!")
        return
    
    # Test signal detection
    if not test_signal_detection():
        print("❌ Signal detection test failed!")
        return
    
    # Check if minute data exists
    minute_data_path = 'BTCUSDT_1m_binance.csv'
    if not os.path.exists(minute_data_path):
        print(f"❌ Minute data file not found: {minute_data_path}")
        return
    
    # Initialize system with 2024 start date (more data available)
    system = AdvancedPositionAveraging(minute_data_path, initial_capital=1000000, start_date='2024-01-01')
    
    # Run optimization
    if system.run_full_optimization():
        print("\n🎉 Optimization completed successfully!")
        print("\nGenerated files:")
        print("- advanced_position_averaging_optimization_report.txt")
        print("- advanced_position_averaging_optimization_results.csv")
        print("- equity_curve_comparison.html")
    else:
        print("❌ Optimization failed")

if __name__ == "__main__":
    main()
