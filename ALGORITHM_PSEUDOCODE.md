# Advanced Position Averaging Algorithm - Pseudo-Code

## System Overview
This system implements a multi-strategy position averaging trading system with cross-validation, deflated Sharpe ratio, and triple barrier method.

---

## Main Algorithm Flow

```pseudo-code
//--------------------------------------------------------------------
// PHASE 1: DATA LOADING & PREPARATION
//--------------------------------------------------------------------

1. Load minute-by-minute price data (BTCUSDT_1m_binance.csv)
2. Filter data from start_date (e.g., 2024-01-01)
3. Calculate technical indicators:
   - Returns, Volatility
   - SMA(20), SMA(50), EMA(12), EMA(26)
   - RSI(14)
   - Bollinger Bands (BB_period=20, BB_std=2.0)
   - MACD (12, 26, 9)
   - Volume indicators (volume_sma, volume_ratio)
   - ATR(14)
   - Support/Resistance (50-bar rolling max/min)
   - Momentum(14), Momentum(5)

4. Create cross-validation splits:
   - n_splits = 5 folds
   - Train size = 30% of data
   - Test size = (remaining 70% / n_splits)
   - Embargo period = 1% between train and test
```

---

## Signal Detection (Multi-Indicator Confluence)

```pseudo-code
//--------------------------------------------------------------------
// PHASE 2: SIGNAL DETECTION (Similar to Strategy 5.2.4)
//--------------------------------------------------------------------

FOR each day in the dataset:
    day_data = GetMinuteDataForDay(date)
    
    // Calculate base confidence from volatility
    daily_vol = standard_deviation(recent_returns[0:20])
    base_confidence = min(daily_vol * 100, 1.0)
    
    // Skip if volatility too low
    IF base_confidence < 0.01:
        CONTINUE
    END IF
    
    // STRATEGY 1: Multi-Indicator Breakout (Primary)
    signals = DetectMultiIndicatorBreakout(day_data, strategy_type, params)
    
    // STRATEGY 2: Mean Reversion
    signals += DetectMeanReversion(day_data, strategy_type, params)
    
    // STRATEGY 3: Momentum
    signals += DetectMomentum(day_data, strategy_type, params)
    
    // STRATEGY 4: Support/Resistance
    signals += DetectSupportResistance(day_data, strategy_type, params)
END FOR
```

---

## Multi-Indicator Breakout Detection (Primary Strategy)

```pseudo-code
//--------------------------------------------------------------------
// MULTI-INDICATOR BREAKOUT SIGNAL (Inspired by Strategy 5.2.4)
//--------------------------------------------------------------------

Function: DetectMultiIndicatorBreakout(day_data, strategy_type, params)
    signals = []
    
    FOR each minute in day_data:
        // Get indicators (using PREVIOUS bar to avoid lookahead bias)
        signal_price = close[i-1]
        bb_position = (close[i-1] - bb_lower[i-1]) / (bb_upper[i-1] - bb_lower[i-1])
        rsi = rsi[i-1]
        macd_hist = macd_histogram[i-1]
        volume_ratio = volume_ratio[i-1]
        
        // LONG ENTRY SIGNAL (Similar to Strategy 5.2.4's LongEntrySignal)
        IF strategy_type in ['longs_only', 'combined']:
            IF (bb_position < 0.2 AND                    // Below lower Bollinger Band
                rsi < 35 AND                              // Oversold
                macd_hist < -0.0002 AND                   // MACD falling (momentum)
                volume_ratio > 1.2):                       // Volume confirmation
                
                confidence = base_confidence * volume_ratio / 1.2
                ADD signal:
                    - Direction: LONG
                    - Price: signal_price
                    - Strategy: 'multi_indicator_breakout'
                    - Confidence: confidence
                    - Size: 35% of position
            END IF
        END IF
        
        // SHORT ENTRY SIGNAL (Similar to Strategy 5.2.4's ShortEntrySignal)
        IF strategy_type in ['shorts_only', 'combined']:
            IF (bb_position > 0.8 AND                    // Above upper Bollinger Band
                rsi > 65 AND                              // Overbought
                macd_hist > 0.0002 AND                    // MACD rising (momentum)
                volume_ratio > 1.2):                       // Volume confirmation
                
                confidence = base_confidence * volume_ratio / 1.2
                ADD signal:
                    - Direction: SHORT
                    - Price: signal_price
                    - Strategy: 'multi_indicator_breakout'
                    - Confidence: confidence
                    - Size: 35% of position
            END IF
        END IF
    END FOR
    
    RETURN signals
END Function
```

---

## Position Creation (Position Averaging)

```pseudo-code
//--------------------------------------------------------------------
// PHASE 3: POSITION AVERAGING
//--------------------------------------------------------------------

Function: CreatePositions(signals, params)
    positions = []
    signal_groups = GroupSignalsByDay(signals)
    
    FOR each day in signal_groups:
        day_signals = signal_groups[day]
        
        // Group by direction
        long_signals = FILTER(day_signals, direction == 'long')
        short_signals = FILTER(day_signals, direction == 'short')
        
        // Create averaged long position
        IF long_signals.length > 0:
            long_position = CreateAveragedPosition(long_signals, 'long', params)
            positions.ADD(long_position)
        END IF
        
        // Create averaged short position
        IF short_signals.length > 0:
            short_position = CreateAveragedPosition(short_signals, 'short', params)
            positions.ADD(short_position)
        END IF
    END FOR
    
    RETURN positions
END Function

Function: CreateAveragedPosition(signals, direction, params)
    // Sort by confidence, select best N
    signals.SORT_BY(confidence, DESC)
    max_entries = params['max_entries_per_position'] // e.g., 3
    
    selected_signals = signals[0:max_entries]
    
    // Calculate average entry price
    entry_prices = selected_signals.price
    avg_entry_price = MEAN(entry_prices)
    total_size_pct = SUM(selected_signals.size_pct)
    
    // Calculate ATR for stop loss/take profit
    atr = GetATRForDate(selected_signals[0].timestamp.date)
    
    // Calculate stop loss and take profit (like Strategy 5.2.4)
    stop_loss_mult = params['stop_loss_atr_multiplier']  // e.g., 2.0
    take_profit_mult = params['take_profit_atr_multiplier'] // e.g., 2.5
    
    IF direction == 'long':
        stop_loss = avg_entry_price - (stop_loss_mult * atr)
        take_profit = avg_entry_price + (take_profit_mult * atr)
    ELSE: // short
        stop_loss = avg_entry_price + (stop_loss_mult * atr)
        take_profit = avg_entry_price - (take_profit_mult * atr)
    END IF
    
    RETURN {
        date: selected_signals[0].timestamp.date,
        direction: direction,
        entry_prices: entry_prices,
        avg_entry_price: avg_entry_price,
        total_size_pct: total_size_pct,
        stop_loss: stop_loss,
        take_profit: take_profit,
        num_entries: selected_signals.length,
        strategies: [signal.strategy for signal in selected_signals],
        avg_confidence: MEAN([signal.confidence for signal in selected_signals])
    }
END Function
```

---

## Triple Barrier Method with Trailing Stop

```pseudo-code
//--------------------------------------------------------------------
// PHASE 4: TRIPLE BARRIER PnL CALCULATION (Like Strategy 5.2.4)
//--------------------------------------------------------------------

Function: CalculateTripleBarrierPnL(positions, data, params)
    pnl_data = []
    cumulative_pnl = 0
    
    FOR each position in positions:
        position_date = position.date
        entry_price = position.avg_entry_price
        stop_loss = position.stop_loss
        take_profit = position.take_profit
        direction = position.direction
        
        // Find position in data
        position_idx = FindPositionInData(position_date, data)
        
        // Time barrier: maximum 5 days
        max_lookahead = min(5 * 24 * 60, data.length - position_idx)
        
        // Trailing stop parameters (like Strategy 5.2.4)
        move_sl_to_be_coef = params['move_sl_to_be_coef']      // e.g., 3.0
        sl_to_be_add_pips_coef = params['sl_to_be_add_pips_coef'] // e.g., 3.5
        
        // Calculate ATR for trailing stop
        atr_45 = MEAN(data.atr[position_idx-45:position_idx])
        atr_55 = MEAN(data.atr[position_idx-55:position_idx])
        
        // Breakeven trigger
        breakeven_trigger = move_sl_to_be_coef * atr_45
        
        // Stop loss adjustment when moved to BE
        breakeven_adjustment = sl_to_be_add_pips_coef * atr_55
        
        // Initialize tracking variables
        trailing_stop_activated = FALSE
        current_stop_loss = stop_loss
        exit_price = entry_price
        exit_reason = 'unknown'
        
        // Check each minute after entry
        FOR i = 1 TO max_lookahead:
            current_bar = data[position_idx + i]
            current_high = current_bar.high
            current_low = current_bar.low
            current_close = current_bar.close
            
            IF direction == 'long':
                // Calculate unrealized profit
                profit_amount = current_close - entry_price
                
                // TRAILING STOP TO BREAKEVEN LOGIC (like Strategy 5.2.4)
                IF profit_amount >= breakeven_trigger AND NOT trailing_stop_activated:
                    current_stop_loss = entry_price + breakeven_adjustment
                    trailing_stop_activated = TRUE
                END IF
                
                // Check stop loss
                IF current_low <= current_stop_loss:
                    exit_price = current_stop_loss
                    exit_reason = IF trailing_stop_activated THEN 'trailing_stop' ELSE 'stop_loss'
                    BREAK
                END IF
                
                // Check take profit
                IF current_high >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    BREAK
                END IF
                
            ELSE: // short
                // Calculate unrealized profit
                profit_amount = entry_price - current_close
                
                // TRAILING STOP TO BREAKEVEN LOGIC
                IF profit_amount >= breakeven_trigger AND NOT trailing_stop_activated:
                    current_stop_loss = entry_price - breakeven_adjustment
                    trailing_stop_activated = TRUE
                END IF
                
                // Check stop loss
                IF current_high >= current_stop_loss:
                    exit_price = current_stop_loss
                    exit_reason = IF trailing_stop_activated THEN 'trailing_stop' ELSE 'stop_loss'
                    BREAK
                END IF
                
                // Check take profit
                IF current_low <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    BREAK
                END IF
            END IF
        END FOR
        
        // If no barrier hit, exit at final price
        IF exit_reason == 'unknown':
            exit_price = data[min(position_idx + max_lookahead - 1, data.length - 1)].close
            exit_reason = 'final_price'
        END IF
        
        // Calculate PnL
        pnl_pct = (exit_price - entry_price) / entry_price
        IF direction == 'short':
            pnl_pct = -pnl_pct
        END IF
        
        // Dollar PnL
        position_value = initial_capital * position.total_size_pct
        dollar_pnl = position_value * pnl_pct
        cumulative_pnl += dollar_pnl
        
        ADD to pnl_data:
            - date: position.date
            - direction: position.direction
            - entry_price: entry_price
            - exit_price: exit_price
            - pnl_pct: pnl_pct
            - dollar_pnl: dollar_pnl
            - cumulative_pnl: cumulative_pnl
            - exit_reason: exit_reason
    END FOR
    
    RETURN pnl_data
END Function
```

---

## Optimization & Cross-Validation

```pseudo-code
//--------------------------------------------------------------------
// PHASE 5: PARAMETER OPTIMIZATION WITH CROSS-VALIDATION
//--------------------------------------------------------------------

Function: OptimizeParameters(strategy_type)
    // Generate all parameter combinations
    param_combinations = cartesian_product(param_space)
    
    // Create CV splits
    splits = CreateCVSplits(n_splits=5)
    
    // Calculate Buy & Hold benchmark
    bh_benchmark = CalculateBuyHoldBenchmark(minute_data)
    PRINT "Buy & Hold: Return=" + bh_benchmark.return + 
          ", Sharpe=" + bh_benchmark.sharpe
    
    // Parallel processing
    num_processes = cpu_count() - 4
    tasks = SplitIntoChunks(param_combinations, num_processes)
    
    FOR each task in tasks:
        IN PARALLEL:
            EvaluateParametersOnAllCVSplits(task, strategy_type)
        END PARALLEL
    END FOR
    
    // Find best parameters by deflated Sharpe ratio
    best_params = MAX_BY(optimization_results, avg_deflated_sharpe)
    
    RETURN best_params
END Function

Function: EvaluateStrategy(train_data, test_data, params, strategy_type)
    // Calculate indicators
    train_data = CalculateTechnicalIndicators(train_data, params)
    test_data = CalculateTechnicalIndicators(test_data, params)
    
    // Detect signals on test data only
    signals = DetectSignals(test_data, strategy_type, params)
    
    // Create positions
    positions = CreatePositions(signals, params)
    
    // Calculate PnL
    pnl_df = CalculateTripleBarrierPnL(positions, test_data, params)
    
    // Calculate metrics
    returns = pnl_df.dollar_pnl / initial_capital
    total_return = SUM(returns)
    sharpe_ratio = MEAN(returns) / STD(returns) * SQRT(252)
    
    // Calculate deflated Sharpe ratio (Lopez de Prado)
    deflated_sharpe = CalculateDeflatedSharpeRatio(returns, n_param_combinations)
    
    // Calculate max drawdown
    cumulative_returns = CUMULATIVE_PRODUCT(1 + returns)
    running_max = EXPANDING_MAX(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = MIN(drawdowns)
    
    // Win rate
    win_rate = MEAN(returns > 0)
    
    RETURN {
        sharpe_ratio: sharpe_ratio,
        deflated_sharpe: deflated_sharpe,
        total_return: total_return,
        max_drawdown: max_drawdown,
        num_trades: pnl_df.length,
        win_rate: win_rate,
        params: params
    }
END Function
```

---

## Market Regime Detection with Hurst Exponent

```pseudo-code
//--------------------------------------------------------------------
// MARKET REGIME DETECTION (NEW ENHANCEMENT)
//--------------------------------------------------------------------

Function: CalculateHurstExponent(prices, period=100)
    // Calculate Hurst exponent using R/S (Rescaled Range) analysis
    log_returns = diff(log(prices))
    
    FOR each lag from 2 to max_lag:
        // Divide returns into windows of size 'lag'
        FOR each window:
            // Calculate mean return
            mean_return = MEAN(window)
            
            // Calculate cumulative deviations from mean
            cumulative = CUMULATIVE_SUM(window - mean_return)
            
            // Calculate range R
            R = MAX(cumulative) - MIN(cumulative)
            
            // Calculate standard deviation S
            S = STD(window)
            
            // R/S ratio
            IF S > 0:
                rs_values.ADD(R / S)
            END IF
        END FOR
    END FOR
    
    // Fit log-log regression: log(R/S) = H * log(lag) + c
    hurst = POLYFIT(log(lags), log(rs_values), 1)[0]
    
    RETURN CLAMP(hurst, 0.0, 1.0)
END Function

Function: ClassifyMarketRegime(hurst)
    IF hurst > 0.55:
        RETURN 'trending'        // Strong trending market
    ELSE IF hurst > 0.52:
        RETURN 'weak_trending'   // Weak trending
    ELSE IF hurst < 0.45:
        RETURN 'mean_reverting'  // Strong mean reversion
    ELSE IF hurst < 0.48:
        RETURN 'weak_mean_reverting'
    ELSE:
        RETURN 'neutral'         // Random walk
END Function

Function: GetRegimeStrategyWeights(regime)
    IF regime == 'trending':
        RETURN {
            'breakout': 1.5,           // 150% weight
            'momentum': 1.8,           // 180% weight
            'mean_reversion': 0.3,     // 30% weight
            'support_resistance': 0.5  // 50% weight
        }
    ELSE IF regime == 'mean_reverting':
        RETURN {
            'breakout': 0.4,
            'momentum': 0.3,
            'mean_reversion': 1.6,
            'support_resistance': 1.5
        }
    ELSE: // neutral
        RETURN {
            'breakout': 1.0,
            'momentum': 1.0,
            'mean_reversion': 1.0,
            'support_resistance': 1.0
        }
    END IF
END Function

// In signal detection:
FOR each day:
    // Calculate Hurst exponent (rolling window)
    hurst = CalculateHurstExponentRolling(prices, period=100)
    regime = ClassifyMarketRegime(hurst)
    
    // Get strategy weights based on regime
    weights = GetRegimeStrategyWeights(regime)
    
    // Generate signals with regime-weighted confidence
    breakout_signals = DetectBreakoutSignals(...)
    FOR each signal in breakout_signals:
        signal.confidence *= weights['breakout']
    END FOR
    
    momentum_signals = DetectMomentumSignals(...)
    FOR each signal in momentum_signals:
        signal.confidence *= weights['momentum']
    END FOR
    
    // Mean reversion signals get higher weight in mean-reverting markets
    mean_reversion_signals = DetectMeanReversionSignals(...)
    FOR each signal in mean_reversion_signals:
        signal.confidence *= weights['mean_reversion']
    END FOR
END FOR
```

---

## Key Differences from Strategy 5.2.4

```pseudo-code
// Strategy 5.2.4 uses:
// - BearsPower/BullsPower momentum
// - BB extremes for entry
// - MACD confirmation
// - ADX, StdDev, CCI for exit
// - Single entry positions
// - Fixed SL/PT at entry

// Our system uses:
// - RSI instead of Bears/Bulls Power
// - Same BB extremes logic
// - Same MACD confirmation
// - Multiple entry signals per day (position averaging)
// - ATR-based dynamic SL/PT
// - Trailing stop to breakeven + buffer
// - Cross-validation for parameter optimization
// - Deflated Sharpe ratio for multiple testing correction
// - MARKET REGIME DETECTION with Hurst exponent (NEW!)
// - Dynamic strategy weighting based on regime
// - Optional adaptive indicator periods
```

---

## Parameter Explanation

```pseudo-code
position_size_pct: 0.05-0.15
    - How much capital to risk per position (5-15%)
    
stop_loss_atr_multiplier: 1.5-2.0
    - Stop loss = multiplier × ATR(14)
    - Wider stops allow more room for volatility
    
take_profit_atr_multiplier: 2.5-4.0
    - Take profit = multiplier × ATR(14)
    - Higher targets capture bigger moves
    
move_sl_to_be_coef: 2.5-3.0
    - When profit >= this × ATR(45), move SL to breakeven
    - Lower = faster protection, Higher = more room
    
sl_to_be_add_pips_coef: 3.5-4.5
    - When moving to BE, add this × ATR(55) as buffer
    - Protects from getting stopped at exact entry
    
volume_threshold: 1.0-1.2
    - Require volume >= threshold × average volume
    - Ensures real market moves, not low liquidity
    
breakout_threshold: 0.001-0.002
    - Price movement threshold for breakout detection
    - Higher = more selective signals

hurst_period: 50-200
    - Rolling window for Hurst exponent calculation
    - Lower = faster regime detection but more noise
    - Higher = smoother but slower response
    - Recommended: 100 bars (about 100 minutes)

use_dynamic_periods: False/True
    - Enable adaptive indicator periods based on volatility
    - Higher volatility = shorter periods (faster response)
    - Lower volatility = longer periods (smoother signals)
    - Similar to non-lag MA concept
```

---

## Market Regime Strategy Weighting

```pseudo-code
// Strategy confidence is adjusted based on market regime:

TRENDING MARKET (Hurst > 0.55):
    - Momentum signals: 180% confidence
    - Breakout signals: 150% confidence
    - Mean reversion: 30% confidence (avoid in trends!)
    - Support/Resistance: 50% confidence

MEAN REVERTING MARKET (Hurst < 0.45):
    - Mean reversion signals: 160% confidence
    - Support/Resistance: 150% confidence
    - Momentum: 30% confidence (avoid chasing!)
    - Breakout: 40% confidence

NEUTRAL MARKET (Hurst ≈ 0.5):
    - All strategies: 100% confidence (balanced)
```

