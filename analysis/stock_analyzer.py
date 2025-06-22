import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import ta
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self):
        self.cache = {}  # Simple cache for stock data
        
    def analyze_stock(self, ticker: str, period: str = "1y") -> Dict:
        """
        Comprehensive stock analysis including technical and fundamental indicators
        
        Args:
            ticker: Stock ticker symbol
            period: Analysis period (1y, 6mo, 3mo, etc.)
            
        Returns:
            Dictionary containing complete analysis
        """
        try:
            logger.info(f"Analyzing stock: {ticker}")
            
            # Get stock data
            stock_data = self.get_stock_data(ticker, period)
            if stock_data.empty:
                return {"error": f"No data available for {ticker}"}
            
            # Get stock info
            stock_info = self.get_stock_info(ticker)
            
            # Perform various analyses
            analysis = {
                "ticker": ticker,
                "company_name": stock_info.get("longName", ticker),
                "current_price": stock_data['Close'].iloc[-1],
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                
                # Basic metrics
                "basic_metrics": self._calculate_basic_metrics(stock_data),
                
                # Technical analysis
                "technical_analysis": self._technical_analysis(stock_data),
                
                # Fundamental analysis
                "fundamental_analysis": self._fundamental_analysis(stock_info, stock_data),
                
                # Risk analysis
                "risk_analysis": self._risk_analysis(stock_data),
                
                # Price targets and recommendations
                "price_targets": self._calculate_price_targets(stock_data, stock_info),
                
                # Overall score
                "overall_score": 0  # Will be calculated later
            }
            
            # Calculate overall investment score
            analysis["overall_score"] = self._calculate_overall_score(analysis)
            analysis["recommendation"] = self._get_recommendation(analysis["overall_score"])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock {ticker}: {e}")
            return {"error": str(e)}
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical stock data"""
        try:
            cache_key = f"{ticker}_{period}"
            
            # Check cache (simple time-based cache)
            if cache_key in self.cache:
                cached_data, cache_time = self.cache[cache_key]
                if datetime.now() - cache_time < timedelta(minutes=15):
                    return cached_data
            
            # Fetch fresh data
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if not data.empty:
                # Cache the data
                self.cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get stock fundamental information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {e}")
            return {}
    
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate basic stock metrics"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Price changes
            price_1d = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_7d = data['Close'].iloc[-7] if len(data) > 7 else current_price
            price_30d = data['Close'].iloc[-30] if len(data) > 30 else current_price
            
            # Volume analysis
            avg_volume = data['Volume'].mean()
            current_volume = data['Volume'].iloc[-1]
            
            return {
                "current_price": round(current_price, 2),
                "change_1d": round(((current_price - price_1d) / price_1d) * 100, 2),
                "change_7d": round(((current_price - price_7d) / price_7d) * 100, 2),
                "change_30d": round(((current_price - price_30d) / price_30d) * 100, 2),
                "volume_ratio": round(current_volume / avg_volume, 2),
                "avg_volume": int(avg_volume),
                "price_range_52w": {
                    "high": round(data['High'].max(), 2),
                    "low": round(data['Low'].min(), 2)
                }
            }
        except:
            return {}
    
    def _technical_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform technical analysis"""
        try:
            # Moving averages
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_histogram'] = macd.macd_diff()
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bollinger.bollinger_hband()
            data['BB_lower'] = bollinger.bollinger_lband()
            data['BB_middle'] = bollinger.bollinger_mavg()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
            data['Stoch_k'] = stoch.stoch()
            data['Stoch_d'] = stoch.stoch_signal()
            
            current_price = data['Close'].iloc[-1]
            
            return {
                "moving_averages": {
                    "sma_20": round(data['SMA_20'].iloc[-1], 2),
                    "sma_50": round(data['SMA_50'].iloc[-1], 2),
                    "ema_12": round(data['EMA_12'].iloc[-1], 2),
                    "ema_26": round(data['EMA_26'].iloc[-1], 2),
                    "price_vs_sma20": "Above" if current_price > data['SMA_20'].iloc[-1] else "Below",
                    "price_vs_sma50": "Above" if current_price > data['SMA_50'].iloc[-1] else "Below"
                },
                "momentum_indicators": {
                    "rsi": round(data['RSI'].iloc[-1], 2),
                    "rsi_signal": self._get_rsi_signal(data['RSI'].iloc[-1]),
                    "macd": round(data['MACD'].iloc[-1], 4),
                    "macd_signal": round(data['MACD_signal'].iloc[-1], 4),
                    "macd_histogram": round(data['MACD_histogram'].iloc[-1], 4),
                    "stochastic_k": round(data['Stoch_k'].iloc[-1], 2),
                    "stochastic_d": round(data['Stoch_d'].iloc[-1], 2)
                },
                "volatility": {
                    "bollinger_position": self._get_bollinger_position(
                        current_price, data['BB_upper'].iloc[-1], data['BB_lower'].iloc[-1]
                    ),
                    "bb_upper": round(data['BB_upper'].iloc[-1], 2),
                    "bb_lower": round(data['BB_lower'].iloc[-1], 2),
                    "volatility_20d": round(data['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100, 2)
                }
            }
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}
    
    def _fundamental_analysis(self, info: Dict, data: pd.DataFrame) -> Dict:
        """Perform fundamental analysis"""
        try:
            return {
                "valuation": {
                    "market_cap": info.get("marketCap", 0),
                    "enterprise_value": info.get("enterpriseValue", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "forward_pe": info.get("forwardPE", 0),
                    "peg_ratio": info.get("pegRatio", 0),
                    "price_to_book": info.get("priceToBook", 0),
                    "price_to_sales": info.get("priceToSalesTrailing12Months", 0)
                },
                "profitability": {
                    "profit_margin": info.get("profitMargins", 0),
                    "operating_margin": info.get("operatingMargins", 0),
                    "return_on_equity": info.get("returnOnEquity", 0),
                    "return_on_assets": info.get("returnOnAssets", 0),
                    "revenue_growth": info.get("revenueGrowth", 0),
                    "earnings_growth": info.get("earningsGrowth", 0)
                },
                "financial_health": {
                    "debt_to_equity": info.get("debtToEquity", 0),
                    "current_ratio": info.get("currentRatio", 0),
                    "quick_ratio": info.get("quickRatio", 0),
                    "cash_per_share": info.get("totalCashPerShare", 0),
                    "free_cash_flow": info.get("freeCashflow", 0)
                },
                "dividend": {
                    "dividend_yield": info.get("dividendYield", 0),
                    "dividend_rate": info.get("dividendRate", 0),
                    "payout_ratio": info.get("payoutRatio", 0),
                    "ex_dividend_date": info.get("exDividendDate", None)
                }
            }
        except:
            return {}
    
    def _risk_analysis(self, data: pd.DataFrame) -> Dict:
        """Analyze risk metrics"""
        try:
            returns = data['Close'].pct_change().dropna()
            
            # Calculate various risk metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            var_95 = np.percentile(returns, 5)  # Value at Risk (95% confidence)
            
            # Beta calculation (vs SPY as market proxy)
            try:
                spy = yf.Ticker("SPY")
                spy_data = spy.history(period="1y")
                spy_returns = spy_data['Close'].pct_change().dropna()
                
                # Align dates
                common_dates = returns.index.intersection(spy_returns.index)
                if len(common_dates) > 30:
                    stock_aligned = returns.loc[common_dates]
                    spy_aligned = spy_returns.loc[common_dates]
                    
                    covariance = np.cov(stock_aligned, spy_aligned)[0][1]
                    spy_variance = np.var(spy_aligned)
                    beta = covariance / spy_variance if spy_variance != 0 else 1.0
                else:
                    beta = 1.0
            except:
                beta = 1.0
            
            # Sharpe ratio (assuming risk-free rate of 3%)
            risk_free_rate = 0.03 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            
            return {
                "volatility": round(volatility * 100, 2),  # As percentage
                "beta": round(beta, 2),
                "var_95": round(var_95 * 100, 2),  # As percentage
                "sharpe_ratio": round(sharpe_ratio, 2),
                "max_drawdown": round(self._calculate_max_drawdown(data['Close']) * 100, 2),
                "risk_grade": self._get_risk_grade(volatility, beta)
            }
        except:
            return {}
    
    def _calculate_price_targets(self, data: pd.DataFrame, info: Dict) -> Dict:
        """Calculate price targets based on various methods"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Technical price targets
            resistance_levels = self._find_resistance_levels(data)
            support_levels = self._find_support_levels(data)
            
            # Analyst targets (if available)
            analyst_target = info.get("targetMeanPrice", 0)
            
            # Simple moving average targets
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            return {
                "current_price": round(current_price, 2),
                "analyst_target": round(analyst_target, 2) if analyst_target else "N/A",
                "technical_targets": {
                    "resistance_1": round(resistance_levels[0], 2) if resistance_levels else "N/A",
                    "resistance_2": round(resistance_levels[1], 2) if len(resistance_levels) > 1 else "N/A",
                    "support_1": round(support_levels[0], 2) if support_levels else "N/A",
                    "support_2": round(support_levels[1], 2) if len(support_levels) > 1 else "N/A"
                },
                "moving_average_targets": {
                    "sma_20": round(sma_20, 2),
                    "sma_50": round(sma_50, 2)
                },
                "upside_potential": round(((analyst_target - current_price) / current_price * 100), 2) if analyst_target else "N/A"
            }
        except:
            return {}
    
    def _calculate_overall_score(self, analysis: Dict) -> float:
        """Calculate overall investment score (0-100)"""
        try:
            score = 50  # Base score
            
            # Technical indicators scoring
            technical = analysis.get("technical_analysis", {})
            if technical:
                # RSI scoring
                rsi = technical.get("momentum_indicators", {}).get("rsi", 50)
                if 30 <= rsi <= 70:  # Good range
                    score += 10
                elif rsi < 30:  # Oversold
                    score += 5
                elif rsi > 70:  # Overbought
                    score -= 5
                
                # MACD scoring
                macd = technical.get("momentum_indicators", {}).get("macd", 0)
                macd_signal = technical.get("momentum_indicators", {}).get("macd_signal", 0)
                if macd > macd_signal:  # Bullish
                    score += 5
                
                # Moving average scoring
                ma_info = technical.get("moving_averages", {})
                if ma_info.get("price_vs_sma20") == "Above":
                    score += 5
                if ma_info.get("price_vs_sma50") == "Above":
                    score += 5
            
            # Fundamental scoring
            fundamental = analysis.get("fundamental_analysis", {})
            if fundamental:
                valuation = fundamental.get("valuation", {})
                pe_ratio = valuation.get("pe_ratio", 0)
                if 0 < pe_ratio < 25:  # Reasonable PE
                    score += 10
                
                profitability = fundamental.get("profitability", {})
                if profitability.get("profit_margin", 0) > 0.1:  # >10% profit margin
                    score += 5
                if profitability.get("revenue_growth", 0) > 0.05:  # >5% revenue growth
                    score += 5
            
            # Risk adjustment
            risk = analysis.get("risk_analysis", {})
            if risk:
                volatility = risk.get("volatility", 50)
                if volatility < 30:  # Low volatility
                    score += 5
                elif volatility > 60:  # High volatility
                    score -= 10
            
            return max(0, min(100, score))  # Clamp between 0-100
            
        except:
            return 50  # Default score
    
    def _get_recommendation(self, score: float) -> str:
        """Get investment recommendation based on score"""
        if score >= 80:
            return "Strong Buy"
        elif score >= 70:
            return "Buy"
        elif score >= 60:
            return "Hold"
        elif score >= 40:
            return "Weak Hold"
        else:
            return "Avoid"
    
    # Helper methods
    def _get_rsi_signal(self, rsi: float) -> str:
        if rsi > 70:
            return "Overbought"
        elif rsi < 30:
            return "Oversold"
        else:
            return "Neutral"
    
    def _get_bollinger_position(self, price: float, upper: float, lower: float) -> str:
        if price > upper:
            return "Above Upper Band"
        elif price < lower:
            return "Below Lower Band"
        else:
            return "Within Bands"
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _get_risk_grade(self, volatility: float, beta: float) -> str:
        """Assign risk grade based on volatility and beta"""
        risk_score = volatility * 0.6 + abs(beta - 1) * 0.4
        
        if risk_score < 0.2:
            return "Low Risk"
        elif risk_score < 0.4:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Find key resistance levels"""
        try:
            highs = data['High'].rolling(10).max()
            resistance_levels = []
            
            for i in range(len(highs) - 20, len(highs)):
                if i > 10 and i < len(highs) - 10:
                    if highs.iloc[i] == highs.iloc[i-10:i+10].max():
                        resistance_levels.append(highs.iloc[i])
            
            return sorted(set(resistance_levels), reverse=True)[:3]
        except:
            return []
    
    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Find key support levels"""
        try:
            lows = data['Low'].rolling(10).min()
            support_levels = []
            
            for i in range(len(lows) - 20, len(lows)):
                if i > 10 and i < len(lows) - 10:
                    if lows.iloc[i] == lows.iloc[i-10:i+10].min():
                        support_levels.append(lows.iloc[i])
            
            return sorted(set(support_levels))[:3]
        except:
            return [] 