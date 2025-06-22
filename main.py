import os
import sys
import logging
from datetime import datetime
from typing import List, Dict
import json

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scrapers'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))

from scrapers.openinsider_scraper import OpenInsiderScraper
from analysis.stock_analyzer import StockAnalyzer
from ai.news_analyzer import NewsAnalyzer
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockInvestmentAnalyzer:
    def __init__(self):
        self.scraper = OpenInsiderScraper()
        self.stock_analyzer = StockAnalyzer()
        self.news_analyzer = NewsAnalyzer(
            openai_api_key=Config.OPENAI_API_KEY,
            news_api_key=Config.NEWS_API_KEY
        )
        
    def run_complete_analysis(self, days_back: int = 30, min_value: float = 100000, 
                            max_stocks: int = 10) -> Dict:
        """
        Run complete analysis pipeline:
        1. Scrape OpenInsider for CEO purchases
        2. Analyze each stock technically and fundamentally
        3. Get AI-powered news analysis
        4. Generate final recommendations
        """
        try:
            logger.info("Starting complete stock investment analysis...")
            
            # Step 1: Get CEO purchases from OpenInsider
            logger.info("Step 1: Scraping OpenInsider for CEO purchases...")
            ceo_purchases = self.scraper.get_ceo_purchases(days_back=days_back, min_value=min_value)
            
            if not ceo_purchases:
                logger.warning("No CEO purchases found")
                return {"error": "No CEO purchases found in the specified timeframe"}
            
            logger.info(f"Found {len(ceo_purchases)} CEO purchases")
            
            # Step 2: Analyze top stocks
            analyzed_stocks = []
            for i, purchase in enumerate(ceo_purchases[:max_stocks]):
                try:
                    logger.info(f"Step 2.{i+1}: Analyzing {purchase['ticker']} ({purchase['company_name']})")
                    
                    # Technical and fundamental analysis
                    stock_analysis = self.stock_analyzer.analyze_stock(purchase['ticker'])
                    
                    # News analysis
                    news_analysis = self.news_analyzer.analyze_stock_news(
                        ticker=purchase['ticker'],
                        company_name=purchase['company_name'],
                        days_back=Config.NEWS_LOOKBACK_DAYS
                    )
                    
                    # Combine all data
                    complete_analysis = {
                        "insider_purchase": purchase,
                        "stock_analysis": stock_analysis,
                        "news_analysis": news_analysis,
                        "final_recommendation": self._generate_final_recommendation(
                            purchase, stock_analysis, news_analysis
                        )
                    }
                    
                    analyzed_stocks.append(complete_analysis)
                    logger.info(f"Completed analysis for {purchase['ticker']}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {purchase['ticker']}: {e}")
                    continue
            
            # Step 3: Rank and sort recommendations
            final_results = self._rank_recommendations(analyzed_stocks)
            
            return {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_ceo_purchases_found": len(ceo_purchases),
                "stocks_analyzed": len(analyzed_stocks),
                "top_recommendations": final_results,
                "analysis_parameters": {
                    "days_back": days_back,
                    "min_purchase_value": min_value,
                    "max_stocks_analyzed": max_stocks
                }
            }
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            return {"error": str(e)}
    
    def _generate_final_recommendation(self, purchase: Dict, stock_analysis: Dict, 
                                     news_analysis: Dict) -> Dict:
        """Generate final investment recommendation combining all factors"""
        try:
            # Extract key metrics
            insider_value = purchase.get('value', 0)
            stock_score = stock_analysis.get('overall_score', 50)
            stock_recommendation = stock_analysis.get('recommendation', 'Hold')
            
            # News sentiment
            news_sentiment = 0
            news_recommendation = "Hold"
            if 'overall_recommendation' in news_analysis:
                news_sentiment = news_analysis['overall_recommendation'].get('overall_sentiment_score', 0)
                news_recommendation = news_analysis['overall_recommendation'].get('recommendation', 'Hold')
            
            # Calculate combined score (weighted average)
            # Insider activity: 30%, Technical/Fundamental: 40%, News: 30%
            insider_score = min(100, (insider_value / 1000000) * 20)  # Scale large purchases
            combined_score = (insider_score * 0.3) + (stock_score * 0.4) + ((news_sentiment + 100) / 2 * 0.3)
            
            # Determine final recommendation
            final_recommendation = self._determine_final_recommendation(
                combined_score, stock_recommendation, news_recommendation
            )
            
            # Calculate confidence based on agreement between methods
            confidence = self._calculate_confidence(stock_recommendation, news_recommendation, combined_score)
            
            return {
                "final_recommendation": final_recommendation,
                "combined_score": round(combined_score, 1),
                "confidence": confidence,
                "component_scores": {
                    "insider_activity_score": round(insider_score, 1),
                    "technical_fundamental_score": stock_score,
                    "news_sentiment_score": news_sentiment
                },
                "key_factors": self._extract_key_factors(purchase, stock_analysis, news_analysis),
                "risk_assessment": self._assess_overall_risk(stock_analysis, news_analysis),
                "investment_thesis": self._create_investment_thesis(purchase, stock_analysis, news_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating final recommendation: {e}")
            return {
                "final_recommendation": "Hold",
                "combined_score": 50,
                "confidence": "Low",
                "error": str(e)
            }
    
    def _determine_final_recommendation(self, combined_score: float, stock_rec: str, news_rec: str) -> str:
        """Determine final recommendation based on combined analysis"""
        if combined_score >= 80:
            return "Strong Buy"
        elif combined_score >= 70:
            return "Buy"
        elif combined_score >= 60:
            return "Moderate Buy"
        elif combined_score >= 40:
            return "Hold"
        elif combined_score >= 30:
            return "Weak Hold"
        else:
            return "Avoid"
    
    def _calculate_confidence(self, stock_rec: str, news_rec: str, combined_score: float) -> str:
        """Calculate confidence based on agreement between different analyses"""
        # Convert recommendations to numeric scores for comparison
        rec_scores = {
            "Strong Buy": 90, "Buy": 75, "Moderate Buy": 65, "Hold": 50,
            "Weak Hold": 35, "Avoid": 20, "Sell": 15, "Strong Sell": 5
        }
        
        stock_score = rec_scores.get(stock_rec, 50)
        news_score = rec_scores.get(news_rec, 50)
        
        # Calculate agreement
        score_diff = abs(stock_score - news_score)
        combined_diff = abs(combined_score - stock_score)
        
        if score_diff <= 15 and combined_diff <= 15:
            return "High"
        elif score_diff <= 25 and combined_diff <= 25:
            return "Medium"
        else:
            return "Low"
    
    def _extract_key_factors(self, purchase: Dict, stock_analysis: Dict, news_analysis: Dict) -> List[str]:
        """Extract key factors influencing the recommendation"""
        factors = []
        
        # Insider activity factor
        insider_value = purchase.get('value', 0)
        if insider_value > 1000000:
            factors.append(f"Large CEO purchase of {purchase.get('value_formatted', 'N/A')}")
        
        # Technical factors
        if 'technical_analysis' in stock_analysis:
            tech = stock_analysis['technical_analysis']
            if 'moving_averages' in tech:
                ma = tech['moving_averages']
                if ma.get('price_vs_sma20') == 'Above' and ma.get('price_vs_sma50') == 'Above':
                    factors.append("Price above key moving averages")
            
            if 'momentum_indicators' in tech:
                momentum = tech['momentum_indicators']
                rsi = momentum.get('rsi', 50)
                if 30 <= rsi <= 70:
                    factors.append(f"Healthy RSI level ({rsi})")
        
        # Fundamental factors
        if 'fundamental_analysis' in stock_analysis:
            fund = stock_analysis['fundamental_analysis']
            if 'profitability' in fund:
                prof = fund['profitability']
                if prof.get('profit_margin', 0) > 0.1:
                    factors.append("Strong profit margins")
                if prof.get('revenue_growth', 0) > 0.05:
                    factors.append("Positive revenue growth")
        
        # News factors
        if 'overall_recommendation' in news_analysis:
            news_rec = news_analysis['overall_recommendation']
            key_reasons = news_rec.get('key_reasons', [])
            factors.extend(key_reasons[:2])  # Add top 2 news reasons
        
        return factors[:5]  # Limit to top 5 factors
    
    def _assess_overall_risk(self, stock_analysis: Dict, news_analysis: Dict) -> Dict:
        """Assess overall investment risk"""
        risk_factors = []
        risk_level = "Medium"
        
        # Technical risk factors
        if 'risk_analysis' in stock_analysis:
            risk = stock_analysis['risk_analysis']
            volatility = risk.get('volatility', 30)
            beta = risk.get('beta', 1.0)
            
            if volatility > 50:
                risk_factors.append("High volatility")
                risk_level = "High"
            
            if beta > 1.5:
                risk_factors.append("High market sensitivity")
        
        # News risk factors
        if 'overall_recommendation' in news_analysis:
            news_risks = news_analysis['overall_recommendation'].get('risks_to_watch', [])
            risk_factors.extend(news_risks[:2])
        
        # Company-specific risks
        if 'company_news_analysis' in news_analysis:
            company_risks = news_analysis['company_news_analysis'].get('risk_factors', [])
            risk_factors.extend(company_risks[:2])
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors[:4],  # Limit to top 4 risk factors
            "risk_mitigation": [
                "Consider position sizing based on risk tolerance",
                "Monitor news and earnings closely",
                "Set stop-loss levels based on technical analysis"
            ]
        }
    
    def _create_investment_thesis(self, purchase: Dict, stock_analysis: Dict, news_analysis: Dict) -> str:
        """Create a concise investment thesis"""
        try:
            company_name = purchase.get('company_name', 'Company')
            ticker = purchase.get('ticker', 'STOCK')
            insider_name = purchase.get('insider_name', 'CEO')
            value_formatted = purchase.get('value_formatted', 'significant amount')
            
            # Get key metrics
            current_price = stock_analysis.get('current_price', 'N/A')
            recommendation = stock_analysis.get('recommendation', 'Hold')
            
            # Get news sentiment
            news_sentiment = "neutral"
            if 'overall_recommendation' in news_analysis:
                sentiment_label = news_analysis['overall_recommendation'].get('sentiment_label', 'Neutral')
                if 'Bullish' in sentiment_label:
                    news_sentiment = "positive"
                elif 'Bearish' in sentiment_label:
                    news_sentiment = "negative"
            
            thesis = f"""
            {company_name} ({ticker}) presents an investment opportunity based on recent insider activity. 
            {insider_name} recently purchased {value_formatted} worth of shares, signaling confidence in the company's prospects. 
            
            Technical analysis suggests a {recommendation} rating at the current price of ${current_price}. 
            News sentiment analysis indicates a {news_sentiment} outlook based on recent company and market developments.
            
            The combination of insider buying, technical indicators, and news sentiment provides a foundation for 
            investment consideration, though investors should conduct their own due diligence and consider their risk tolerance.
            """
            
            return thesis.strip()
            
        except Exception as e:
            return f"Investment thesis could not be generated due to: {str(e)}"
    
    def _rank_recommendations(self, analyzed_stocks: List[Dict]) -> List[Dict]:
        """Rank and sort stocks by their final recommendation scores"""
        try:
            # Sort by combined score (descending)
            ranked_stocks = sorted(
                analyzed_stocks,
                key=lambda x: x.get('final_recommendation', {}).get('combined_score', 0),
                reverse=True
            )
            
            # Add ranking
            for i, stock in enumerate(ranked_stocks):
                stock['rank'] = i + 1
            
            return ranked_stocks
            
        except Exception as e:
            logger.error(f"Error ranking recommendations: {e}")
            return analyzed_stocks

def main():
    """Main function for command-line usage"""
    print("🚀 Stock Investment Analyzer - CEO Purchase Tracker")
    print("=" * 50)
    
    analyzer = StockInvestmentAnalyzer()
    
    try:
        # Run analysis
        results = analyzer.run_complete_analysis(
            days_back=30,
            min_value=100000,
            max_stocks=5
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Display results
        print(f"\n📊 Analysis completed on {results['analysis_date']}")
        print(f"📈 Found {results['total_ceo_purchases_found']} CEO purchases")
        print(f"🔍 Analyzed {results['stocks_analyzed']} stocks\n")
        
        print("🏆 TOP RECOMMENDATIONS:")
        print("-" * 40)
        
        for stock in results['top_recommendations'][:3]:  # Show top 3
            purchase = stock['insider_purchase']
            final_rec = stock['final_recommendation']
            
            print(f"\n#{stock['rank']} {purchase['ticker']} - {purchase['company_name']}")
            print(f"   📍 Recommendation: {final_rec['final_recommendation']}")
            print(f"   📊 Combined Score: {final_rec['combined_score']}/100")
            print(f"   🎯 Confidence: {final_rec['confidence']}")
            print(f"   💰 CEO Purchase: {purchase['value_formatted']}")
            print(f"   📰 Top Factor: {final_rec['key_factors'][0] if final_rec['key_factors'] else 'N/A'}")
        
        # Save results to file
        with open('analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Full results saved to 'analysis_results.json'")
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")

if __name__ == "__main__":
    main() 