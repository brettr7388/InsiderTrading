import openai
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from newsapi import NewsApiClient
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self, openai_api_key: str, news_api_key: Optional[str] = None):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.news_api = NewsApiClient(api_key=news_api_key) if news_api_key else None
        
    def analyze_stock_news(self, ticker: str, company_name: str, days_back: int = 7) -> Dict:
        """
        Analyze news sentiment and market conditions for a specific stock
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            days_back: Days to look back for news
            
        Returns:
            Dictionary with news analysis and sentiment scores
        """
        try:
            logger.info(f"Analyzing news for {ticker} ({company_name})")
            
            # Get company-specific news
            company_news = self._get_company_news(company_name, ticker, days_back)
            
            # Get market/sector news
            market_news = self._get_market_news(days_back)
            
            # Analyze news with AI
            company_analysis = self._analyze_news_with_ai(company_news, "company", ticker, company_name)
            market_analysis = self._analyze_news_with_ai(market_news, "market", ticker, company_name)
            
            # Get overall investment recommendation
            overall_recommendation = self._get_investment_recommendation(
                company_analysis, market_analysis, ticker, company_name
            )
            
            return {
                "ticker": ticker,
                "company_name": company_name,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "company_news_analysis": company_analysis,
                "market_news_analysis": market_analysis,
                "overall_recommendation": overall_recommendation,
                "news_summary": self._create_news_summary(company_news, market_news)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news for {ticker}: {e}")
            return {"error": str(e)}
    
    def _get_company_news(self, company_name: str, ticker: str, days_back: int) -> List[Dict]:
        """Get company-specific news articles"""
        news_articles = []
        
        try:
            # Use NewsAPI if available
            if self.news_api:
                from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                # Search for company news
                query_terms = [company_name, ticker, f"{ticker} stock"]
                
                for query in query_terms:
                    try:
                        articles = self.news_api.get_everything(
                            q=query,
                            from_param=from_date,
                            language='en',
                            sort_by='relevancy',
                            page_size=10
                        )
                        
                        if articles['articles']:
                            for article in articles['articles'][:5]:  # Limit per query
                                news_articles.append({
                                    'title': article['title'],
                                    'description': article['description'],
                                    'content': article['content'][:500] if article['content'] else '',
                                    'url': article['url'],
                                    'published_at': article['publishedAt'],
                                    'source': article['source']['name']
                                })
                    except Exception as e:
                        logger.warning(f"Error fetching news for query '{query}': {e}")
                        continue
            
            # Fallback: Use alternative news sources or web scraping
            if not news_articles:
                news_articles = self._get_fallback_news(company_name, ticker, days_back)
            
            # Remove duplicates
            unique_articles = []
            seen_titles = set()
            for article in news_articles:
                if article['title'] not in seen_titles:
                    unique_articles.append(article)
                    seen_titles.add(article['title'])
            
            return unique_articles[:10]  # Limit to 10 most relevant articles
            
        except Exception as e:
            logger.error(f"Error getting company news: {e}")
            return []
    
    def _get_market_news(self, days_back: int) -> List[Dict]:
        """Get general market and economic news"""
        try:
            market_articles = []
            
            if self.news_api:
                from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                # Market-related queries
                market_queries = [
                    "stock market",
                    "Federal Reserve",
                    "inflation",
                    "economic outlook",
                    "market volatility"
                ]
                
                for query in market_queries:
                    try:
                        articles = self.news_api.get_everything(
                            q=query,
                            from_param=from_date,
                            language='en',
                            sort_by='relevancy',
                            page_size=5
                        )
                        
                        if articles['articles']:
                            for article in articles['articles'][:3]:  # Limit per query
                                market_articles.append({
                                    'title': article['title'],
                                    'description': article['description'],
                                    'content': article['content'][:500] if article['content'] else '',
                                    'url': article['url'],
                                    'published_at': article['publishedAt'],
                                    'source': article['source']['name']
                                })
                    except Exception as e:
                        logger.warning(f"Error fetching market news for '{query}': {e}")
                        continue
            
            # Remove duplicates
            unique_articles = []
            seen_titles = set()
            for article in market_articles:
                if article['title'] not in seen_titles:
                    unique_articles.append(article)
                    seen_titles.add(article['title'])
            
            return unique_articles[:8]  # Limit to 8 market articles
            
        except Exception as e:
            logger.error(f"Error getting market news: {e}")
            return []
    
    def _get_fallback_news(self, company_name: str, ticker: str, days_back: int) -> List[Dict]:
        """Fallback method to get news when NewsAPI is not available"""
        # This could implement web scraping from financial news sites
        # For now, return empty list
        logger.info("Using fallback news method (placeholder)")
        return []
    
    def _analyze_news_with_ai(self, articles: List[Dict], news_type: str, ticker: str, company_name: str) -> Dict:
        """Use OpenAI to analyze news sentiment and extract insights"""
        try:
            if not articles:
                return {
                    "sentiment_score": 0,
                    "sentiment_label": "Neutral",
                    "key_insights": [],
                    "risk_factors": [],
                    "positive_factors": []
                }
            
            # Prepare news content for analysis
            news_content = self._prepare_news_content(articles)
            
            # Create AI prompt
            prompt = self._create_analysis_prompt(news_content, news_type, ticker, company_name)
            
            # Get AI analysis
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in news sentiment analysis and stock market insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse AI response
            analysis_text = response.choices[0].message.content
            parsed_analysis = self._parse_ai_analysis(analysis_text)
            
            return parsed_analysis
            
        except Exception as e:
            logger.error(f"Error in AI news analysis: {e}")
            return {
                "sentiment_score": 0,
                "sentiment_label": "Neutral",
                "key_insights": ["Error analyzing news"],
                "risk_factors": [],
                "positive_factors": []
            }
    
    def _prepare_news_content(self, articles: List[Dict]) -> str:
        """Prepare news articles for AI analysis"""
        content_parts = []
        
        for i, article in enumerate(articles[:5]):  # Limit to 5 articles
            content = f"Article {i+1}:\n"
            content += f"Title: {article['title']}\n"
            if article['description']:
                content += f"Description: {article['description']}\n"
            if article['content']:
                content += f"Content: {article['content'][:300]}...\n"
            content += f"Source: {article['source']}\n"
            content += f"Date: {article['published_at']}\n\n"
            content_parts.append(content)
        
        return "\n".join(content_parts)
    
    def _create_analysis_prompt(self, news_content: str, news_type: str, ticker: str, company_name: str) -> str:
        """Create prompt for AI analysis"""
        if news_type == "company":
            prompt = f"""
            Analyze the following news articles about {company_name} ({ticker}) and provide investment insights:

            {news_content}

            Please provide your analysis in the following JSON format:
            {{
                "sentiment_score": [number between -100 and 100, where -100 is very bearish, 0 is neutral, 100 is very bullish],
                "sentiment_label": "[Very Bearish/Bearish/Neutral/Bullish/Very Bullish]",
                "key_insights": ["insight 1", "insight 2", "insight 3"],
                "risk_factors": ["risk 1", "risk 2"],
                "positive_factors": ["positive 1", "positive 2"],
                "impact_assessment": "[High/Medium/Low] - how much these news items might impact stock price"
            }}

            Focus on factors that could affect stock price, investor sentiment, and company fundamentals.
            """
        else:  # market news
            prompt = f"""
            Analyze the following market and economic news and assess how it might impact {company_name} ({ticker}):

            {news_content}

            Please provide your analysis in the following JSON format:
            {{
                "sentiment_score": [number between -100 and 100 for overall market sentiment],
                "sentiment_label": "[Very Bearish/Bearish/Neutral/Bullish/Very Bullish]",
                "key_insights": ["market insight 1", "market insight 2"],
                "risk_factors": ["market risk 1", "market risk 2"],
                "positive_factors": ["market positive 1", "market positive 2"],
                "sector_impact": "How these market conditions might specifically affect {company_name}'s sector"
            }}

            Focus on macroeconomic factors, market trends, and sector-specific impacts.
            """
        
        return prompt
    
    def _parse_ai_analysis(self, analysis_text: str) -> Dict:
        """Parse AI analysis response"""
        try:
            # Try to extract JSON from the response
            start = analysis_text.find('{')
            end = analysis_text.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = analysis_text[start:end]
                return json.loads(json_str)
            else:
                # Fallback: parse manually
                return self._manual_parse_analysis(analysis_text)
                
        except json.JSONDecodeError:
            return self._manual_parse_analysis(analysis_text)
    
    def _manual_parse_analysis(self, text: str) -> Dict:
        """Manually parse analysis if JSON parsing fails"""
        # Simple fallback parsing
        sentiment_score = 0
        sentiment_label = "Neutral"
        
        # Basic sentiment detection
        positive_words = ['positive', 'bullish', 'growth', 'strong', 'good', 'up', 'gain']
        negative_words = ['negative', 'bearish', 'decline', 'weak', 'bad', 'down', 'loss']
        
        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)
        
        if positive_count > negative_count:
            sentiment_score = min(50, positive_count * 10)
            sentiment_label = "Bullish"
        elif negative_count > positive_count:
            sentiment_score = max(-50, -negative_count * 10)
            sentiment_label = "Bearish"
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "key_insights": ["Analysis could not be fully parsed"],
            "risk_factors": [],
            "positive_factors": []
        }
    
    def _get_investment_recommendation(self, company_analysis: Dict, market_analysis: Dict, 
                                     ticker: str, company_name: str) -> Dict:
        """Generate overall investment recommendation based on news analysis"""
        try:
            # Combine company and market sentiment
            company_sentiment = company_analysis.get("sentiment_score", 0)
            market_sentiment = market_analysis.get("sentiment_score", 0)
            
            # Weight company news more heavily (70/30 split)
            overall_sentiment = (company_sentiment * 0.7) + (market_sentiment * 0.3)
            
            # Create comprehensive recommendation prompt
            prompt = f"""
            Based on the following analysis for {company_name} ({ticker}), provide an investment recommendation:

            Company News Sentiment: {company_sentiment}/100 ({company_analysis.get('sentiment_label', 'Neutral')})
            Market News Sentiment: {market_sentiment}/100 ({market_analysis.get('sentiment_label', 'Neutral')})
            Combined Sentiment Score: {overall_sentiment:.1f}/100

            Company Insights: {company_analysis.get('key_insights', [])}
            Market Insights: {market_analysis.get('key_insights', [])}
            Risk Factors: {company_analysis.get('risk_factors', []) + market_analysis.get('risk_factors', [])}
            Positive Factors: {company_analysis.get('positive_factors', []) + market_analysis.get('positive_factors', [])}

            Provide a recommendation in JSON format:
            {{
                "recommendation": "[Strong Buy/Buy/Hold/Sell/Strong Sell]",
                "confidence": "[High/Medium/Low]",
                "time_horizon": "[Short-term/Medium-term/Long-term] - best time horizon for this recommendation",
                "key_reasons": ["reason 1", "reason 2", "reason 3"],
                "risks_to_watch": ["risk 1", "risk 2"],
                "price_catalysts": ["catalyst 1", "catalyst 2"] - what could drive price movement
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert investment advisor providing clear, actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            recommendation_text = response.choices[0].message.content
            parsed_recommendation = self._parse_ai_analysis(recommendation_text)
            
            # Add overall sentiment score
            parsed_recommendation["overall_sentiment_score"] = round(overall_sentiment, 1)
            parsed_recommendation["sentiment_label"] = self._get_sentiment_label(overall_sentiment)
            
            return parsed_recommendation
            
        except Exception as e:
            logger.error(f"Error generating investment recommendation: {e}")
            return {
                "recommendation": "Hold",
                "confidence": "Low",
                "overall_sentiment_score": 0,
                "sentiment_label": "Neutral",
                "key_reasons": ["Error generating recommendation"]
            }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score >= 60:
            return "Very Bullish"
        elif score >= 30:
            return "Bullish"
        elif score >= -30:
            return "Neutral"
        elif score >= -60:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def _create_news_summary(self, company_news: List[Dict], market_news: List[Dict]) -> Dict:
        """Create a summary of all news articles"""
        return {
            "company_articles_count": len(company_news),
            "market_articles_count": len(market_news),
            "latest_company_headline": company_news[0]['title'] if company_news else "No company news found",
            "latest_market_headline": market_news[0]['title'] if market_news else "No market news found",
            "news_sources": list(set([article['source'] for article in company_news + market_news]))
        } 