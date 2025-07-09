import openai
import ollama
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from newsapi import NewsApiClient
import json
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self, openai_api_key: str, news_api_key: Optional[str] = None, use_ollama: bool = True):
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.news_api = NewsApiClient(api_key=news_api_key) if news_api_key else None
        self.use_ollama = use_ollama
        
        # Test Ollama connection
        if use_ollama:
            try:
                ollama.list()
                logger.info("Ollama is available - using free local AI")
            except Exception as e:
                logger.warning(f"Ollama not available: {e}. Falling back to OpenAI.")
                self.use_ollama = False
        
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
            
            # Always supplement with enhanced sources for better coverage
            enhanced_articles = self._get_fallback_news(company_name, ticker, days_back)
            news_articles.extend(enhanced_articles)
            
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
        """Enhanced fallback method using multiple free sources"""
        logger.info(f"Using enhanced news sources for {ticker}")
        news_articles = []
        
        # Source 1: DuckDuckGo Search (real-time)
        news_articles.extend(self._get_duckduckgo_news(company_name, ticker))
        
        # Source 2: Yahoo Finance scraping
        news_articles.extend(self._scrape_yahoo_finance_news(ticker))
        
        # Source 3: Google Finance scraping
        news_articles.extend(self._scrape_google_finance_news(ticker))
        
        # Source 4: Financial sentiment from social sources
        news_articles.extend(self._get_financial_sentiment_news(ticker, company_name))
        
        # Source 5: SEC filings and official announcements
        news_articles.extend(self._get_sec_filings_news(ticker, company_name))
        
        # Remove duplicates and filter by relevance
        unique_articles = self._deduplicate_and_filter_news(news_articles, ticker, company_name)
        
        return unique_articles[:10]  # Return top 10 most relevant
    
    def _get_duckduckgo_news(self, company_name: str, ticker: str) -> List[Dict]:
        """Get real-time news using DuckDuckGo search"""
        try:
            articles = []
            search_queries = [
                f"{ticker} stock news",
                f"{company_name} earnings",
                f"{ticker} financial news",
                f"{company_name} stock price"
            ]
            
            with DDGS() as ddgs:
                for query in search_queries[:2]:  # Limit to avoid rate limiting
                    try:
                        results = list(ddgs.news(keywords=query, max_results=5))
                        for result in results:
                            articles.append({
                                'title': result.get('title', ''),
                                'description': result.get('body', '')[:200],
                                'content': result.get('body', ''),
                                'url': result.get('url', ''),
                                'published_at': result.get('date', datetime.now().isoformat()),
                                'source': result.get('source', 'DuckDuckGo News')
                            })
                    except Exception as e:
                        logger.warning(f"Error with DuckDuckGo search '{query}': {e}")
                        continue
            
            logger.info(f"Found {len(articles)} articles from DuckDuckGo for {ticker}")
            return articles
            
        except Exception as e:
            logger.error(f"Error getting DuckDuckGo news: {e}")
            return []
    
    def _scrape_yahoo_finance_news(self, ticker: str) -> List[Dict]:
        """Scrape news from Yahoo Finance"""
        try:
            articles = []
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find news articles (Yahoo Finance structure)
                news_items = soup.find_all('h3', limit=5)  # Limit to avoid overload
                
                for item in news_items:
                    try:
                        title_element = item.find('a')
                        if title_element:
                            title = title_element.get_text(strip=True)
                            if title and len(title) > 10:  # Filter out short/irrelevant titles
                                articles.append({
                                    'title': title,
                                    'description': title[:200],
                                    'content': title,
                                    'url': f"https://finance.yahoo.com{title_element.get('href', '')}",
                                    'published_at': datetime.now().isoformat(),
                                    'source': 'Yahoo Finance'
                                })
                    except Exception as e:
                        continue
            
            logger.info(f"Found {len(articles)} articles from Yahoo Finance for {ticker}")
            return articles
            
        except Exception as e:
            logger.warning(f"Error scraping Yahoo Finance for {ticker}: {e}")
            return []
    
    def _scrape_google_finance_news(self, ticker: str) -> List[Dict]:
        """Scrape news from Google Finance"""
        try:
            articles = []
            # Use DuckDuckGo to search Google Finance specifically
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    keywords=f"site:finance.google.com {ticker}",
                    max_results=3
                ))
                
                for result in results:
                    articles.append({
                        'title': result.get('title', ''),
                        'description': result.get('body', '')[:200],
                        'content': result.get('body', ''),
                        'url': result.get('url', ''),
                        'published_at': result.get('date', datetime.now().isoformat()),
                        'source': 'Google Finance'
                    })
            
            logger.info(f"Found {len(articles)} articles from Google Finance for {ticker}")
            return articles
            
        except Exception as e:
            logger.warning(f"Error getting Google Finance news for {ticker}: {e}")
            return []
    
    def _get_financial_sentiment_news(self, ticker: str, company_name: str) -> List[Dict]:
        """Get financial sentiment and discussions from social sources"""
        try:
            articles = []
            
            # Search for recent discussions and sentiment
            search_queries = [
                f"{ticker} stock reddit",
                f"{ticker} earnings call",
                f"{ticker} analyst rating",
                f"{company_name} stock analysis"
            ]
            
            with DDGS() as ddgs:
                for query in search_queries[:2]:  # Limit searches
                    try:
                        results = list(ddgs.news(keywords=query, max_results=3))
                        for result in results:
                            # Focus on financial analysis and sentiment
                            title = result.get('title', '').lower()
                            body = result.get('body', '').lower()
                            
                            # Score for financial relevance
                            financial_keywords = [
                                'buy', 'sell', 'hold', 'bullish', 'bearish', 'earnings', 
                                'revenue', 'profit', 'analysis', 'rating', 'upgrade', 'downgrade',
                                'target price', 'valuation', 'forecast'
                            ]
                            
                            relevance_score = sum(1 for keyword in financial_keywords 
                                                if keyword in title or keyword in body)
                            
                            if relevance_score >= 2:  # Only include financially relevant content
                                articles.append({
                                    'title': result.get('title', ''),
                                    'description': result.get('body', '')[:200],
                                    'content': result.get('body', ''),
                                    'url': result.get('url', ''),
                                    'published_at': result.get('date', datetime.now().isoformat()),
                                    'source': 'Financial Sentiment',
                                    'relevance_score': relevance_score + 2  # Boost for financial content
                                })
                    
                    except Exception as e:
                        logger.warning(f"Error with sentiment search '{query}': {e}")
                        continue
            
            logger.info(f"Found {len(articles)} financial sentiment articles for {ticker}")
            return articles
            
        except Exception as e:
            logger.warning(f"Error getting financial sentiment for {ticker}: {e}")
            return []
    
    def _get_sec_filings_news(self, ticker: str, company_name: str) -> List[Dict]:
        """Get recent SEC filings and official company announcements"""
        try:
            articles = []
            
            # Search for SEC filings and official announcements
            search_queries = [
                f"{ticker} SEC filing",
                f"{ticker} 10-K 10-Q 8-K",
                f"{company_name} press release",
                f"{ticker} earnings report"
            ]
            
            with DDGS() as ddgs:
                for query in search_queries[:2]:  # Limit to avoid rate limiting
                    try:
                        results = list(ddgs.news(keywords=query, max_results=3))
                        for result in results:
                            title = result.get('title', '').lower()
                            body = result.get('body', '').lower()
                            url = result.get('url', '').lower()
                            
                            # Prioritize official sources
                            official_score = 0
                            if 'sec.gov' in url or 'investor' in url:
                                official_score += 5
                            if any(filing in title for filing in ['10-k', '10-q', '8-k', 'proxy', 'earnings']):
                                official_score += 3
                            if 'press release' in title or 'announces' in title:
                                official_score += 2
                            
                            if official_score >= 2:  # Only include official/semi-official content
                                articles.append({
                                    'title': result.get('title', ''),
                                    'description': result.get('body', '')[:200],
                                    'content': result.get('body', ''),
                                    'url': result.get('url', ''),
                                    'published_at': result.get('date', datetime.now().isoformat()),
                                    'source': 'SEC/Official',
                                    'relevance_score': official_score + 3  # High boost for official content
                                })
                    
                    except Exception as e:
                        logger.warning(f"Error with SEC search '{query}': {e}")
                        continue
            
            logger.info(f"Found {len(articles)} SEC/official articles for {ticker}")
            return articles
            
        except Exception as e:
            logger.warning(f"Error getting SEC filings for {ticker}: {e}")
        return []
    
    def _deduplicate_and_filter_news(self, articles: List[Dict], ticker: str, company_name: str) -> List[Dict]:
        """Remove duplicates and filter for relevance"""
        if not articles:
            return []
        
        # Remove duplicates by title similarity
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower()
            title_words = set(re.findall(r'\w+', title))
            
            # Check if this title is too similar to existing ones
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(re.findall(r'\w+', seen_title))
                if len(title_words & seen_words) / max(len(title_words), len(seen_words)) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate and title:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Filter for relevance (contains ticker or company name)
        relevant_articles = []
        for article in unique_articles:
            title_content = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            if (ticker.lower() in title_content or 
                any(word.lower() in title_content for word in company_name.split() if len(word) > 3)):
                
                # Calculate relevance score
                score = 0
                if ticker.lower() in title_content:
                    score += 3
                if 'earnings' in title_content:
                    score += 2
                if any(word in title_content for word in ['stock', 'shares', 'price', 'buy', 'sell']):
                    score += 1
                
                article['relevance_score'] = score
                relevant_articles.append(article)
        
        # Sort by relevance score (highest first)
        relevant_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"Filtered to {len(relevant_articles)} relevant articles for {ticker}")
        return relevant_articles
    
    def _get_ai_response(self, prompt: str, system_message: str) -> str:
        """Get AI response using Ollama (free) or OpenAI (fallback)"""
        if self.use_ollama:
            try:
                response = ollama.chat(
                    model='llama3.1:8b',
                    messages=[
                        {'role': 'system', 'content': system_message},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                return response['message']['content']
            except Exception as e:
                logger.warning(f"Ollama failed: {e}. Trying OpenAI...")
                self.use_ollama = False  # Switch to OpenAI for this session
        
        # Use OpenAI as fallback
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Using cheaper model
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Both Ollama and OpenAI failed: {e}")
                raise
        else:
            raise Exception("No AI provider available")
    
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
            
            # Get AI analysis using Ollama or OpenAI
            analysis_text = self._get_ai_response(prompt, "You are a financial analyst expert in news sentiment analysis and stock market insights.")
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
            
            recommendation_text = self._get_ai_response(prompt, "You are an expert investment advisor providing clear, actionable recommendations.")
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