#!/usr/bin/env python3
"""
Basic test script to test Stock Investment Analyzer without API keys
"""
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scrapers'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

from scrapers.openinsider_scraper import OpenInsiderScraper
from analysis.stock_analyzer import StockAnalyzer

def test_scraper():
    """Test the OpenInsider scraper"""
    print("🧪 Testing OpenInsider Scraper...")
    print("=" * 50)
    
    scraper = OpenInsiderScraper()
    
    # Test with different parameters
    test_cases = [
        (7, 25000),    # 7 days, $25K minimum
        (14, 50000),   # 14 days, $50K minimum
        (30, 100000),  # 30 days, $100K minimum
    ]
    
    for days, min_val in test_cases:
        print(f"\n📅 Testing {days} days back, min value ${min_val:,}")
        try:
            purchases = scraper.get_ceo_purchases(days_back=days, min_value=min_val)
            print(f"✅ Found {len(purchases)} CEO purchases")
            
            if purchases:
                print("📊 Sample purchases:")
                for i, purchase in enumerate(purchases[:3]):  # Show top 3
                    print(f"   {i+1}. {purchase['ticker']} - {purchase['company_name']}")
                    print(f"      💰 {purchase['value_formatted']} by {purchase['insider_name']}")
                    print(f"      📋 {purchase['title']}")
                    print(f"      📅 {purchase['trade_date']}")
                    print()
                break  # Found some data, no need to continue
            else:
                print("ℹ️  No purchases found with these criteria")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return purchases if 'purchases' in locals() else []

def test_stock_analyzer():
    """Test the stock analyzer"""
    print("\n🧪 Testing Stock Analyzer...")
    print("=" * 50)
    
    analyzer = StockAnalyzer()
    
    # Test with popular stocks
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    for ticker in test_stocks:
        print(f"\n📊 Testing {ticker}...")
        try:
            analysis = analyzer.analyze_stock(ticker, period='1mo')
            
            if 'error' not in analysis:
                print(f"✅ Analysis successful for {ticker}")
                print(f"   💰 Current Price: ${analysis.get('current_price', 'N/A')}")
                print(f"   📈 Overall Score: {analysis.get('overall_score', 'N/A')}/100")
                print(f"   🎯 Recommendation: {analysis.get('recommendation', 'N/A')}")
                
                # Show basic metrics if available
                if 'basic_metrics' in analysis:
                    metrics = analysis['basic_metrics']
                    print(f"   📊 1-Day Change: {metrics.get('change_1d', 'N/A')}%")
                    print(f"   📊 7-Day Change: {metrics.get('change_7d', 'N/A')}%")
                
                # Show technical indicators if available
                if 'technical_analysis' in analysis and 'momentum_indicators' in analysis['technical_analysis']:
                    tech = analysis['technical_analysis']['momentum_indicators']
                    print(f"   📉 RSI: {tech.get('rsi', 'N/A')}")
                    print(f"   📈 MACD: {tech.get('macd', 'N/A')}")
                
                return True  # Success, return
            else:
                print(f"❌ Error analyzing {ticker}: {analysis['error']}")
                
        except Exception as e:
            print(f"❌ Exception analyzing {ticker}: {e}")
    
    return False

def test_web_scraping():
    """Test if web scraping is working at all"""
    print("\n🧪 Testing Basic Web Scraping...")
    print("=" * 50)
    
    import requests
    from bs4 import BeautifulSoup
    
    try:
        # Test basic web request
        url = "http://openinsider.com"
        response = requests.get(url, timeout=10)
        print(f"✅ Successfully connected to {url}")
        print(f"   Status Code: {response.status_code}")
        
        # Test parsing
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title')
        if title:
            print(f"   Page Title: {title.text.strip()}")
        
        # Look for the trading table
        table = soup.find('table', {'class': 'tinytable'})
        if table:
            print("✅ Found trading table on the page")
            rows = table.find_all('tr')
            print(f"   Table has {len(rows)} rows")
        else:
            print("❌ Could not find trading table - page structure may have changed")
            
        return True
        
    except Exception as e:
        print(f"❌ Web scraping test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Stock Investment Analyzer - Basic Tests")
    print("=" * 60)
    
    # Test 1: Web scraping basics
    web_works = test_web_scraping()
    
    # Test 2: OpenInsider scraper
    if web_works:
        purchases = test_scraper()
    else:
        purchases = []
    
    # Test 3: Stock analyzer
    stock_works = test_stock_analyzer()
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 50)
    print(f"🌐 Web Scraping: {'✅ Working' if web_works else '❌ Failed'}")
    print(f"📊 OpenInsider Scraper: {'✅ Working' if purchases else '⚠️  No data found'}")
    print(f"📈 Stock Analyzer: {'✅ Working' if stock_works else '❌ Failed'}")
    
    if not web_works:
        print("\n💡 Recommendations:")
        print("   - Check your internet connection")
        print("   - OpenInsider.com might be temporarily unavailable")
        print("   - Consider using a VPN if access is restricted")
    
    if not stock_works:
        print("\n💡 Recommendations:")
        print("   - Yahoo Finance API might be temporarily unavailable")
        print("   - Try again in a few minutes")
        print("   - Check if yfinance library needs updating")
    
    if not purchases and web_works:
        print("\n💡 Note about CEO purchases:")
        print("   - CEO purchases are not constant - there may be quiet periods")
        print("   - Try adjusting the date range or minimum purchase value")
        print("   - The scraper filters for specific executive titles")

if __name__ == "__main__":
    main() 