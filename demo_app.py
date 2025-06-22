#!/usr/bin/env python3
"""
Demo Flask app to showcase Stock Investment Analyzer functionality
without requiring API keys
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
from datetime import datetime

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))
from analysis.stock_analyzer import StockAnalyzer

app = Flask(__name__)
app.secret_key = 'demo-secret-key'
CORS(app)

# Global analyzer instance
analyzer = StockAnalyzer()

@app.route('/')
def index():
    """Demo dashboard page"""
    return render_template('demo.html')

@app.route('/api/analyze-stock', methods=['POST'])
def analyze_stock():
    """Analyze a single stock"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper().strip()
        
        if not ticker:
            return jsonify({
                'success': False,
                'error': 'Please provide a stock ticker'
            }), 400
        
        # Analyze the stock
        analysis = analyzer.analyze_stock(ticker, period='3mo')
        
        if 'error' not in analysis:
            return jsonify({
                'success': True,
                'ticker': ticker,
                'analysis': analysis
            })
        else:
            return jsonify({
                'success': False,
                'error': analysis['error']
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/demo-portfolio')
def demo_portfolio():
    """Get analysis for a demo portfolio of popular stocks"""
    try:
        demo_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META']
        portfolio = []
        
        for ticker in demo_stocks[:5]:  # Limit to 5 for demo
            try:
                analysis = analyzer.analyze_stock(ticker, period='1mo')
                if 'error' not in analysis:
                    portfolio.append({
                        'ticker': ticker,
                        'company_name': analysis.get('company_name', ticker),
                        'current_price': analysis.get('current_price', 0),
                        'overall_score': analysis.get('overall_score', 0),
                        'recommendation': analysis.get('recommendation', 'Hold'),
                        'change_1d': analysis.get('basic_metrics', {}).get('change_1d', 0),
                        'rsi': analysis.get('technical_analysis', {}).get('momentum_indicators', {}).get('rsi', 50)
                    })
            except:
                continue
        
        # Sort by score
        portfolio.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'portfolio': portfolio,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Stock Investment Analyzer Demo")
    print("🌐 Open http://localhost:5001 in your browser")
    print("=" * 50)
    print("📊 This demo showcases:")
    print("   - Real-time stock analysis")
    print("   - Technical indicators (RSI, MACD, etc.)")
    print("   - Investment scoring and recommendations")
    print("   - No API keys required!")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001) 