from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import sys
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from main import StockInvestmentAnalyzer
from config import Config

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
CORS(app)

# Global analyzer instance
analyzer = StockInvestmentAnalyzer()

# Store analysis results and progress
analysis_cache = {}
analysis_progress = {}

def make_json_serializable(obj):
    """Convert pandas objects and other non-serializable objects to JSON-serializable formats"""
    if isinstance(obj, dict):
        # Handle dict keys that might be Timestamps or other non-string types
        new_dict = {}
        for key, value in obj.items():
            # Convert non-string keys to strings
            if isinstance(key, (pd.Timestamp, datetime)):
                new_key = str(key)
            elif isinstance(key, (np.integer, np.floating)):
                new_key = key.item()
            else:
                new_key = str(key) if not isinstance(key, (str, int, float, bool)) else key
            new_dict[new_key] = make_json_serializable(value)
        return new_dict
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
        return None
    else:
        return obj

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/start-analysis', methods=['POST'])
def start_analysis():
    """Start the stock analysis process"""
    try:
        data = request.get_json()
        time_frame = data.get('time_frame', 'week')
        min_value = data.get('min_value', 100000)
        max_stocks = data.get('max_stocks', 10)
        
        # Generate session ID for this analysis
        session_id = f"analysis_{int(time.time())}"
        session['analysis_id'] = session_id
        
        # Initialize progress tracking
        analysis_progress[session_id] = {
            'status': 'starting',
            'current_step': 'Initializing analysis...',
            'progress': 0,
            'total_steps': max_stocks + 2,
            'current_stock': None
        }
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(
            target=run_analysis_background,
            args=(session_id, time_frame, min_value, max_stocks)
        )
        analysis_thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Analysis started successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analysis-progress/<session_id>')
def get_analysis_progress(session_id):
    """Get progress of ongoing analysis"""
    progress = analysis_progress.get(session_id, {
        'status': 'not_found',
        'progress': 0,
        'current_step': 'Analysis not found'
    })
    
    return jsonify(progress)

@app.route('/api/analysis-results/<session_id>')
def get_analysis_results(session_id):
    """Get completed analysis results"""
    if session_id in analysis_cache:
        return jsonify({
            'success': True,
            'results': make_json_serializable(analysis_cache[session_id])
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Results not found'
        }), 404

@app.route('/api/stock-details/<ticker>')
def get_stock_details(ticker):
    """Get detailed information for a specific stock"""
    try:
        # Get stock analysis
        stock_analysis = analyzer.stock_analyzer.analyze_stock(ticker)
        
        # Get news analysis
        company_name = stock_analysis.get('company_name', ticker)
        news_analysis = analyzer.news_analyzer.analyze_stock_news(
            ticker=ticker,
            company_name=company_name,
            days_back=7
        )
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'stock_analysis': make_json_serializable(stock_analysis),
            'news_analysis': make_json_serializable(news_analysis)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def run_analysis_background(session_id, time_frame, min_value, max_stocks):
    """Run analysis in background thread with progress updates"""
    try:
        # Update progress
        def update_progress(step, total, message, stock=None):
            analysis_progress[session_id] = {
                'status': 'running',
                'current_step': message,
                'progress': int((step / total) * 100),
                'total_steps': total,
                'current_stock': stock
            }
        
        update_progress(0, max_stocks + 2, "Starting analysis...")
        
        # Step 1: Scrape OpenInsider
        update_progress(1, max_stocks + 2, f"Scraping OpenInsider ({time_frame})...")
        ceo_purchases = analyzer.scraper.get_ceo_purchases(
            min_value=min_value,
            time_frame=time_frame
        )
        
        if not ceo_purchases:
            analysis_progress[session_id] = {
                'status': 'error',
                'error': 'No CEO purchases found in the specified timeframe'
            }
            return
        
        # Step 2: Analyze each stock
        analyzed_stocks = []
        for i, purchase in enumerate(ceo_purchases[:max_stocks]):
            stock_name = f"{purchase['ticker']} ({purchase['company_name']})"
            update_progress(
                i + 2, 
                max_stocks + 2, 
                f"Analyzing {stock_name}...",
                purchase['ticker']
            )
            
            try:
                # Technical and fundamental analysis
                stock_analysis = analyzer.stock_analyzer.analyze_stock(purchase['ticker'])
                
                # News analysis
                news_analysis = analyzer.news_analyzer.analyze_stock_news(
                    ticker=purchase['ticker'],
                    company_name=purchase['company_name'],
                    days_back=Config.NEWS_LOOKBACK_DAYS
                )
                
                # Combine all data
                complete_analysis = {
                    "insider_purchase": purchase,
                    "stock_analysis": stock_analysis,
                    "news_analysis": news_analysis,
                    "final_recommendation": analyzer._generate_final_recommendation(
                        purchase, stock_analysis, news_analysis
                    )
                }
                
                analyzed_stocks.append(complete_analysis)
                
            except Exception as e:
                print(f"Error analyzing {purchase['ticker']}: {e}")
                continue
        
        # Step 3: Finalize results
        update_progress(max_stocks + 2, max_stocks + 2, "Finalizing results...")
        
        final_results = analyzer._rank_recommendations(analyzed_stocks)
        
        results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_ceo_purchases_found": len(ceo_purchases),
            "stocks_analyzed": len(analyzed_stocks),
            "top_recommendations": final_results,
            "analysis_parameters": {
                "time_frame": time_frame,
                "min_purchase_value": min_value,
                "max_stocks_analyzed": max_stocks
            }
        }
        
        # Store results
        analysis_cache[session_id] = make_json_serializable(results)
        
        # Mark as completed
        analysis_progress[session_id] = {
            'status': 'completed',
            'current_step': 'Analysis completed successfully!',
            'progress': 100,
            'total_steps': max_stocks + 2,
            'stocks_found': len(analyzed_stocks)
        }
        
    except Exception as e:
        analysis_progress[session_id] = {
            'status': 'error',
            'error': str(e)
        }

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🚀 Starting Stock Investment Analyzer Web App")
    print("🌐 Open http://localhost:5005 in your browser")
    print("=" * 50)
    print("📊 Features available:")
    print("   - Real-time insider trading analysis")
    print("   - AI-powered news sentiment analysis")
    print("   - Technical and fundamental stock analysis")
    print("   - Investment recommendations")
    print("=" * 50)
    
    app.run(debug=Config.FLASK_DEBUG, host='0.0.0.0', port=5005) 