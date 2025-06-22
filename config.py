import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # OpenInsider settings
    OPENINSIDER_BASE_URL = "http://openinsider.com"
    
    # Stock analysis settings
    ANALYSIS_PERIOD = "1y"  # 1 year of data
    MIN_PURCHASE_VALUE = 100000  # Minimum purchase value to consider
    
    # News analysis settings
    NEWS_LOOKBACK_DAYS = 7
    MAX_NEWS_ARTICLES = 10 