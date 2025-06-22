# 📈 Stock Investment Analyzer - Insider Trading Tracker

An AI-powered stock investment analyzer that combines insider trading data, technical analysis, and news sentiment to provide intelligent investment recommendations.

## 🚀 Features

- **📊 Insider Trading Analysis**: Scrapes OpenInsider.com for recent CEO and insider stock purchases
- **🤖 AI-Powered News Analysis**: Uses ChatGPT to analyze company and market news sentiment
- **📈 Technical Analysis**: RSI, MACD, Bollinger Bands, and other technical indicators
- **💰 Fundamental Analysis**: P/E ratios, profit margins, and financial health metrics
- **🌐 Modern Web Interface**: Real-time progress tracking and responsive design
- **⚡ Multi-Factor Scoring**: Combines insider activity (30%), technical/fundamental analysis (40%), and news sentiment (30%)

## 🏗️ Architecture

```
├── scrapers/           # Web scraping modules
│   └── openinsider_scraper.py
├── analysis/           # Stock analysis modules
│   └── stock_analyzer.py
├── ai/                # AI-powered news analysis
│   └── news_analyzer.py
├── templates/          # Web interface
│   └── index.html
├── app.py             # Flask web application
├── main.py            # Core orchestrator
├── config.py          # Configuration management
└── requirements.txt   # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/brettr7388/InsiderTrading.git
   cd InsiderTrading
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   NEWS_API_KEY=your_news_api_key_here
   ```

   **Get API Keys:**
   - OpenAI API: https://platform.openai.com/api-keys
   - News API: https://newsapi.org/register

## 🚀 Usage

### Web Interface (Recommended)
```bash
python3 app.py
```
Then open http://localhost:5005 in your browser.

### Command Line
```bash
python3 main.py
```

## 📊 How It Works

1. **Insider Data Collection**: Scrapes OpenInsider for recent insider purchases with configurable time frames:
   - Today's purchases
   - Past week purchases  
   - Past month purchases

2. **Stock Analysis**: For each stock with insider activity:
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Fundamental metrics (P/E ratio, profit margins)
   - Current price and volume analysis

3. **AI News Analysis**: Uses ChatGPT to analyze:
   - Company-specific news articles
   - Market and economic conditions
   - Sentiment scoring (-100 to +100)
   - Risk factors and growth catalysts

4. **Final Recommendation**: Combines all factors to generate:
   - Investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
   - Confidence level (High/Medium/Low)
   - Combined score (0-100)
   - Key factors and reasoning

## 🎯 Example Output

```
#1 NUVB - Nuvation Bio Inc
├─ CEO Purchase: $894K by David Hung
├─ Combined Score: 85/100
├─ Recommendation: Strong Buy
├─ Confidence: High
└─ Key Factors:
   • Significant insider purchase indicates confidence
   • Strong technical momentum (RSI: 65)
   • Positive news sentiment (+42)
   • Biotech sector showing growth potential
```

## ⚙️ Configuration

Edit `config.py` to customize:
- Analysis parameters
- API endpoints
- Scoring weights
- Time frames

## 🔧 Troubleshooting

**Port 5000 in use?**
The app runs on port 5005 by default to avoid conflicts with macOS AirPlay.

**NewsAPI rate limits?**
Free NewsAPI accounts have 100 requests/day. The system continues working with limited news data when rate limits are hit.

**No insider purchases found?**
Try different time frames (Today/Week/Month) or lower the minimum purchase value threshold.

## 📈 Performance

- **Analysis Speed**: ~30-60 seconds per stock (depending on API response times)
- **Data Sources**: OpenInsider.com, Yahoo Finance, NewsAPI, OpenAI
- **Accuracy**: Combines multiple data sources for comprehensive analysis

## 🚨 Disclaimer

**This tool is for educational and research purposes only. It is NOT financial advice.**

- Always do your own research before making investment decisions
- Past performance does not guarantee future results
- Consider consulting with a qualified financial advisor
- The authors are not responsible for any financial losses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenInsider.com for insider trading data
- Yahoo Finance for stock data
- OpenAI for AI-powered analysis
- NewsAPI for news data

---

**⭐ If this project helped you, please give it a star!** 