import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Dict
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenInsiderScraper:
    def __init__(self):
        self.base_url = "http://openinsider.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_insider_purchases(self, days_back: int = 30, min_value: float = 50000, time_frame: str = "week") -> List[Dict]:
        """
        Scrape OpenInsider for recent insider stock purchases (CEOs, Directors, Officials, 10% owners)
        
        Args:
            days_back: Number of days to look back (for additional filtering)
            min_value: Minimum purchase value to consider
            time_frame: Time frame to search ("day", "week", "month")
            
        Returns:
            List of dictionaries containing purchase data
        """
        try:
            # Map time frames to correct OpenInsider URLs
            time_frame_urls = {
                "day": f"{self.base_url}/top-officer-purchases-of-the-day",
                "week": f"{self.base_url}/top-officer-purchases-of-the-week", 
                "month": f"{self.base_url}/top-officer-purchases-of-the-month"
            }
            
            # Default to week if invalid time frame provided
            if time_frame not in time_frame_urls:
                time_frame = "week"
                
            url = time_frame_urls[time_frame]
            
            logger.info(f"Scraping OpenInsider ({time_frame}): {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table with insider trading data
            table = soup.find('table', {'class': 'tinytable'})
            if not table:
                logger.error("Could not find trading table on OpenInsider")
                return []
            
            purchases = []
            rows = table.find_all('tr')[1:]  # Skip header row
            
            logger.info(f"Processing {len(rows)} rows from OpenInsider ({time_frame})")
            
            for row in rows[:100]:  # Process more rows to get more data
                try:
                    cells = row.find_all('td')
                    if len(cells) < 13:  # Need at least 13 cells based on structure
                        continue
                    
                    # Extract data from cells based on actual structure
                    # Based on the test: ['X', 'Filing Date', 'Trade Date', 'Ticker', 'Company Name', 
                    # 'Insider Name', 'Title', 'Trade Type', 'Price', 'Qty', 'Owned', 'ΔOwn', 'Value', ...]
                    filing_date = cells[1].text.strip()
                    trade_date = cells[2].text.strip()
                    ticker = cells[3].text.strip()
                    company_name = cells[4].text.strip()
                    insider_name = cells[5].text.strip()
                    title = cells[6].text.strip()
                    trade_type = cells[7].text.strip()
                    price = cells[8].text.strip()
                    qty = cells[9].text.strip()
                    owned = cells[10].text.strip() if len(cells) > 10 else ""
                    delta_own = cells[11].text.strip() if len(cells) > 11 else ""
                    value = cells[12].text.strip() if len(cells) > 12 else ""
                    
                    # Filter for relevant insider titles (expanded criteria)
                    if not self._is_relevant_insider(title):
                        continue
                    
                    # Filter for purchases only
                    if not self._is_purchase(trade_type):
                        continue
                    
                    # Parse and validate data
                    parsed_data = self._parse_trade_data(
                        filing_date, trade_date, ticker, company_name,
                        insider_name, title, price, qty, value, owned, delta_own
                    )
                    
                    if parsed_data and parsed_data['value'] >= min_value:
                        # For additional filtering, check if within custom date range
                        if time_frame == "month" and days_back < 30:
                            # If user wants less than 30 days but selected month view, filter further
                            if not self._is_within_date_range(parsed_data['trade_date'], days_back):
                                continue
                        elif time_frame == "week" and days_back < 7:
                            # If user wants less than 7 days but selected week view, filter further
                            if not self._is_within_date_range(parsed_data['trade_date'], days_back):
                                continue
                        
                        purchases.append(parsed_data)
                        logger.info(f"Found purchase: {ticker} - {parsed_data['value_formatted']} by {insider_name}")
                            
                except Exception as e:
                    logger.warning(f"Error parsing row: {e}")
                    continue
            
            logger.info(f"Found {len(purchases)} qualifying insider purchases")
            
            # Combine duplicate tickers
            combined_purchases = self._combine_duplicate_tickers(purchases)
            logger.info(f"After combining duplicates: {len(combined_purchases)} unique tickers")
            
            return sorted(combined_purchases, key=lambda x: x['value'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error scraping OpenInsider: {e}")
            return []
    
    # Keep backward compatibility
    def get_ceo_purchases(self, days_back: int = 30, min_value: float = 100000, time_frame: str = "week") -> List[Dict]:
        """Legacy method name - now calls get_insider_purchases"""
        return self.get_insider_purchases(days_back, min_value, time_frame)
    
    def _is_relevant_insider(self, title: str) -> bool:
        """Check if the title indicates a relevant insider (CEO, Director, Officer, 10% owner, etc.)"""
        if not title:
            return False
            
        title_lower = title.lower()
        
        # Expanded list of relevant insider titles
        relevant_indicators = [
            # C-Suite executives
            'ceo', 'chief executive officer', 'chief exec officer',
            'cfo', 'chief financial officer', 'chief finance officer',
            'coo', 'chief operating officer', 'chief operations officer',
            'cto', 'chief technology officer', 'chief tech officer',
            'cmo', 'chief marketing officer',
            'cso', 'chief strategy officer',
            'chief', 'president', 'vice president', 'vp',
            
            # Board and governance
            'chairman', 'chair', 'director', 'board', 'trustee',
            
            # Senior management
            'founder', 'co-founder', 'owner', 'partner',
            'executive', 'senior', 'managing', 'general manager',
            
            # Ownership indicators
            '10%', 'ten percent', 'beneficial owner', 'principal',
            
            # Other key roles
            'officer', 'secretary', 'treasurer', 'controller'
        ]
        
        return any(indicator in title_lower for indicator in relevant_indicators)
    
    def _is_purchase(self, trade_type: str) -> bool:
        """Check if the trade type indicates a purchase"""
        if not trade_type:
            return False
        
        trade_lower = trade_type.lower()
        purchase_indicators = ['p - purchase', 'purchase', 'p', 'buy']
        
        return any(indicator in trade_lower for indicator in purchase_indicators)
    
    def _parse_trade_data(self, filing_date: str, trade_date: str, ticker: str,
                         company_name: str, insider_name: str, title: str,
                         price: str, qty: str, value: str, owned: str = "", delta_own: str = "") -> Dict:
        """Parse and clean trade data"""
        try:
            # Clean and parse numeric values
            price_clean = re.sub(r'[,$]', '', price.replace('$', ''))
            qty_clean = re.sub(r'[,+]', '', qty)
            value_clean = re.sub(r'[,$+]', '', value.replace('$', ''))
            
            # Parse value - handle different formats like +$1,312,425
            value_num = 0
            if value_clean:
                try:
                    if 'K' in value_clean.upper():
                        value_num = float(value_clean.upper().replace('K', '')) * 1000
                    elif 'M' in value_clean.upper():
                        value_num = float(value_clean.upper().replace('M', '')) * 1000000
                    else:
                        value_num = float(value_clean)
                except:
                    # If we can't parse value, try to calculate from price * quantity
                    try:
                        price_val = float(price_clean) if price_clean else 0
                        qty_val = float(qty_clean) if qty_clean else 0
                        value_num = price_val * qty_val
                    except:
                        value_num = 0
            
            # Parse trade date - handle different formats
            parsed_trade_date = self._parse_date(trade_date)
            
            return {
                'filing_date': filing_date,
                'trade_date': parsed_trade_date,
                'ticker': ticker.upper(),
                'company_name': company_name,
                'insider_name': insider_name,
                'title': title,
                'price': float(price_clean) if price_clean else 0,
                'quantity': int(float(qty_clean)) if qty_clean else 0,
                'value': value_num,
                'value_formatted': self._format_currency(value_num),
                'owned_after': owned,
                'ownership_change': delta_own
            }
        except Exception as e:
            logger.warning(f"Error parsing trade data: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date string into consistent format"""
        try:
            # Handle different date formats
            if '-' in date_str and len(date_str) == 10:  # YYYY-MM-DD
                return date_str
            elif '/' in date_str:  # MM/DD/YYYY
                return date_str
            else:
                return date_str
        except:
            return date_str
    
    def _is_within_date_range(self, trade_date: str, days_back: int) -> bool:
        """Check if trade date is within the specified range"""
        try:
            # Try different date formats
            trade_dt = None
            
            if '-' in trade_date and len(trade_date) == 10:  # YYYY-MM-DD
                trade_dt = datetime.strptime(trade_date, '%Y-%m-%d')
            elif '/' in trade_date:  # MM/DD/YYYY
                trade_dt = datetime.strptime(trade_date, '%m/%d/%Y')
            
            if trade_dt:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                return trade_dt >= cutoff_date
            else:
                return True  # Include if we can't parse date
        except:
            return True  # Include if we can't parse date
    
    def _format_currency(self, value: float) -> str:
        """Format currency value for display"""
        if value >= 1000000:
            return f"${value/1000000:.1f}M"
        elif value >= 1000:
            return f"${value/1000:.0f}K"
        else:
            return f"${value:,.0f}"
    
    def _combine_duplicate_tickers(self, purchases: List[Dict]) -> List[Dict]:
        """Combine multiple purchases of the same ticker into a single record"""
        if not purchases:
            return purchases
        
        # Group purchases by ticker
        ticker_groups = {}
        for purchase in purchases:
            ticker = purchase['ticker']
            if ticker not in ticker_groups:
                ticker_groups[ticker] = []
            ticker_groups[ticker].append(purchase)
        
        combined_purchases = []
        
        for ticker, group in ticker_groups.items():
            if len(group) == 1:
                # Single purchase, keep as is
                combined_purchases.append(group[0])
            else:
                # Multiple purchases, combine them
                logger.info(f"Combining {len(group)} purchases for {ticker}")
                
                # Take the most recent purchase as the base
                most_recent = max(group, key=lambda x: x['trade_date'])
                
                # Calculate combined values
                total_value = sum(p['value'] for p in group)
                total_quantity = sum(p['quantity'] for p in group)
                
                # Get all unique insider names
                insider_names = list(set(p['insider_name'] for p in group))
                combined_insider_name = insider_names[0] if len(insider_names) == 1 else f"{insider_names[0]} (+{len(insider_names)-1} others)"
                
                # Get date range
                dates = [p['trade_date'] for p in group]
                earliest_date = min(dates)
                latest_date = max(dates)
                date_display = latest_date if earliest_date == latest_date else f"{earliest_date} to {latest_date}"
                
                # Create combined record
                combined_purchase = {
                    'filing_date': most_recent['filing_date'],
                    'trade_date': date_display,
                    'ticker': ticker,
                    'company_name': most_recent['company_name'],
                    'insider_name': combined_insider_name,
                    'title': most_recent['title'],
                    'price': sum(p['price'] * p['quantity'] for p in group) / total_quantity if total_quantity > 0 else 0,  # Weighted average price
                    'quantity': total_quantity,
                    'value': total_value,
                    'value_formatted': self._format_currency(total_value),
                    'owned_after': most_recent['owned_after'],
                    'ownership_change': most_recent['ownership_change'],
                    'purchase_count': len(group)  # Track how many purchases were combined
                }
                
                combined_purchases.append(combined_purchase)
                logger.info(f"Combined {ticker}: {len(group)} purchases totaling {combined_purchase['value_formatted']}")
        
        return combined_purchases
    
    def get_ticker_details(self, ticker: str) -> Dict:
        """Get additional details for a specific ticker from OpenInsider"""
        try:
            url = f"{self.base_url}/screener?s={ticker}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract additional details if available
            details = {
                'ticker': ticker,
                'recent_activity': self._extract_recent_activity(soup),
                'insider_summary': self._extract_insider_summary(soup)
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting ticker details for {ticker}: {e}")
            return {'ticker': ticker}
    
    def _extract_recent_activity(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract recent insider activity from ticker page"""
        try:
            table = soup.find('table', {'class': 'tinytable'})
            if not table:
                return []
            
            activities = []
            rows = table.find_all('tr')[1:6]  # Get first 5 rows
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 8:
                    activity = {
                        'date': cells[1].text.strip(),
                        'insider': cells[2].text.strip(),
                        'title': cells[3].text.strip(),
                        'trade_type': cells[4].text.strip(),
                        'price': cells[5].text.strip(),
                        'quantity': cells[6].text.strip(),
                        'value': cells[7].text.strip()
                    }
                    activities.append(activity)
            
            return activities
            
        except Exception as e:
            logger.warning(f"Error extracting recent activity: {e}")
            return []
    
    def _extract_insider_summary(self, soup: BeautifulSoup) -> Dict:
        """Extract insider trading summary statistics"""
        try:
            # This would extract summary statistics if available
            # Implementation would depend on OpenInsider's page structure
            return {
                'total_insiders': 0,
                'recent_purchases': 0,
                'recent_sales': 0,
                'net_activity': 'Unknown'
            }
        except:
            return {} 