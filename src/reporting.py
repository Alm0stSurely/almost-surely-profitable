"""
Reporting module for weekly and monthly performance reports.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils import _is_finite_number, dump_json_safe, load_valid_daily_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_start_value(value: Any, default: float = 10000.0) -> float:
    """Return a finite, positive start value, falling back to *default*."""
    if _is_finite_number(value) and value > 0:
        return float(value)
    return default


def _day_return_pct(day: Dict) -> Optional[float]:
    """Return the finite total_return_pct of a daily result, or None."""
    value = day.get('portfolio_after', {}).get('total_return_pct')
    return float(value) if _is_finite_number(value) else None


def _best_worst_days(daily_results: List[Dict]) -> tuple:
    """Return (best_day, worst_day) by finite total_return_pct."""
    finite = [
        (day, pct) for day in daily_results
        if (pct := _day_return_pct(day)) is not None
    ]
    if not finite:
        return daily_results[0], daily_results[0]
    best = max(finite, key=lambda x: x[1])[0]
    worst = min(finite, key=lambda x: x[1])[0]
    return best, worst


def _position_pnl_pct(position: Dict) -> Optional[float]:
    """Return the finite unrealized_pnl_pct of a position, or None."""
    value = position.get('unrealized_pnl_pct')
    return float(value) if _is_finite_number(value) else None


def _best_worst_positions(positions: List[Dict]) -> tuple:
    """Return (best_position, worst_position) by finite unrealized_pnl_pct."""
    finite = [
        (pos, pct) for pos in positions
        if (pct := _position_pnl_pct(pos)) is not None
    ]
    if not finite:
        return None, None
    best = max(finite, key=lambda x: x[1])[0]
    worst = min(finite, key=lambda x: x[1])[0]
    return best, worst


class ReportGenerator:
    """Generates periodic trading reports."""
    
    def __init__(self, results_dir: str = "results/daily"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_daily_results(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Load valid daily results from JSON files."""
        results = load_valid_daily_results(str(self.results_dir))

        if start_date:
            results = [r for r in results if r.get('date', '') >= start_date]
        if end_date:
            results = [r for r in results if r.get('date', '') <= end_date]

        return results
    
    def generate_weekly_report(self, year: int, week: int) -> Dict:
        """
        Generate weekly performance report.
        
        Args:
            year: Year number
            week: ISO week number
        
        Returns:
            Weekly report dictionary
        """
        # Calculate week date range using ISO week calendar
        # ISO week 1 is the week containing the first Thursday of the year
        start_of_week = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u")
        end_of_week = start_of_week + timedelta(days=6)
        
        start_str = start_of_week.strftime("%Y-%m-%d")
        end_str = end_of_week.strftime("%Y-%m-%d")
        
        logger.info(f"Generating weekly report for {year}-W{week:02d} ({start_str} to {end_str})")
        
        # Load daily results
        daily_results = self.load_daily_results(start_str, end_str)
        
        if not daily_results:
            logger.warning(f"No data found for week {year}-W{week:02d}")
            return {}
        
        # Calculate metrics
        raw_start_value = daily_results[0].get('portfolio_after', {}).get('total_value', 0)
        start_value = _safe_start_value(raw_start_value)
        raw_end_value = daily_results[-1].get('portfolio_after', {}).get('total_value', 0)
        end_value = raw_end_value if _is_finite_number(raw_end_value) else None

        if end_value is None:
            weekly_return = None
        else:
            weekly_return = (end_value / start_value) - 1

        # Count trades
        total_trades = sum(
            len(day.get('executed_trades', []))
            for day in daily_results
        )

        # Get positions at end of week
        final_positions = daily_results[-1].get('portfolio_after', {}).get('positions', [])

        # Calculate daily returns
        daily_returns = []
        for i, day in enumerate(daily_results):
            if i == 0:
                continue
            prev_value = daily_results[i-1].get('portfolio_after', {}).get('total_value', 0)
            curr_value = day.get('portfolio_after', {}).get('total_value', 0)
            if _is_finite_number(prev_value) and prev_value > 0 and _is_finite_number(curr_value) and curr_value > 0:
                daily_returns.append((curr_value / prev_value) - 1)

        volatility = pd.Series(daily_returns).std() if daily_returns else 0

        # Find best/worst days
        best_day, worst_day = _best_worst_days(daily_results)
        
        report = {
            "period": f"{year}-W{week:02d}",
            "period_type": "weekly",
            "start_date": start_str,
            "end_date": end_str,
            "start_value": start_value,
            "end_value": end_value if end_value is not None else None,
            "weekly_return_pct": weekly_return * 100 if weekly_return is not None else None,
            "total_trades": total_trades,
            "num_trading_days": len(daily_results),
            "volatility": volatility * 100 if _is_finite_number(volatility) else 0,
            "best_day": {
                "date": best_day.get('date'),
                "return_pct": best_day.get('portfolio_after', {}).get('total_return_pct', 0)
            },
            "worst_day": {
                "date": worst_day.get('date'),
                "return_pct": worst_day.get('portfolio_after', {}).get('total_return_pct', 0)
            },
            "final_positions": [
                {
                    "ticker": p.get('ticker'),
                    "quantity": p.get('quantity'),
                    "market_value": p.get('market_value'),
                    "unrealized_pnl_pct": p.get('unrealized_pnl_pct')
                }
                for p in final_positions
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def generate_monthly_report(self, year: int, month: int) -> Dict:
        """
        Generate monthly performance report.
        
        Args:
            year: Year number
            month: Month number (1-12)
        
        Returns:
            Monthly report dictionary
        """
        start_str = f"{year}-{month:02d}-01"
        
        # Calculate end of month (last day, inclusive)
        if month == 12:
            next_month_start = datetime(year + 1, 1, 1)
        else:
            next_month_start = datetime(year, month + 1, 1)
        end_of_month = next_month_start - timedelta(days=1)
        end_str = end_of_month.strftime("%Y-%m-%d")
        
        logger.info(f"Generating monthly report for {year}-{month:02d}")
        
        # Load daily results
        daily_results = self.load_daily_results(start_str, end_str)
        
        if not daily_results:
            logger.warning(f"No data found for {year}-{month:02d}")
            return {}

        # Calculate metrics
        raw_start_value = daily_results[0].get('portfolio_after', {}).get('total_value', 0)
        start_value = _safe_start_value(raw_start_value)
        raw_end_value = daily_results[-1].get('portfolio_after', {}).get('total_value', 0)
        end_value = raw_end_value if _is_finite_number(raw_end_value) else None

        if end_value is None:
            monthly_return = None
        else:
            monthly_return = (end_value / start_value) - 1

        # Calculate vs benchmarks (if available)
        spy_return = self._get_benchmark_return('SPY', start_str, end_str)
        cac_return = self._get_benchmark_return('^FCHI', start_str, end_str)

        vs_spy_pct = (monthly_return - spy_return) * 100 if monthly_return is not None and spy_return is not None and _is_finite_number(spy_return) else None
        vs_cac_pct = (monthly_return - cac_return) * 100 if monthly_return is not None and cac_return is not None and _is_finite_number(cac_return) else None

        # Count trades
        trades = []
        for day in daily_results:
            trades.extend(day.get('executed_trades', []))

        # Group trades by ticker
        trades_by_ticker = {}
        for trade in trades:
            ticker = trade.get('ticker')
            if ticker not in trades_by_ticker:
                trades_by_ticker[ticker] = []
            trades_by_ticker[ticker].append(trade)

        # Find best/worst positions
        final_positions = daily_results[-1].get('portfolio_after', {}).get('positions', [])
        best_position, worst_position = _best_worst_positions(final_positions)

        report = {
            "period": f"{year}-{month:02d}",
            "period_type": "monthly",
            "start_date": start_str,
            "end_date": daily_results[-1].get('date'),
            "start_value": start_value,
            "end_value": end_value if end_value is not None else None,
            "monthly_return_pct": monthly_return * 100 if monthly_return is not None else None,
            "vs_spy_pct": vs_spy_pct,
            "vs_cac_pct": vs_cac_pct,
            "total_trades": len(trades),
            "trades_by_ticker": {
                ticker: len(t) for ticker, t in trades_by_ticker.items()
            },
            "num_trading_days": len(daily_results),
            "best_position": {
                "ticker": best_position.get('ticker'),
                "pnl_pct": best_position.get('unrealized_pnl_pct')
            } if best_position else None,
            "worst_position": {
                "ticker": worst_position.get('ticker'),
                "pnl_pct": worst_position.get('unrealized_pnl_pct')
            } if worst_position else None,
            "final_positions_count": len(final_positions),
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def _get_benchmark_return(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[float]:
        """Get benchmark return for comparison."""
        try:
            import yfinance as yf
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                # Newer yfinance returns MultiIndex columns; flatten to scalar
                close_values = data['Close'].values.flatten()
                start_price = float(close_values[0])
                end_price = float(close_values[-1])
                if not _is_finite_number(start_price) or start_price <= 0 or not _is_finite_number(end_price):
                    return None
                return (end_price / start_price) - 1
        except Exception:
            pass
        return None
    
    def save_report(self, report: Dict, output_dir: str = "results/reports") -> str:
        """Save report to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        period = report.get('period', 'unknown')
        period_type = report.get('period_type', 'report')
        
        filename = f"{period_type}_{period}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            dump_json_safe(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
        return str(filepath)
    
    def print_report(self, report: Dict):
        """Print report to console in readable format."""
        if not report:
            print("No report data available")
            return
        
        period = report.get('period', 'Unknown')
        period_type = report.get('period_type', 'report')
        
        print("\n" + "="*70)
        print(f"{period_type.upper()} REPORT: {period}")
        print("="*70)
        
        print(f"\nPeriod: {report.get('start_date')} to {report.get('end_date')}")
        print(f"Trading Days: {report.get('num_trading_days', 0)}")
        
        print(f"\n--- Performance ---")
        start_value = report.get('start_value')
        end_value = report.get('end_value')
        start_str = f"€{start_value:,.2f}" if _is_finite_number(start_value) else "n/a"
        end_str = f"€{end_value:,.2f}" if _is_finite_number(end_value) else "n/a"
        print(f"Start Value: {start_str}")
        print(f"End Value:   {end_str}")
        
        return_key = 'weekly_return_pct' if period_type == 'weekly' else 'monthly_return_pct'
        return_val = report.get(return_key)
        if return_val is None:
            print("Return:      n/a")
        else:
            print(f"Return:      {return_val:+.2f}%")
        
        # Benchmark comparison (monthly only)
        if 'vs_spy_pct' in report and report['vs_spy_pct'] is not None:
            print(f"vs SPY:      {report['vs_spy_pct']:+.2f}%")
        if 'vs_cac_pct' in report and report['vs_cac_pct'] is not None:
            print(f"vs CAC:      {report['vs_cac_pct']:+.2f}%")
        
        print(f"\n--- Trading Activity ---")
        print(f"Total Trades: {report.get('total_trades', 0)}")
        
        if 'trades_by_ticker' in report:
            print("Trades by Ticker:")
            for ticker, count in report['trades_by_ticker'].items():
                print(f"  {ticker}: {count}")
        
        if 'best_position' in report and report['best_position']:
            bp = report['best_position']
            pnl = bp.get('pnl_pct')
            pnl_str = f"{pnl:+.2f}%" if pnl is not None else "n/a"
            print(f"\nBest Position: {bp.get('ticker')} ({pnl_str})")
        
        if 'worst_position' in report and report['worst_position']:
            wp = report['worst_position']
            pnl = wp.get('pnl_pct')
            pnl_str = f"{pnl:+.2f}%" if pnl is not None else "n/a"
            print(f"Worst Position: {wp.get('ticker')} ({pnl_str})")
        
        if 'best_day' in report and report['best_day']:
            bd = report['best_day']
            ret = bd.get('return_pct')
            ret_str = f"{ret:+.2f}%" if ret is not None else "n/a"
            print(f"\nBest Day: {bd.get('date')} ({ret_str})")
        
        if 'worst_day' in report and report['worst_day']:
            wd = report['worst_day']
            ret = wd.get('return_pct')
            ret_str = f"{ret:+.2f}%" if ret is not None else "n/a"
            print(f"Worst Day: {wd.get('date')} ({ret_str})")
        
        print("="*70)


def generate_latest_report():
    """Generate report for current week/month."""
    generator = ReportGenerator()
    
    now = datetime.now()
    
    # Generate weekly report
    year, week, _ = now.isocalendar()
    weekly_report = generator.generate_weekly_report(year, week)
    
    if weekly_report:
        generator.print_report(weekly_report)
        generator.save_report(weekly_report)
    
    # Generate monthly report if end of month or first day
    if now.day <= 5:  # First few days of month
        month = now.month - 1 if now.month > 1 else 12
        year = now.year if now.month > 1 else now.year - 1
        monthly_report = generator.generate_monthly_report(year, month)
        
        if monthly_report:
            generator.print_report(monthly_report)
            generator.save_report(monthly_report)


if __name__ == "__main__":
    generate_latest_report()
