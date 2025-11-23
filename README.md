# Asset Management Hackathon Toolkit  
### 3rd Place • McGill FIAM Asset Management Hackathon 2025

🔗 **Interactive Results Site:** [thegoingconcern.vercel.app](https://thegoingconcern.vercel.app)  
🏆 **Official Hackathon Page:** [https://www.mcgill.ca/fiam/hackathon](https://hackathonfinance.com/)

## Overview
- Research + execution stack that earned a 3rd-place finish at the 2025 McGill FIAM Asset Management Hackathon.
- `Model/` produces cross-sectional return forecasts and feature importance analytics.
- `PortfolioBuilder/` turns the model outputs into tradable long/short books with regime-aware risk controls.
- Modular design so teams can swap data loaders, model configs, or portfolio constraints without touching the full stack.

## Tech Stack
- Python 3.9
- LightGBM, Optuna, pandas, NumPy, SciPy
- Jupyter for exploratory research
- Matplotlib for reporting and diagnostics

## Repository Layout
- `Model/main`: production backtest entry point (`main.py`) plus model utilities.
- `Model/src`: reusable components including data loading, feature engineering, tuning, and sentiment scoring.
- `PortfolioBuilder/backtester`: portfolio construction, optimizer, and reporting modules.
- `PortfolioBuilder/run_baseline.py`: end-to-end baseline portfolio script that ingests model scores and emits performance reports.

## Results & Recognition
- 3rd-place finish at the McGill FIAM Asset Management Hackathon 2025.
- Key highlights:
  - Regime-aware lightGBM factor model maintained positive IC across macro cycles.
  - Portfolio builder met institutional diversification and turnover thresholds.
  - Automated reporting captured rolling Sharpe, drawdowns, and capital allocation diagnostics.
