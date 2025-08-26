import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging
from scipy.stats import linregress, chi2, norm
from scipy.linalg import solve_toeplitz
import cvxpy as cp
from arch import arch_model
from sklearn.covariance import LedoitWolf
from functools import lru_cache
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed, retry_if_exception_type
import random
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from statsmodels.tsa.api import Holt
from sklearn.preprocessing import RobustScaler
import statsmodels.tsa.stattools as smt
import plotly.express as px
import pandas_datareader.data as web
from hmmlearn.hmm import GaussianHMM
import statsmodels.tsa.api as sm
# Add these to your main script's import section
import arch
from sklearn.metrics import mean_absolute_error
from plotly.subplots import make_subplots
# --- Basic Configuration ---
logging.basicConfig(filename='stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="Quantitative Portfolio Analysis", layout="wide")

# --- Data Structures and Mappings ---
sector_etf_map = {
    'Technology': 'XLK', 'Consumer Cyclical': 'XLY', 'Communication Services': 'XLC',
    'Financial Services': 'XLF', 'Industrials': 'XLI', 'Basic Materials': 'XLB',
    'Energy': 'XLE', 'Real Estate': 'XLRE', 'Healthcare': 'XLV',
    'Consumer Defensive': 'XLP', 'Utilities': 'XLU'
}
factor_etfs = ['QQQ', 'IWM', 'DIA', 'EEM', 'EFA', 'IVE', 'IVW', 'MDY', 'MTUM', 'RSP', 'SPY', 'QUAL', 'SIZE', 'USMV']
etf_list = list(set(sector_etf_map.values()) | set(factor_etfs))
default_weights = {
    "(Dividends + Share Buyback) / FCF": 5.0, "CapEx / (Depr + Amor)": 4.5, "Debt Ratio": 6.0,
    "Gross Profit Margin": 7.5, "Inventory Turnover": 4.5, "Net Profit Margin": 6.5,
    "Return on Assets": 6.0, "Assets Growth TTM": 5.5, "Assets Growth QOQ": 7.0,
    "Assets Growth YOY": 5.5, "FCF Growth TTM": 5.0, "FCF Growth QOQ": 6.0,
    "FCF Growth YOY": 6.0, "Dividend Yield": 4.0, "FCF Yield": 6.5, "Operating Margin": 4.5,
    "Liabilities to Equity Ratio": 4.5, "Earnings Per Share, Diluted": 4.5, "Dividend Payout Ratio": 0,
    "Return On Invested Capital": 6.0, "Piotroski F-Score": 6.5, "Operating Leverage": 5.5,
    "Cash Return On Invested Capital": 6.0, "Asset Turnover": 4.5, "Current Ratio": 6.0,
    "Dividends / FCF": 5.5, "Interest Coverage": 2.5, "Quick Ratio": 4.5, "Return on Equity": 7.0,
    "Share Buyback / FCF": 5.5, "Earnings Growth TTM": 5.5, "Earnings Growth QOQ": 6.5,
    "Earnings Growth YOY": 6.5, "Sales Growth TTM": 5.5, "Sales Growth QOQ": 5.5,
    "Sales Growth YOY": 6.5, "Earnings Yield": 6.5, "Market-Cap": 4.5, "P/E": 4.5, "P/Sales": 4.5,
    "Free Cash Flow": 3.5, "Free Cash Flow to Net Income": 5.5, "Sales Per Share": 4.5,
    "Free Cash Flow Per Share": 5.5, "Sharpe Ratio": 18.0, "Relative Z-Score": 18.0,
    "Market Correlation": 4.5, "Correlation_Score": 4.5, "Trend": 5.5, "Q Score": 6.5,
    "Coverage Score": 5.5, "Beta_to_SPY": -10, "GARCH_Vol": 5.5, "Vision": 4.5,
    "Value Factor": 6.0, "Profitability_Factor": 6.0, "Log_Log_Utility": 5.0,
    "Vol_Autocorr": 5.0, "Log_Log_Sharpe": 10.0, "Stop_Loss_Impact": 2.5, "AR_Coeff": 4.0,
    "Tangible_Book_Value": 4.5, "Return_On_Tangible_Equity": 6.0, "Insider_Ownership_Ratio": 6.0,
    "Earnings_Growth_Rate_5y": 5.5, "Peter_Lynch_Fair_Value": 4.5, "Peter_Lynch_Fair_Value_Upside": 6.0,
    "Revenue_Growth_Rate_5y": 5.5, "Meets_Triple_Rule": 3.0, "Return_21d": 4.0, "Return_63d": 4.5,
    "Return_126d": 5.0, "Return_252d": 5.5, "Audit Risk": 3.0, "Board Risk": 3.0, "Compensation Risk": 3.0,
    "Shareholder Rights Risk": 3.0, "Overall Risk": 4.0, "Institutional Ownership Ratio": 6.0,
    "Hurst Exponent (Lo's R/S)": 6.0, "5-Day Return": 3.0,
    "10-Day Return": 3.5
}
# --- FULLY CORRECTED 'columns' LIST ---
# --- FULLY CORRECTED 'columns' LIST ---
# (Existing columns are preserved, and new ones are added)
columns = [
    "Ticker", "Name","Market_Cap", "Dividend_Yield", "PE_Ratio", "EPS_Diluted",
    "Sales_Per_Share", "FCF_Per_Share", "Asset_Turnover", "CapEx_to_DepAmor",
    "Current_Ratio", "Debt_Ratio", "Dividends_to_FCF", "Gross_Profit_Margin",
    "Interest_Coverage", "Inventory_Turnover", "Net_Profit_Margin", "Quick_Ratio",
    "ROA", "ROE", "Share_Buyback_to_FCF", "Dividends_Plus_Buyback_to_FCF",
    "Assets_Growth_TTM", "Earnings_Growth_TTM", "FCF_Growth_TTM", "Sales_Growth_TTM",
    "Assets_Growth_QOQ", "Earnings_Growth_QOQ", "FCF_Growth_QOQ", "Sales_Growth_QOQ",
    "Assets_Growth_YOY", "Earnings_Growth_YOY", "FCF_Growth_YOY", "Sales_Growth_YOY",
    "FCF_Yield", "RD_to_Gross_Profit_2Y_Avg", "Earnings_Yield", "PS_Ratio", "Free_Cash_Flow", "Operating_Margin",
    "FCF_to_Net_Income", "Liabilities_to_Equity", "Dividend_Payout_Ratio",
    "Operating_Leverage", "Piotroski_F-Score", "ROIC", "Cash_ROIC",
    "Dollar_Volume_90D", "Score", "Sharpe_Ratio", "Relative_Z_Score",
    "Rolling_Market_Correlation", "Correlation_Score", "Trend", "Q_Score",
    "Coverage_Score", "Risk_Flag", "Beta_to_SPY", "GARCH_Vol", "Vision",
    "Best_Factor", "Value_Factor", "Profitability_Factor", "Log_Log_Utility",
    "Vol_Autocorr", "Log_Log_Sharpe", "Stop_Loss_Impact", "AR_Coeff", "Sector", "Tangible_Book_Value", "Return_On_Tangible_Equity",
    "Insider_Ownership_Ratio", "Earnings_Growth_Rate_5y",
    "Peter_Lynch_Fair_Value", "Peter_Lynch_Fair_Value_Upside",
    "Revenue_Growth_Rate_5y", "Meets_Triple_Rule", "Return_21d", "Return_63d", "Return_126d", "Return_252d",
    "Audit_Risk", "Board_Risk", "Compensation_Risk", "Shareholder_Rights_Risk",
    "Overall_Risk", "Institutional_Ownership_Ratio", "Hurst_Exponent",
    "Momentum", "Growth", "Earnings_Price_Ratio","Book_to_Market_Ratio",
    "Beta_Down_Market", "Beta_Up_Market", "Carry", "Relative_Carry", "CS_Mean_Reversion",
    "Return_5d", "Return_10d",
    # --- START: NEWLY ADDED FACTORS ---
    'LogMktCap', 'LogMktCapCubed', 'IndRel_TobinQ', 'ROEStddev20Q', 'ROAStddev20Q', 'ShareChg',
    'EQ-style', 'AccrualRatioBS', 'SusGrwRate', 'AstGrwth', 'HG-style', 'IndRel_SusGrwRate',
    '3YAvgAnnSalesGrw', 'Sz-style', 'AdjAccruals', 'Chg1YCACL', '6MAvgChg1MRecC', 'CashCycle',
    'ChgDeprCapex', 'DA', 'IndRel_AccrualRatioCF', 'UnexpectedRecChg', 'AE-style',
    'IndRel_CashCycle', 'IndRel_DA', 'WCAccruals', 'IndRel_WCAccruals', 'TotalAccruals',
    'IndRel_AdjAccruals', '5YRel_QuickRatio', 'IndRel_ShareChg', 'AdjIntCov', 'IntCovRatio',
    '5YRel_CACL', 'IndRel_ChgSalesMargin', 'ROA60MSlope', 'ChgSalesMargin', 'LTDE',
    'IndRel_LTDA', '5YRel_ChgSalesMargin', 'InvTurn', '5YRel_LTDA', 'WCapToSales',
    'IndRel_IntCovRatio', '5YRel_AdjAccruals', 'Chg3YEPS', 'SGPToSales', 'UnexpectedInvChg',
    'IndRel_LTDE', 'SalesToEPSChg', '3YAvgAnnEPSGrw', 'DIVIDENDGROWTH', 'InvToAsset',
    'EPSEstDispFY2C', 'IndRel_QuickRatio', 'RONA', 'DivperShareGrowth', 'AstAdjChg3YEPS',
    'Chg3YCF', 'IndRel_InvToAsset', 'IndRel_InvTurn', 'IndRel_EPSEstDispFY1C',
    '5YRel_WCA', 'GPMargin', 'IndRel_PAdjChg3YSales', 'IndRel_SGPToSales',
    'IndRel_SGPToSalesChg1Y', 'Chg3YOPM', 'AstAdjChg3YCF', 'WCapToAst', '5YRel_ROA', '5YRel_OEA',
    'IndRel_CACL', 'WCA', 'WCA_2', '5YRel_GPMargin', '5YRel_WCTurn', 'FinLev', 'LogAssets',
    '5YRel_PTMargin', '5YRel_FCFP', '5YRel_OPM', 'Chg1YLTDA', 'IndRel_TotalAccruals', 'CACL',
    'CFIC', 'EstDiffC', 'EBITMargin', 'IndRel_DepToCapex', '5YRel_EBITMargin', 'IndRel_WCTurn',
    'IndRel_WCA', 'CashAst', 'AstAdjChg3YFCF', 'EPSEstDispFY1C', 'DepToCapex',
    'IndRel_ROE', 'IndRel_DebtChg1Y', 'Chg3YFCF', '5YRel_ROE', '5YRel_ROIC', '5YRel_CashAst',
    'CFAst', 'PAdjChg3YFCF', 'AdjEPSNumRevFY1C', 'IndRel_InvToAst', 'IndRel_CashAst',
    'IndRel_ROA', 'IndRel_OEA', 'CashRatio', 'IndRel_NetProfitMargin', 'AstAdjChg3YOCF',
    'AstAdjChg1YFCF', 'IndRel_ROIC', 'IndRel_ChgDeprCapex', 'IndRel_CapAcqRatio',
    'IndRel_FCFEV', 'IndRel_FCFP', 'IndRel_GPMargin', 'OEA', 'AdjEPSNumRevFY2C',
    'PAdjChg3YEPS', 'MVEToTL', 'IndRel_WCapToSales', 'SolvencyRatio', 'ROA_2', 'AstAdjChg1YOCF',
    'IndRel_CapExToAst', 'IndRel_PAdjChg1YFCF', 'CFEqt', 'Chg3YAstTo', 'IndRel_PTMargin',
    'IndRel_Chg1YOPM', 'IndRel_RecTurn', 'AstAdjChg1YCF', '5YRel_Chg1YEPS', 'EbitToAst_2',
    'IndRel_EBITMargin', 'IndRel_OPM', 'Chg1YFCF', 'IndRel_Chg1YOCF', 'IndRel_PAdjChg1YSales',
    'EPSNumRevFY1C', 'AstAdjChg1YEPS', 'SalesToInvCap', 'IndRel_PAdjChg1YEPS', 'PAdjChg3YOCF',
    'FCFEV', 'NIStab', 'Chg3YOCF', '5YRel_RecTurn', 'IndRel_AssetTurn', 'LogTTMSales',
    'IndRel_EPSToSalesChg1Y', 'FCFP', 'Altman_ZScore', 'Chg1YGPMargin', 'Chg1YOPM',
    'IndRel_Chg1YFCF', '6MChgTgtPrc', 'IndRel_PAdjChg1YCF', 'IndRel_FCFEqt', 'IndRel_CFROIC',
    'Chg1YEPS', 'PAdjChg3YCF', 'CapAcqRatio', 'IndRel_Chg1YEPS', 'FCFEqt',
    'Rev3MFY1C', 'PAdjChg1YEPS', 'CashBurn', 'PAdjChg3YSales',
    'IndRel_FwdFCFPC', 'Chg1YROA', '6MTTMSalesMom', 'Chg1YCF',
    'Rev3MFY2C', 'IndRel_NCAP', 'RecTurn', 'BuyToSellRecLess3MSMA', 'CFROIC', '3MSalesMom',
    'OCFEqt', 'IndRel_SEV', 'AdjRevMagC', 'BuyToSellRecLess3MEMA', 'IndRel_CashEV',
    'ChgATO_2', 'IndRel_OCFP', 'Chg1YAstTo', 'OCFRatio', 'IndRel_OCFEV', 'OCFAst_2',
    'RevMagFY1C', 'IndRel_SP', 'FwdFCFPC', '6MChgTgtPrcGap', 'PAdjChg1YSales', 'IndRel_CFP',
    'IndRel_EP', 'IndRel_AstP', 'IndRel_OEP', 'IndRel_CFEV', 'IndRel_EBITDAP', 'REToAst',
    'IndRel_GFP', 'IndRel_PTIP', 'REToAst_2', '5YRel_OCFP', 'IndRel_EBITDAEV', '4WChgFwd12MEPS',
    'CurLiaP', 'NCAP', 'IndRel_BP', 'IndRel_CashP', 'IndRel_DivP', '8WChgFwd12MEPS', 'OCFP',
    '5YRel_CFEV', 'AdjEBITP', 'OCFEV', 'OEP', '5YRel_CFP', '5YRel_EBITDAP', 'CashEV', 'CashP',
    'AstP', 'GFP', 'SP', 'EBITDAEV', 'SEV', 'BP', '5YRel_AstP', '5YRel_BP', 'STO', 'AnnVol1M',
    '5DVolSig', 'PM1M', 'PrcTo260DL', 'IndRel_PM1M', 'Chg1YTurnover', '14DayRSI', '10DMACD',
    'IndRel_PM5D', 'MaxRetPayoff', 'PM5D', 'IndRel_50DVolSig', 'PM6M', '50DVolSig', 'AnnVol12M',
    'IndRel_PM6M', 'AdjSTO_6M', 'RSI26W', 'IndRel_MaxRetPayoff', 'PM9M', 'IndRel_PM9M',
    '24MResRtnVar', '90DCV', 'PM-style', 'Alpha60M', '20DStochastic', 'PA52WL20DLag',
    '5DMoneyFlowVol', 'SharpeRatio', 'PSlopeSERR_26W', 'LogUnadjPrice', '50To200PrcRatio',
    'StdErr180D', 'PrcTo52WH', 'PRatio15To36W', '39WRtnLag4W', 'IndRel_PM12M1M', 'PM12M1M',
    '52WSlope', 'BookLev', 'Alpha18M6MPChg', 'YoYChgDA', 'Alpha12M6MPChg', 'Beta60M',
    'VolAdjRtn12M', 'RelPrStr_12M', 'Alpha36M6MPChg', 'CVVolPrc60D', 'CVVolPrc30D',
    'CVVolPrc20D', 'STO_6M', 'RskAdjRS', '130DMinRtn', 'HL1M', 'Chg1YAmihud', '4To52WPrcOsc',
    'HL52W', 'Amihud', 'Vol-style'
    # --- END: NEWLY ADDED FACTORS ---
]

# --- FULLY CORRECTED 'METRIC_NAME_MAP' ---
# (Existing mappings are preserved, and new ones are added)
# --- FULLY CORRECTED 'METRIC_NAME_MAP' ---
METRIC_NAME_MAP = {
    # --- Existing Mappings ---
    "Dividends_Plus_Buyback_to_FCF": "(Dividends + Share Buyback) / FCF", "CapEx_to_DepAmor": "CapEx / (Depr + Amor)",
    "ROA": "Return on Assets", "ROE": "Return on Equity", "ROIC": "Return On Invested Capital", "Cash_ROIC": "Cash Return On Invested Capital",
    "PS_Ratio": "P/Sales", "EPS_Diluted": "Earnings Per Share, Diluted", "Free_Cash_Flow": "Free Cash Flow", "FCF_to_Net_Income": "Free Cash Flow to Net Income",
    "Sales_Per_Share": "Sales Per Share", "FCF_Per_Share": "Free Cash Flow Per Share", "Piotroski_F-Score": "Piotroski F-Score", "PE_Ratio": "P/E",
    "Dividend_Yield": "Dividend Yield", "FCF_Yield": "FCF Yield", "Operating_Margin": "Operating Margin", "Liabilities_to_Equity": "Liabilities to Equity Ratio",
    "Dividend_Payout_Ratio": "Dividend Payout Ratio", "Operating_Leverage": "Operating Leverage", "Assets_Growth_TTM": "Assets Growth TTM",
    "Earnings_Growth_TTM": "Earnings Growth TTM", "FCF_Growth_TTM": "FCF Growth TTM", "Sales_Growth_TTM": "Sales Growth TTM", "Assets_Growth_QOQ": "Assets Growth QOQ",
    "Earnings_Growth_QOQ": "Earnings Growth QOQ", "FCF_Growth_QOQ": "FCF Growth QOQ", "Sales_Growth_QOQ": "Sales Growth QOQ", "Assets_Growth_YOY": "Assets Growth YOY",
    "Earnings_Growth_YOY": "Earnings Growth YOY", "FCF_Growth_YOY": "FCF Growth YOY", "Sales_Growth_YOY": "Sales Growth YOY", "Earnings_Yield": "Earnings Yield",
    "Market_Cap": "Market-Cap", "Debt_Ratio": "Debt Ratio", "Gross_Profit_Margin": "Gross Profit Margin", "Inventory_Turnover": "Inventory Turnover",
    "Net_Profit_Margin": "Net Profit Margin", "Asset_Turnover": "Asset Turnover", "Current_Ratio": "Current Ratio", "Dividends_to_FCF": "Dividends / FCF",
    "Interest_Coverage": "Interest Coverage", "Quick_Ratio": "Quick Ratio", "Share_Buyback_to_FCF": "Share Buyback / FCF", "Sharpe_Ratio": "Sharpe Ratio",
    "Relative_Z_Score": "Relative Z-Score", "Rolling_Market_Correlation": "Market Correlation", "Correlation_Score": "Correlation_Score", "Trend": "Trend",
    "Q_Score": "Q Score", "Coverage_Score": "Coverage Score", "Risk_Flag": "Risk_Flag", "Beta_to_SPY": "Beta_to_SPY", "GARCH_Vol": "GARCH_Vol",
    "Vision": "Vision", "Best_Factor": "Best_Factor", "Value_Factor": "Value Factor", "Profitability_Factor": "Profitability Factor",
    "Log_Log_Utility": "Log_Log_Utility", "Vol_Autocorr": "Vol_Autocorr", "Log_Log_Sharpe": "Log_Log_Sharpe", "Stop_Loss_Impact": "Stop_Loss_Impact",
    "AR_Coeff": "AR_Coeff", "Tangible_Book_Value": "Tangible Book Value", "Return_On_Tangible_Equity": "Return on Tangible Equity",
    "Insider_Ownership_Ratio": "Insider Ownership Ratio", "Earnings_Growth_Rate_5y": "5-Year Earnings Growth Rate", "Peter_Lynch_Fair_Value": "Peter Lynch Fair Value",
    "Peter_Lynch_Fair_Value_Upside": "Peter Lynch Fair Value Upside", "Revenue_Growth_Rate_5y": "5-Year Revenue Growth Rate", "Meets_Triple_Rule": "Meets Triple Rule",
    "Return_21d": "21-Day Return", "Return_63d": "63-Day Return", "Return_126d": "126-Day Return", "Return_252d": "252-Day Return",
    "Audit_Risk": "Audit Risk", "Board_Risk": "Board Risk", "Compensation_Risk": "Compensation Risk", "Shareholder_Rights_Risk": "Shareholder Rights Risk",
    "Overall_Risk": "Overall Risk", "Institutional_Ownership_Ratio": "Institutional Ownership Ratio", "Hurst_Exponent": "Hurst Exponent (Lo's R/S)",
    "Momentum": "Momentum", "Growth": "Growth", "Earnings_Price_Ratio": "Earnings-Price Ratio (E/P)", "Book_to_Market_Ratio": "Book-to-Market Ratio (B/M)",
    "Beta_Down_Market": "Down-Market Beta", "Beta_Up_Market": "Up-Market Beta", "Carry": "Shareholder Yield (Carry)", "Relative_Carry": "Relative Carry vs Sector",
    "CS_Mean_Reversion": "Cross-Sectional Mean Reversion","Return_5d": "5-Day Return", "Return_10d": "10-Day Return",

    # --- START: NEW MAPPINGS (COMPLETE LIST) ---
    'LogMktCap': 'Log Market Cap', 'LogMktCapCubed': 'Log Market Cap Cubed', 'IndRel_TobinQ': 'Industry-Relative Tobin Q', 'ROEStddev20Q': 'ROE Stdev (20 Qtrs)',
    'ROAStddev20Q': 'ROA Stdev (20 Qtrs)', 'ShareChg': 'Share Count Change (YoY)', 'EQ-style': 'Equity Style Factor', 'AccrualRatioBS': 'Accrual Ratio (Balance Sheet)',
    'SusGrwRate': 'Sustainable Growth Rate', 'AstGrwth': 'Asset Growth (YoY)', 'HG-style': 'High Growth Style Factor', 'IndRel_SusGrwRate': 'Industry-Relative Sustainable Growth Rate',
    '3YAvgAnnSalesGrw': '3Y Avg Annual Sales Growth', 'Sz-style': 'Size Style Factor', 'AdjAccruals': 'Adjusted Accruals', 'Chg1YCACL': '1Y Change in Current Assets/Liabilities Ratio',
    '6MAvgChg1MRecC': '6M Avg Change in 1M Analyst Recommendations', 'CashCycle': 'Cash Conversion Cycle', 'ChgDeprCapex': 'Change in Depreciation to Capex', 'DA': 'Discretionary Accruals',
    'IndRel_AccrualRatioCF': 'Industry-Relative Accrual Ratio (Cash Flow)', 'UnexpectedRecChg': 'Unexpected Receivables Change', 'AE-style': 'Analyst Expectations Style Factor',
    'IndRel_CashCycle': 'Industry-Relative Cash Cycle', 'IndRel_DA': 'Industry-Relative Discretionary Accruals', 'WCAccruals': 'Working Capital Accruals', 'IndRel_WCAccruals': 'Industry-Relative Working Capital Accruals',
    'TotalAccruals': 'Total Accruals', 'IndRel_AdjAccruals': 'Industry-Relative Adjusted Accruals', '5YRel_QuickRatio': '5-Year Relative Quick Ratio', 'IndRel_ShareChg': 'Industry-Relative Share Change',
    'AdjIntCov': 'Adjusted Interest Coverage', 'IntCovRatio': 'Interest Coverage Ratio', '5YRel_CACL': '5-Year Relative Current Assets/Liabilities', 'IndRel_ChgSalesMargin': 'Industry-Relative Change in Sales Margin',
    'ROA60MSlope': 'ROA 60-Month Slope', 'ChgSalesMargin': 'Change in Sales Margin', 'LTDE': 'Long-Term Debt to Equity', 'IndRel_LTDA': 'Industry-Relative Long-Term Debt to Assets',
    '5YRel_ChgSalesMargin': '5-Year Relative Change in Sales Margin', 'InvTurn': 'Inventory Turnover', '5YRel_LTDA': '5-Year Relative Long-Term Debt to Assets', 'WCapToSales': 'Working Capital to Sales',
    'IndRel_IntCovRatio': 'Industry-Relative Interest Coverage', '5YRel_AdjAccruals': '5-Year Relative Adjusted Accruals', 'Chg3YEPS': '3Y EPS Change', 'SGPToSales': 'Sales & General Profit to Sales',
    'UnexpectedInvChg': 'Unexpected Inventory Change', 'IndRel_LTDE': 'Industry-Relative Long-Term Debt to Equity', 'SalesToEPSChg': 'Sales to EPS Change Ratio', '3YAvgAnnEPSGrw': '3Y Avg Annual EPS Growth',
    'DIVIDENDGROWTH': 'Dividend Growth', 'InvToAsset': 'Inventory to Assets', 'EPSEstDispFY2C': 'EPS Estimate Dispersion (FY2)', 'IndRel_QuickRatio': 'Industry-Relative Quick Ratio',
    'RONA': 'Return on Net Assets', 'DivperShareGrowth': 'Dividend Per Share Growth', 'AstAdjChg3YEPS': 'Asset-Adj 3Y EPS Change', 'Chg3YCF': '3Y Cash Flow Change',
    'IndRel_InvToAsset': 'Industry-Relative Inventory to Assets', 'IndRel_InvTurn': 'Industry-Relative Inventory Turnover', 'IndRel_EPSEstDispFY1C': 'Industry-Relative EPS Estimate Dispersion (FY1)',
    '5YRel_WCA': '5-Year Relative Working Capital to Assets', 'GPMargin': 'Gross Profit Margin', 'IndRel_PAdjChg3YSales': 'Industry-Relative Price-Adj 3Y Sales Change', 'IndRel_SGPToSales': 'Industry-Relative SGP to Sales',
    'IndRel_SGPToSalesChg1Y': 'Industry-Relative 1Y SGP to Sales Change', 'Chg3YOPM': '3Y Operating Margin Change', 'AstAdjChg3YCF': 'Asset-Adj 3Y Cash Flow Change', 'WCapToAst': 'Working Capital to Assets',
    '5YRel_ROA': '5-Year Relative ROA', '5YRel_OEA': '5-Year Relative Operating Earnings to Assets', 'IndRel_CACL': 'Industry-Relative Current Assets/Liabilities', 'WCA': 'Working Capital to Assets', 'WCA_2': 'Working Capital to Assets (Alt)',
    '5YRel_GPMargin': '5-Year Relative Gross Profit Margin', '5YRel_WCTurn': '5-Year Relative Working Capital Turnover', 'FinLev': 'Financial Leverage', 'LogAssets': 'Log of Total Assets',
    '5YRel_PTMargin': '5-Year Relative Pre-Tax Margin', '5YRel_FCFP': '5-Year Relative FCF to Price', '5YRel_OPM': '5-Year Relative Operating Margin', 'Chg1YLTDA': '1Y Change in Long-Term Debt to Assets',
    'IndRel_TotalAccruals': 'Industry-Relative Total Accruals', 'CACL': 'Current Assets to Current Liabilities', 'CFIC': 'Cash Flow to Invested Capital', 'EstDiffC': 'Estimate Difference (Consensus)',
    'EBITMargin': 'EBIT Margin', 'IndRel_DepToCapex': 'Industry-Relative Depreciation to Capex', '5YRel_EBITMargin': '5-Year Relative EBIT Margin', 'IndRel_WCTurn': 'Industry-Relative Working Capital Turnover',
    'IndRel_WCA': 'Industry-Relative Working Capital to Assets', 'CashAst': 'Cash to Assets', 'AstAdjChg3YFCF': 'Asset-Adj 3Y FCF Change', 'EPSEstDispFY1C': 'EPS Estimate Dispersion (FY1)',
    'DepToCapex': 'Depreciation to Capex', 'IndRel_ROE': 'Industry-Relative ROE', 'IndRel_DebtChg1Y': 'Industry-Relative 1Y Debt Change', 'Chg3YFCF': '3Y FCF Change',
    '5YRel_ROE': '5-Year Relative ROE', '5YRel_ROIC': '5-Year Relative ROIC', '5YRel_CashAst': '5-Year Relative Cash to Assets', 'CFAst': 'Cash Flow to Assets',
    'PAdjChg3YFCF': 'Price-Adj 3Y FCF Change', 'AdjEPSNumRevFY1C': 'Adjusted EPS Num Revisions (FY1)', 'IndRel_InvToAst': 'Industry-Relative Inventory to Assets', 'IndRel_CashAst': 'Industry-Relative Cash to Assets',
    'IndRel_ROA': 'Industry-Relative ROA', 'IndRel_OEA': 'Industry-Relative Operating Earnings to Assets', 'CashRatio': 'Cash Ratio', 'IndRel_NetProfitMargin': 'Industry-Relative Net Profit Margin',
    'AstAdjChg3YOCF': 'Asset-Adj 3Y OCF Change', 'AstAdjChg1YFCF': 'Asset-Adj 1Y FCF Change', 'IndRel_ROIC': 'Industry-Relative ROIC', 'IndRel_ChgDeprCapex': 'Industry-Relative Change in Depr to Capex',
    'IndRel_CapAcqRatio': 'Industry-Relative Capital Acquisition Ratio', 'IndRel_FCFEV': 'Industry-Relative FCF to EV', 'IndRel_FCFP': 'Industry-Relative FCF to Price', 'IndRel_GPMargin': 'Industry-Relative Gross Profit Margin',
    'OEA': 'Operating Earnings to Assets', 'AdjEPSNumRevFY2C': 'Adjusted EPS Num Revisions (FY2)', 'PAdjChg3YEPS': 'Price-Adj 3Y EPS Change', 'MVEToTL': 'Market Value of Equity to Total Liabilities',
    'IndRel_WCapToSales': 'Industry-Relative Working Capital to Sales', 'SolvencyRatio': 'Solvency Ratio', 'ROA_2': 'Return on Assets (Alt)', 'AstAdjChg1YOCF': 'Asset-Adj 1Y OCF Change',
    'IndRel_CapExToAst': 'Industry-Relative Capex to Assets', 'IndRel_PAdjChg1YFCF': 'Industry-Relative Price-Adj 1Y FCF Change', 'CFEqt': 'Cash Flow to Equity', 'Chg3YAstTo': '3Y Asset Turnover Change',
    'IndRel_PTMargin': 'Industry-Relative Pre-Tax Margin', 'IndRel_Chg1YOPM': 'Industry-Relative 1Y Operating Margin Change', 'IndRel_RecTurn': 'Industry-Relative Receivables Turnover', 'AstAdjChg1YCF': 'Asset-Adj 1Y Cash Flow Change',
    '5YRel_Chg1YEPS': '5-Year Relative 1Y EPS Change', 'EbitToAst_2': 'EBIT to Assets (Alt)', 'IndRel_EBITMargin': 'Industry-Relative EBIT Margin', 'IndRel_OPM': 'Industry-Relative Operating Margin',
    'Chg1YFCF': '1Y FCF Change', 'IndRel_Chg1YOCF': 'Industry-Relative 1Y OCF Change', 'IndRel_PAdjChg1YSales': 'Industry-Relative Price-Adj 1Y Sales Change', 'EPSNumRevFY1C': 'EPS Num Revisions (FY1)',
    'AstAdjChg1YEPS': 'Asset-Adj 1Y EPS Change', 'SalesToInvCap': 'Sales to Invested Capital', 'IndRel_PAdjChg1YEPS': 'Industry-Relative Price-Adj 1Y EPS Change', 'PAdjChg3YOCF': 'Price-Adj 3Y OCF Change',
    'FCFEV': 'Free Cash Flow to Enterprise Value', 'NIStab': 'Net Income Stability', 'Chg3YOCF': '3Y OCF Change', '5YRel_RecTurn': '5-Year Relative Receivables Turnover', 'IndRel_AssetTurn': 'Industry-Relative Asset Turnover',
    'LogTTMSales': 'Log of TTM Sales', 'IndRel_EPSToSalesChg1Y': 'Industry-Relative EPS to Sales 1Y Change', 'FCFP': 'Free Cash Flow to Price', 'Chg1YGPMargin': '1Y Gross Profit Margin Change',
    'Chg1YOPM': '1Y Operating Margin Change', 'IndRel_Chg1YFCF': 'Industry-Relative 1Y FCF Change', '6MChgTgtPrc': '6M Change in Target Price', 'IndRel_PAdjChg1YCF': 'Industry-Relative Price-Adj 1Y CF Change',
    'IndRel_FCFEqt': 'Industry-Relative FCF to Equity', 'IndRel_CFROIC': 'Industry-Relative CFROIC', 'Chg1YEPS': '1Y EPS Change', 'PAdjChg3YCF': 'Price-Adj 3Y CF Change',
    'CapAcqRatio': 'Capital Acquisition Ratio', 'IndRel_Chg1YEPS': 'Industry-Relative 1Y EPS Change', 'FCFEqt': 'Free Cash Flow to Equity', 'Rev3MFY1C': '3M Revision in FY1 Sales Estimate',
    'PAdjChg1YEPS': 'Price-Adj 1Y EPS Change', 'CashBurn': 'Cash Burn Rate', 'PAdjChg3YSales': 'Price-Adj 3Y Sales Change', 'IndRel_FwdFCFPC': 'Industry-Relative Fwd FCF Per Share',
    'Chg1YROA': '1Y ROA Change', '6MTTMSalesMom': '6M TTM Sales Momentum', 'Chg1YCF': '1Y Cash Flow Change', 'Piotroski_FScore': 'Piotroski F-Score', 'Rev3MFY2C': '3M Revision in FY2 Sales Estimate',
    'IndRel_NCAP': 'Industry-Relative Net CAPEX', 'RecTurn': 'Receivables Turnover', 'BuyToSellRecLess3MSMA': 'Buy/Sell Ratio (3M SMA)', 'CFROIC': 'Cash Flow Return on Invested Capital',
    '3MSalesMom': '3M Sales Momentum', 'OCFEqt': 'Operating Cash Flow to Equity', 'IndRel_SEV': 'Industry-Relative Sales to EV', 'AdjRevMagC': 'Adjusted Revision Magnitude (Consensus)',
    'BuyToSellRecLess3MEMA': 'Buy/Sell Ratio (3M EMA)', 'IndRel_CashEV': 'Industry-Relative Cash to EV', 'ChgATO_2': 'Asset Turnover Change (Alt)', 'IndRel_OCFP': 'Industry-Relative OCF to Price',
    'Chg1YAstTo': '1Y Asset Turnover Change', 'OCFRatio': 'Operating Cash Flow Ratio', 'IndRel_OCFEV': 'Industry-Relative OCF to EV', 'OCFAst_2': 'OCF to Assets (Alt)',
    'RevMagFY1C': 'Revision Magnitude (FY1)', 'IndRel_SP': 'Industry-Relative Sales to Price', 'FwdFCFPC': 'Forward FCF Per Share', '6MChgTgtPrcGap': '6M Change Target Price Gap',
    'PAdjChg1YSales': 'Price-Adj 1Y Sales Change', 'IndRel_CFP': 'Industry-Relative CF to Price', 'IndRel_EP': 'Industry-Relative Earnings to Price', 'IndRel_AstP': 'Industry-Relative Assets to Price',
    'IndRel_OEP': 'Industry-Relative Operating Earnings to Price', 'IndRel_CFEV': 'Industry-Relative CF to EV', 'IndRel_EBITDAP': 'Industry-Relative EBITDA to Price', 'REToAst': 'Retained Earnings to Assets',
    'IndRel_GFP': 'Industry-Relative Gross Profit to Price', 'IndRel_PTIP': 'Industry-Relative Pre-Tax Income to Price', 'REToAst_2': 'Retained Earnings to Assets (Alt)', '5YRel_OCFP': '5-Year Relative OCF to Price',
    'IndRel_EBITDAEV': 'Industry-Relative EBITDA to EV', '4WChgFwd12MEPS': '4W Change in Fwd 12M EPS', 'CurLiaP': 'Current Liabilities to Price', 'NCAP': 'Net CAPEX',
    'IndRel_BP': 'Industry-Relative Book to Price', 'IndRel_CashP': 'Industry-Relative Cash to Price', 'IndRel_DivP': 'Industry-Relative Dividend to Price', '8WChgFwd12MEPS': '8W Change in Fwd 12M EPS',
    'OCFP': 'Operating Cash Flow to Price', '5YRel_CFEV': '5-Year Relative CF to EV', 'AdjEBITP': 'Adjusted EBIT to Price', 'OCFEV': 'Operating Cash Flow to Enterprise Value',
    'OEP': 'Operating Earnings to Price', '5YRel_CFP': '5-Year Relative CF to Price', '5YRel_EBITDAP': '5-Year Relative EBITDA to Price', 'CashEV': 'Cash to Enterprise Value',
    'CashP': 'Cash to Price', 'AstP': 'Assets to Price', 'GFP': 'Gross Profit to Price', 'SP': 'Sales to Price', 'EBITDAEV': 'EBITDA to Enterprise Value', 'SEV': 'Sales to Enterprise Value',
    'BP': 'Book to Price', '5YRel_AstP': '5-Year Relative Assets to Price', '5YRel_BP': '5-Year Relative Book to Price', 'STO': 'Share Turnover',
    'AnnVol1M': 'Annualized Volatility (1M)', '5DVolSig': '5-Day Volume Signal', 'PM1M': 'Price Momentum (1M)', 'PrcTo260DL': 'Price to 260-Day Low', 'IndRel_PM1M': 'Industry-Relative 1M Momentum',
    'Chg1YTurnover': '1Y Change in Share Turnover', '14DayRSI': '14-Day RSI', '10DMACD': '10-Day MACD', 'IndRel_PM5D': 'Industry-Relative 5D Momentum', 'MaxRetPayoff': 'Max Return Payoff',
    'PM5D': 'Price Momentum (5D)', 'IndRel_50DVolSig': 'Industry-Relative 50D Volume Signal', 'PM6M': 'Price Momentum (6M)', '50DVolSig': '50-Day Volume Signal',
    'AnnVol12M': 'Annualized Volatility (12M)', 'IndRel_PM6M': 'Industry-Relative 6M Momentum', 'AdjSTO_6M': 'Adjusted Share Turnover (6M)', 'RSI26W': '26-Week RSI',
    'IndRel_MaxRetPayoff': 'Industry-Relative Max Return Payoff', 'PM9M': 'Price Momentum (9M)', 'IndRel_PM9M': 'Industry-Relative 9M Momentum', '24MResRtnVar': '24M Residual Return Variance',
    '90DCV': '90-Day Coefficient of Variation', 'PM-style': 'Momentum Style Factor', 'Alpha60M': 'Alpha (60M)', '20DStochastic': '20-Day Stochastic', 'PA52WL20DLag': 'Price vs 52W-Low (20D Lag)',
    '5DMoneyFlowVol': '5-Day Money Flow Volume', 'SharpeRatio': 'Sharpe Ratio', 'PSlopeSERR_26W': 'Price Slope Std Error (26W)', 'LogUnadjPrice': 'Log Unadjusted Price',
    '50To200PrcRatio': '50-Day to 200-Day Price Ratio', 'StdErr180D': 'Standard Error of Price (180D)', 'PrcTo52WH': 'Price to 52-Week High', 'PRatio15To36W': 'Price Ratio (15W to 36W)',
    '39WRtnLag4W': '39-Week Return (4W Lag)', 'IndRel_PM12M1M': 'Industry-Relative 12M-1M Momentum', 'PM12M1M': 'Price Momentum (12M-1M)', '52WSlope': '52-Week Price Slope',
    'BookLev': 'Book Leverage', 'Alpha18M6MPChg': '18M Alpha (6M P-Chg)', 'YoYChgDA': 'YoY Change in Discretionary Accruals', 'Alpha12M6MPChg': '12M Alpha (6M P-Chg)',
    'Beta60M': 'Beta (60M)', 'VolAdjRtn12M': 'Volatility-Adjusted Return (12M)', 'RelPrStr_12M': 'Relative Price Strength (12M)', 'Alpha36M6MPChg': '36M Alpha (6M P-Chg)',
    'CVVolPrc60D': 'CV of Volume*Price (60D)', 'CVVolPrc30D': 'CV of Volume*Price (30D)', 'CVVolPrc20D': 'CV of Volume*Price (20D)', 'STO_6M': 'Share Turnover (6M)',
    'RskAdjRS': 'Risk-Adjusted Relative Strength', '130DMinRtn': '130-Day Minimum Return', 'HL1M': 'High-Low Range (1M)', 'Chg1YAmihud': '1Y Change in Amihud Illiquidity',
    '4To52WPrcOsc': '4-to-52 Week Price Oscillator', 'HL52W': 'High-Low Range (52W)', 'Amihud': 'Amihud Illiquidity', 'Vol-style': 'Volatility Style Factor'
}
REVERSE_METRIC_NAME_MAP = {v: k for k, v in METRIC_NAME_MAP.items()}
# --- Generic mapping for any unmapped new factors ---
new_factor_list = [f for f in columns if f not in METRIC_NAME_MAP and f not in ["Ticker", "Name", "Sector", "Best_Factor", "Risk_Flag"]]
for factor in new_factor_list:
    # A simple transformation for a readable name
    human_readable_name = factor.replace('_', ' ').title()
    METRIC_NAME_MAP[factor] = human_readable_name
#tickers = ["KLAC", "NVDA", "NEU", "APH", "MSFT", "UI", "CME", "HLI", "EXEL", "HWM", "LRCX", "APP", "SCCO", "NFLX", "LLY", "EHC", "VRSK", "MCO", "RMD", "ALLE", "CTAS", "META", "CRS", "AGX", "INTU", "HALO", "RCL", "HEI", "CF", "GOOG", "EME", "DRI", "WWD", "TT", "TMUS", "BR", "BAH", "FTDR", "TDG", "FCFS", "CBOE", "VST", "JNJ", "AVGO", "BLK", "EAT", "NTAP", "IESC", "COP", "MTDR", "AIT", "AMAT", "SFM", "CW", "TPR", "STX", "RL", "HIMS", "WTS", "HCA", "NDAQ", "VRT", "AMGN", "COST", "ESE", "ROL", "MPLX", "FAST", "NVT", "CTRA", "BMI", "COR", "MELI", "WING", "NEM", "NSC", "FIX", "LRN", "BWXT", "EXPE", "POWL", "WELL", "SXT", "MA", "UTHR", "GD", "USFD", "PM", "ETN", "QDEL", "EVR", "MDT", "VEEV", "BKNG", "WINA", "FICO", "FSS", "FTNT", "PATK", "LECO", "GILD", "RSG", "MCK", "ICE", "CAKE", "PWR", "PLTR", "ALSN", "IPAR", "HESM", "GWW", "SYY", "ITW", "AWI", "PKG", "IBM", "QCOM", "CSCO", "CAH", "ITT", "LII", "DPZ", "URI", "TXRH", "TXN", "MO", "GSHD", "EBAY", "AJG", "FTI", "MSI", "ZTS", "CTRE", "IMO", "IDXX", "ORCL", "ITRI", "DY", "V", "GRMN", "PPC", "SPGI", "LEU", "UBER", "ALV", "LNG", "ADP", "SNPS", "TGLS", "GE", "PRIM", "UNP", "BSY", "MWA", "AXON", "TRGP", "EA", "ABT", "PH", "WAB", "FFIV", "JCI", "LIN", "VRSN", "MPWR", "TEL", "HD", "MAR", "MORN", "CACC", "NYT", "UHS", "QLYS", "SSNC", "CPRX", "LCII", "CL", "JBL", "OVV", "BX", "PJT", "YUM", "CVNA", "NXST", "FOX", "RRR", "DTE", "NTES", "WM", "LMT", "ACN", "DGX", "ROP", "MRK", "UGI", "BYD", "CR", "XOM", "DUOL", "MCD", "NDSN", "KTB", "LAD", "WMB", "APO", "MGY", "REVG", "ADI", "FN", "DE", "DLB", "GEN", "MNST", "HLT", "SBAC", "ENB", "ADT", "SYK", "CTSH", "EW", "BRO", "FLR", "CHTR", "VZ", "ARMK", "PLNT", "EXPD", "PNR", "PAYX", "MSA", "CHE", "GGG", "CDNS", "T", "AON", "CDW", "NRG", "IDCC", "CAT", "PTC", "TNL", "DIS", "UNH", "BRBR", "ATI", "DOCS", "LPLA", "JHG", "BJ", "SANM", "CCK", "ANSS", "PAG", "PR", "AMD", "XEL", "TW", "PIPR", "ATMU", "VMI", "WEC", "KMI", "RTX", "LULU", "HUBB","CORZ", "SHW", "OSK","RDDT", "PCAR", "CPA", "RS", "CACI", "AZO", "AROC", "PINS", "SNEX", "DORM", "PTCT", "MSCI", "PODD", "ECL", "ROK", "WDFC", "JKHY", "MTN", "HAL", "NOW", "TROW", "IRDM", "STE", "FDS", "MLI", "SBRA", "EXE", "PAYC", "HURN", "SCI", "GDDY", "BABA", "TSCO", "EMR", "ESAB", "CSL", "VICI", "DG", "SNA", "TDY", "GVA", "ORLY", "CHEF", "BMRN", "CIVI", "CHDN", "DK", "FHI", "EXR", "SU", "ENS", "MC", "HOOD", "DELL", "NXPI", "VRTX", "HON", "HAS", "UFPT", "KRYS", "CMCSA", "BDC", "ENSG", "LVS", "ULTA", "PG", "HRB", "ISRG", "SPXC", "GLPI", "ABBV", "EOG", "TER", "KR", "VMC", "PEGA", "HES", "GFF", "HSY", "SSD", "CPRT", "CEG", "UPWK", "CROX", "ATR", "EPD", "DCI", "HQY", "LOPE", "CORT", "TJX", "BSX", "BCPC", "ELS", "FLO", "H", "G", "DKS", "AEIS", "COKE", "EQT", "ROAD", "AMP", "VC", "HAE", "CVX", "WSM", "AAPL", "DDS", "REGN", "INGR", "WH", "MEDP", "CMG", "JAZZ", "AMT", "AEE", "PEP", "ET", "NFG", "LAZ", "KFY", "OSIS", "PCTY", "DOV", "WDC", "LAMR", "MMSI", "NTNX", "ZBH", "IT", "GHC", "WMT", "PHM", "ATO", "ACMR", "MATX", "TKR", "MAIN", "OXY", "WCC", "NEE", "TTC", "OKE", "QSR", "HLNE", "TRU", "CRM", "BKR", "IBP", "AYI", "CNM", "EXP", "URBN", "SKYW", "SWKS", "DT", "AME", "WMG", "BOOT", "FLEX", "WES", "ELF", "F", "PBH", "MANH", "AOS", "OTIS", "APPF", "CPNG", "MKSI", "OLED", "XYL", "GPOR", "SLB", "TMO", "TTMI", "ADM", "TYL", "AMZN", "PEG", "EPR", "STAG", "FANG", "PAGP", "SMLR", "NOC", "CWH", "ADBE", "DECK", "TXT", "LOW", "NBIX", "AMCR", "MSM", "CVLT", "K", "CNP", "YUMC", "SPSC", "CRH", "A", "BRK-A", "LNTH", "MKTX", "RPRX", "CSX", "PII", "SAIC", "NET", "ADSK", "GMS", "CARR", "RMBS", "THC", "IVZ", "CTVA", "CRUS", "PAA", "NNN", "CBT", "AR", "WMS", "CHH", "WPC", "MMS", "ES", "TDW", "FDX", "OHI", "KMB", "WCN", "NKE", "POWI", "WAT", "CMI", "CLX", "LOGI", "OTEX", "BMY", "FIVE", "UPS", "YOU", "BBWI", "EXC", "PPL", "NUE", "EFX", "STRL", "AMG", "MPC", "WSO", "DXCM", "CVS", "IEX", "WHD", "DVN", "TRMB", "KAI", "AVY", "WST", "TMDX", "GM", "CIEN", "UAL", "NSP", "HXL", "ZM", "YELP", "KVUE", "TTD", "LNT", "FOUR", "MAT", "FLS", "ROST", "PSX", "RPM", "KEX", "LKQ", "CP", "LTH", "CARG", "DDOG", "CGNX", "BROS", "ANF", "LHX", "PFE", "NVR", "OC", "TMHC", "MARA", "PANW", "AL", "GTLS", "GNTX", "MIDD", "GPN", "EMN", "GLW", "AVT", "PEN", "INFA", "BCO", "STZ", "HST", "CBRE", "FTAI", "ASO", "THO", "D", "TRNO", "OWL", "ZBRA", "RRC", "NI", "CWEN", "BLD", "ABNB", "CHD", "ACIW", "SKX", "POOL", "SKY", "KO", "VAC", "ED", "DVA", "SLGN", "GPC", "RKLB", "AMED", "PSN", "IQV", "FUL", "WFRD", "LH", "CLH", "LPX", "MUSA", "FE", "SMCI", "VIRT", "GEHC", "MMM", "GTES", "LEA", "TDC", "MAS", "ACM", "GNRC", "CVCO", "DOCU", "DBX", "ESS", "COHR", "KDP", "PPG", "DOX", "GPI", "BDX", "VNT", "GIS", "FNF", "PGNY", "OPCH", "BOX", "UDR", "MOH", "RBA", "JLL", "SHOO", "SBUX", "EGP", "BXP", "MLM", "CRC", "TGTX", "MTD", "ODFL", "UFPI", "APA", "MYRG", "CELH", "FI", "VNOM", "GPK", "PARA", "TEX", "RVLV", "BRKR", "KEYS", "OMC", "CDE", "JBHT", "BIIB", "FCX", "SRE", "ACAD", "MDLZ", "CMS", "INCY", "MTZ", "OGN", "ST", "SPG", "BALL", "EPRT", "LFUS", "ATGE", "CUBE", "INSM", "HTHT", "GATX", "IP", "J", "EXLS", "VLO", "HOLX", "XPO", "MTCH", "DAL", "ALKS", "LYB", "ENTG", "SLM", "BL", "ARRY", "CNC", "LW", "FMC", "ELAN", "ARW", "KNF", "YETI", "CAVA", "ESI", "SITE", "TREX", "WEN", "ABG", "VNO", "TGT", "MKC", "HP", "RRX", "BFAM", "DHR", "AVB", "MHO", "R", "TTEK", "SEE", "CNXC", "CRI", "NOVT", "MGM", "AA", "TOST", "BG", "KIM", "MSTR", "SMPL", "LEVI", "REG", "VVV", "PINC", "WSC", "MSGS", "ROIV", "NVS", "ETR", "SMG", "TGNA", "HRI", "ZS", "VRNA", "CCJ", "ALGN", "AMH", "BLDR", "DX", "STLD", "SWX", "LYV", "PSTG", "INOD", "TGI", "PK", "HPE", "BEN", "LYFT", "BZ", "SON", "CHWY", "SHAK", "WDAY", "SO", "CBZ", "EVRG", "ROOT", "FRT", "AN", "GXO", "CHRW", "PYPL", "HR", "KHC", "CCL", "HL", "TCOM", "GDS", "MTSI", "LBRT", "VSEC", "COO", "WYNN", "HOG", "AXTA", "HRL", "INVH", "SR", "GWRE", "HSIC", "FTV", "BCC", "MGNI", "ACLS", "ZTO", "CCCS", "IDA", "BBY", "APTV", "MOG-A", "PZZA", "EQR", "MAA", "FND", "RBLX", "SKT", "LIVN", "DRS", "SWK", "BURL", "ARCC", "KBR", "SNX", "ON", "FIS", "VSCO", "WU", "CSGP", "AAOI", "EPAM", "TFX", "TAP", "BKH", "MOS", "SYNA", "JEF", "SM", "AKAM", "TSLA", "POST", "NSIT", "GOLF", "WHR", "TEAM", "ITGR", "RGC", "VOD", "AAON", "COIN", "FSLR", "DV", "ALNY", "SIG", "BC", "TECH", "BWA", "SAND", "DUK", "TNET", "M", "TRP", "TPH", "PLAY", "IRM", "MOD", "MAC", "TSSI", "NCLH", "NFE", "HGV", "QRVO", "BILL", "QTWO", "IPG", "REXR", "ETSY", "NE", "DASH", "CRGY", "INSP", "BE", "COTY", "PLD", "TMC", "AFRM", "DEI", "CBRL", "SMR", "EG", "DXC", "LSTR", "TIGR", "NJR", "CAG", "DAN", "AEO", "TSN", "HII", "COLM", "WY", "FCN", "KNX", "KRG", "HPQ", "ORA", "CRWD", "FR", "SOFI", "WRBY", "CLSK", "EIX", "OLN", "RYTM", "IONS", "RGLD", "CUZ", "WTRG", "AMKR", "GMED", "SNOW", "BRX", "ABR", "AXSM", "FROG", "WTW", "KTOS", "TDS", "VCYT", "OKTA", "DKNG", "APD", "ICLR", "AVDX", "O", "LSCC", "CRK", "RHI", "CNK", "IONQ", "EQIX", "BPMC", "GME", "CPB", "JXN", "SAIA", "B", "ENPH", "RIOT", "PTGX", "EYE", "APLE", "FUN", "MQ", "SIRI", "TTWO", "KSS", "TARS", "SAP", "BTU", "PNW", "ASGN", "NOK", "KRC", "WW", "GO", "MNDY", "SJM", "SYM", "VRNS", "AEM", "MHK", "MP", "VITL", "KKR", "PVH", "BBIO", "OGS", "KGC", "GEO", "NSA", "MASI", "CAR", "ABM", "GT", "MDB", "AMR", "CPT", "DAR", "APLD", "LITE", "IRTC", "FIVN", "LUV", "AVA", "HASI", "ARQT", "AVTR", "PRGO", "VTR", "HUBS", "CNX", "MCHP", "ROKU", "RXRX", "CENX", "ALK", "AVAV", "NVAX", "SPB", "DD", "CCI", "FRPT", "FUBO", "DOC", "EQH", "NOG", "AES", "PCG", "WBA", "TENB", "EL", "SMTC", "CUK", "PCOR", "OUST", "Z", "ARE", "SAM", "MIR", "XRAY", "CC", "BCRX", "ADMA", "SLAB", "ACHC", "DBRG", "NLY", "SOUN", "TWLO", "RDNT", "GRPN", "TNXP", "DOW", "CRL", "DINO", "BRZE", "BA", "RGTI", "PL", "UPST", "AGCO", "DLTR", "WK", "AKRO", "KYMR", "VERX", "TEVA", "UUUU", "ESTC", "PGY", "PI", "CE", "GH", "GTLB", "IR", "AAL", "STEP", "SGRY", "HSAI", "VTRS", "ICUI", "STLA", "ILMN", "COMP", "AEVA", "PRCT", "ACHR", "MRVL", "CPRI", "HCTI", "PARR", "NVST", "ALKT", "CWAN", "HTZ", "CYTK", "COMM", "MRCY", "TDOC", "CZR", "AMBA", "IFF", "SNAP", "QBTS", "ALGM", "CALX", "KD", "FL", "ZETA", "SLNO", "GKOS", "INTA", "BEAM", "LUMN", "AI", "ALC", "DNB", "RNA", "HCC", "NOV", "PTEN", "MRUS", "W", "CABO", "RNG", "FRSH", "CCOI", "PAAS", "MBLY", "SATS", "SITM", "KMX", "CFLT", "ENVX", "ALHC", "MAN", "MPW", "TVTX", "SG", "IREN", "INTC", "VSAT", "S", "CRNX", "BAX", "NCNO", "RDW", "CLF", "IOVA", "SUI", "CIFR", "PCT", "PATH", "FOLD", "PTON", "NGD", "RVMD", "AG", "VFC", "MUR", "TWST", "RIG", "AGI", "VERA", "FYBR", "DYN", "GENI", "ALB", "NUVL", "HUN", "ACVA", "ASH", "IRT", "CRSP", "EXAS", "NWL", "QS", "PENN", "SRPT", "ZLAB", "SPR", "ALIT", "JOBY", "RGEN", "WBD", "LCID", "IAC", "NVTS", "AAP", "MDGL", "VERV", "HBI", "APVO", "NIO", "TNDM", "VKTX", "APLS", "ACLX", "RELY", "RUN", "PCVX", "RKT", "U", "AMC", "RARE", "DOCN", "NEOG", "AUR", "TXG", "RHP", "OPEN", "CNR", "LUNR", "JBLU", "ASAN", "RIVN", "COLD", "BB", "ESLT", "NTLA", "DAVE", "JPM", "SEDG", "BTI", "GFL", "TME", "CRDO", "MRNA", "WFC", "CASY", "CLS", "QUBT", "TBBK", "PLUG", "SAN", "FUTU", "MS", "GS", "DB", "PFGC", "SEIC", "SRAD", "STT", "RXO", "MAG", "BMO", "NTRS", "FSK", "CCEP", "TIGO", "C", "IBKR", "COF", "ING", "ATAT", "RCAT", "EWBC", "URTH", "BBAI", "WBS", "BPOP", "BK", "OLLI", "BCS", "MT", "TGB", "SCHW", "BBVA", "UNM", "SF", "APG", "MKL", "OMF", "VNET", "SYF", "TCBI", "RYAAY", "AWK", "NVMI", "L", "RJF", "FERG", "BFH", "GRAB", "STWD", "ONB", "OZK", "UBS", "GRRR", "RNST", "GFI", "LYG", "TSM", "BAP", "BUD", "FNB", "NXT", "FHN", "RF", "CFG", "AER", "RITM", "STNE", "HBAN", "DTM", "ERJ", "FLUT", "CALM", "TTE", "CADE", "EQNR", "ZION", "QFIN", "GRND", "KEY", "GNW", "RY", "MTB", "ASND", "CBSH", "ITUB", "RELX", "TS", "CRBG", "LMND", "NWG", "WTFC", "QGEN", "MLGO", "SE", "ASB", "WRB", "CX", "CFR", "ARES", "PNFP", "HUT", "AXP", "SNV", "WULF", "VOYA", "SPHR", "XPEV", "HIG", "HDB", "PNC", "SHEL", "BKU", "BAM", "AXS", "GSK", "BILI", "WGS", "FCNCA", "NGG", "HMC", "BAC", "VLY", "TPL", "TAK", "CMA", "LDOS", "BTBT", "ALLY", "HWC", "UEC", "CINF", "HOLO", "TFC", "UL", "EOSE", "PLMR", "ACI", "FITB", "MUFG", "YMM", "SSRM", "HIW", "OGE", "MMYT", "USB", "DNN", "KC", "CYBR", "AIG", "UBSI", "IBN", "GLNG", "LNC", "TRV", "BOH", "GL", "CAMT", "BTDR", "MU", "STM", "WAL", "UMBF", "BNS", "PGR", "TPG", "CB", "ARGX", "AZN", "SLG", "SMFG", "THG", "WTM", "HMY", "CHKP", "CNO", "ORI", "ALL", "ZWS", "LEGN", "NTRA", "UMC", "KNSL", "AGO", "NU", "ONON", "AMX", "AIZ", "SSB", "PB", "BANC", "VAL", "MTG", "XP", "ZIM", "AEP", "PFSI", "PBF", "SNY", "MULN", "EH", "ABEV", "MET", "ACGL", "OSCR", "AFL", "AFG", "CHX", "RGA", "GFS", "GGB", "FRO", "WIX", "GBCI", "RIO", "CI", "TAL", "GMAB", "SIGI", "ERIE", "COLB", "PECO", "VIST", "PFG", "POR", "STNG", "NICE", "BHF", "WEX", "FMX", "AUB", "HUM", "MMC", "PBR-A", "TM", "SQM", "RNR", "ARR", "CWST", "RLI", "BIO", "JHX", "CMC", "ADC", "HLN", "PRU", "ESNT", "KMPR", "KNTK", "DHI", "DSGX", "DEO", "EEFT", "INFY", "ATKR", "GLBE", "KBH", "PSA", "JD", "WLK", "LEN", "RH", "FAF", "ELV", "PRGS", "GLOB", "AGNC", "ANET", "TRIP", "SRRK", "ASTS", "XENE", "TOL", "MTH", "BHVN", "SMMT"]
tickers = ["KLAC", "NVDA", "NEU", "APH", "MSFT", "UI", "CME", "HLI", "EXEL", "HWM", "LRCX", "APP", "SCCO", "NFLX", "LLY", "EHC", "VRSK", "MCO", "RMD", "ALLE", "CTAS", "META", "CRS", "AGX", "INTU", "HALO", "RCL", "HEI", "CF", "GOOG"]
# --- `default_weights` updated with new factors at 0.0 ---
default_weights = {
    # --- Existing Weights ---
    "(Dividends + Share Buyback) / FCF": 5.0, "CapEx / (Depr + Amor)": 4.5, "Debt Ratio": 6.0, "Gross Profit Margin": 7.5, "Inventory Turnover": 4.5,
    "Net Profit Margin": 6.5, "Return on Assets": 6.0, "Assets Growth TTM": 5.5, "Assets Growth QOQ": 7.0, "Assets Growth YOY": 5.5, "FCF Growth TTM": 5.0,
    "FCF Growth QOQ": 6.0, "FCF Growth YOY": 6.0, "Dividend Yield": 4.0, "FCF Yield": 6.5, "Operating Margin": 4.5, "Liabilities to Equity Ratio": 4.5,
    "Earnings Per Share, Diluted": 4.5, "Dividend Payout Ratio": 0, "Return On Invested Capital": 6.0, "Piotroski F-Score": 6.5, "Operating Leverage": 5.5,
    "Cash Return On Invested Capital": 6.0, "Asset Turnover": 4.5, "Current Ratio": 6.0, "Dividends / FCF": 5.5, "Interest Coverage": 2.5, "Quick Ratio": 4.5,
    "Return on Equity": 7.0, "Share Buyback / FCF": 5.5, "Earnings Growth TTM": 5.5, "Earnings Growth QOQ": 6.5, "Earnings Growth YOY": 6.5,
    "Sales Growth TTM": 5.5, "Sales Growth QOQ": 5.5, "Sales Growth YOY": 6.5, "Earnings Yield": 6.5, "Market-Cap": 4.5, "P/E": 4.5, "P/Sales": 4.5,
    "Free Cash Flow": 3.5, "Free Cash Flow to Net Income": 5.5, "Sales Per Share": 4.5, "Free Cash Flow Per Share": 5.5, "Sharpe Ratio": 18.0,
    "Relative Z-Score": 18.0, "Market Correlation": 4.5, "Correlation_Score": 4.5, "Trend": 5.5, "Q Score": 6.5, "Coverage Score": 5.5, "Beta_to_SPY": -10,
    "GARCH_Vol": 5.5, "Vision": 4.5, "Value Factor": 6.0, "Profitability_Factor": 6.0, "Log_Log_Utility": 5.0, "Vol_Autocorr": 5.0,
    "Log_Log_Sharpe": 10.0, "Stop_Loss_Impact": 2.5, "AR_Coeff": 4.0, "Tangible_Book_Value": 4.5, "Return_On_Tangible_Equity": 6.0,
    "Insider_Ownership_Ratio": 6.0, "Earnings_Growth_Rate_5y": 5.5, "Peter_Lynch_Fair_Value": 4.5, "Peter_Lynch_Fair_Value_Upside": 6.0,
    "Revenue_Growth_Rate_5y": 5.5, "Meets_Triple_Rule": 3.0, "Return_21d": 4.0, "Return_63d": 4.5, "Return_126d": 5.0, "Return_252d": 5.5,
    "Audit Risk": 3.0, "Board Risk": 3.0, "Compensation Risk": 3.0, "Shareholder Rights Risk": 3.0, "Overall Risk": 4.0, "Institutional Ownership Ratio": 6.0,
    "Hurst Exponent (Lo's R/S)": 6.0, "5-Day Return": 3.0, "10-Day Return": 3.5,
}

# --- Add all new factors with a default weight of 0.0 ---
for short_name, long_name in METRIC_NAME_MAP.items():
    if long_name not in default_weights:
        default_weights[long_name] = 0.0

################################################################################
# SECTION 1: ALL FUNCTION DEFINITIONS
################################################################################
# --- ADD THESE NEW HELPER FUNCTIONS to SECTION 1 ---
BUSINESS_DAYS_IN_YEAR = 252

def _get_quarterly_value(df, possible_keys, col_index=0):
    """Safely extracts a quarterly financial value from a DataFrame."""
    if df is None or df.empty: return np.nan
    for key in possible_keys:
        if key in df.index:
            try:
                if len(df.columns) > col_index:
                    return pd.to_numeric(df.loc[key].iloc[col_index], errors='coerce')
            except IndexError: continue
    return np.nan

def _safe_division(numerator, denominator, default_val=np.nan):
    """Performs division, handling NaN/None inputs and zero denominators."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return default_val
    return numerator / denominator

def _calculate_annualized_growth(current_val, past_val, num_periods):
    """Calculates annualized growth rate."""
    if pd.isna(current_val) or pd.isna(past_val) or past_val <= 0 or num_periods == 0: return np.nan
    try:
        ratio = current_val / past_val
        if ratio < 0: return np.nan
        return ((ratio)**(1/num_periods) - 1) * 100
    except Exception: return np.nan

def _calculate_rsi(series, window=14):
    if series.empty or len(series) < window + 1: return np.nan
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    if loss.iloc[-1] == 0: return 100.0 if gain.iloc[-1] > 0 else 50.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return 100 - (100 / (1 + rs))

def _calculate_macd(series, fast_window=12, slow_window=26):
    if series.empty or len(series) < slow_window: return np.nan
    ema_fast = series.ewm(span=fast_window, adjust=False).mean()
    ema_slow = series.ewm(span=slow_window, adjust=False).mean()
    return (ema_fast.iloc[-1] - ema_slow.iloc[-1])

def _calculate_stochastic_oscillator(high, low, close, k_window=14):
    if close.empty or len(close) < k_window: return np.nan
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    return 100 * _safe_division(close.iloc[-1] - lowest_low.iloc[-1], highest_high.iloc[-1] - lowest_low.iloc[-1])

def _calculate_amihud_illiquidity(dollar_volume_series, returns_series):
    if dollar_volume_series.empty or returns_series.empty: return np.nan
    common_index = dollar_volume_series.index.intersection(returns_series.index)
    if len(common_index) < 20: return np.nan
    vol = dollar_volume_series.loc[common_index]
    ret = returns_series.loc[common_index]
    daily_illiquidity = _safe_division(ret.abs(), vol)
    return daily_illiquidity.mean() * 1e6 # Scale for readability

def _calculate_altman_z_score(info, financials, balancesheet, current_price):
    try:
        current_assets = get_value(balancesheet, ['Total Current Assets', 'CurrentAssets'])
        current_liabilities = get_value(balancesheet, ['Total Current Liabilities', 'CurrentLiabilities'])
        total_assets = get_value(balancesheet, ['Total Assets', 'TotalAssets'])
        working_capital = current_assets - current_liabilities
        T1 = _safe_division(working_capital, total_assets)

        retained_earnings = get_value(balancesheet, ['Retained Earnings'])
        T2 = _safe_division(retained_earnings, total_assets)

        ebit = get_value(financials, ['EBIT', 'Operating Income'])
        T3 = _safe_division(ebit, total_assets)

        market_cap = info.get('marketCap')
        total_liabilities = get_value(balancesheet, ['Total Liabilities', 'TotalLiab'])
        T4 = _safe_division(market_cap, total_liabilities)

        total_revenue = get_value(financials, ['Total Revenue', 'TotalRevenue'])
        T5 = _safe_division(total_revenue, total_assets)

        if any(pd.isna([T1, T2, T3, T4, T5])): return np.nan
        return 1.2*T1 + 1.4*T2 + 3.3*T3 + 0.6*T4 + 1.0*T5
    except Exception: return np.nan

# --- REPLACE your old `calculate_ewma_volatility` with this more robust one ---
def simple_ewvol_calc(daily_returns: pd.Series, days: int = 35, min_periods: int = 10, **ignored_kwargs) -> pd.Series:
    if daily_returns.empty or len(daily_returns.dropna()) < min_periods: return pd.Series(np.nan, index=daily_returns.index)
    return daily_returns.ewm(adjust=True, span=days, min_periods=min_periods).std()

def apply_min_vol(vol: pd.Series, vol_abs_min: float = 1e-10) -> pd.Series:
    if vol.empty: return pd.Series(dtype=float)
    return vol.clip(lower=vol_abs_min)

def apply_vol_floor(vol: pd.Series, floor_min_quant: float = 0.05, floor_min_periods: int = 100, floor_days: int = 500) -> pd.Series:
    if vol.empty: return pd.Series(dtype=float)
    vol_min = vol.rolling(window=floor_days, min_periods=floor_min_periods).quantile(q=floor_min_quant).ffill()
    return np.maximum(vol, vol_min)

def backfill_vol(vol: pd.Series) -> pd.Series:
    if vol.empty: return pd.Series(dtype=float)
    return vol.ffill().bfill()
    
# (This is the replacement function)
def calculate_ewma_volatility(returns_series: pd.Series, span: int = 63, annualize: bool = True):
    if returns_series.empty or returns_series.isnull().all(): return pd.Series(np.nan, index=returns_series.index)
    vol = simple_ewvol_calc(returns_series, days=span, min_periods=int(span/3))
    if vol.empty: return pd.Series(np.nan, index=returns_series.index)
    vol = apply_min_vol(vol)
    vol = apply_vol_floor(vol, floor_days=252, floor_min_periods=100)
    vol = backfill_vol(vol)
    if annualize: vol *= np.sqrt(BUSINESS_DAYS_IN_YEAR)
    return vol.reindex(returns_series.index, method='ffill').bfill()
# --- Helper Functions ---
def calculate_growth(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0: return np.nan
    return (current - previous) / abs(previous) * 100

def metric_name(col):
    return METRIC_NAME_MAP.get(col, col)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_macro_data(ticker_symbol, start_date, end_date):
    """Fetches data specifically for the macro regime analysis."""
    data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.error(f"No data found for {ticker_symbol} in the given period.")
        return None
    return data
def get_value(df, possible_keys, col_index=0):
    for key in possible_keys:
        if key in df.index and df.loc[key] is not None:
            series = df.loc[key]
            if isinstance(series, pd.Series) and len(series) > col_index:
                val = series.iloc[col_index]
                return pd.to_numeric(val, errors='coerce') if val is not None else np.nan
            elif not isinstance(series, pd.Series):
                 val = series
                 return pd.to_numeric(val, errors='coerce') if val is not None else np.nan
    return np.nan
def fit_msar_and_plot(df, title):
    """
    Fits a 2-state Markov-Switching Autoregressive model and creates an interactive plot.
    This model allows both the mean and variance to switch between regimes.
    """
    if df.empty or df['Turbulence'].isnull().all() or len(df) < 20:
        st.warning(f"Not enough data to fit MSAR for {title}.")
        return None, None

    # We use the returns of the turbulence series for the MSAR model
    y = df['Turbulence'].pct_change().dropna()
    
    # Define the MSAR model: 2 regimes, switching mean (trend='c'), and switching variance
    model = sm.tsa.MarkovRegression(
        y, k_regimes=2, trend='c', switching_variance=True
    )
    
    try:
        # Fitting with search_reps helps find a better (global) optimum
        res = model.fit(search_reps=10)
    except Exception as e:
        st.error(f"MSAR model fitting failed for {title}: {e}")
        return None, None

    # Get smoothed probabilities for the high-volatility regime
    high_vol_regime = np.argmax(res.params[-2:]) # The last two params are the variances (sigma2)
    df['MSAR_Event'] = res.smoothed_marginal_probabilities[high_vol_regime]

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['MSAR_Event'] * 100,
        name='Probability of High-Vol Regime',
        marker_color='rgba(26, 118, 255, 0.6)'
    ))
    fig.update_layout(
        title=f'MSAR-Detected Probability of High-Volatility Regime in {title}',
        yaxis_title='Probability (%)',
        template='plotly_dark',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Threshold")

    # Create a summary of the regimes
    summary = pd.DataFrame({
        'Mean (const)': res.params[0:2],
        'Volatility (std dev)': np.sqrt(res.params[-2:])
    }, index=[f'Regime {i}' for i in range(2)])
    
    return fig, summary    
def calculate_regime_aware_betas(stock_returns, market_returns, lookback=252):
    """
    Calculates separate betas for up-market and down-market days.
    This version is corrected to always return a dictionary, fixing potential errors.
    """
    # Align and slice data
    df = pd.concat([stock_returns, market_returns], axis=1).dropna().tail(lookback)
    df.columns = ['stock', 'market']

    # Define regimes based on market returns
    down_market_days = df[df['market'] < 0]
    up_market_days = df[df['market'] >= 0]

    # --- THIS IS THE CORRECTED LOGIC ---
    # Check if there's enough data for a stable estimate in both regimes
    if len(df) < 30: # Added total data length check
        return {'down_beta': np.nan, 'up_beta': np.nan, 'conservative_beta': np.nan}
        
    if len(down_market_days) < 10 or len(up_market_days) < 10: # Lowered threshold for robustness
        # If not, fall back to a simple, single beta for all cases.
        # This ensures the function *always* returns the expected dictionary format.
        try:
            # Ensure market returns are not all zero or constant
            if df['market'].std() < 1e-6: # Added variance check
                beta = 0.0 # No market movement to regress against
            else:
                beta = linregress(df['market'], df['stock']).slope
            # Return the single beta for all keys
            return {'down_beta': beta, 'up_beta': beta, 'conservative_beta': beta}
        except ValueError:
            # Handle rare cases where linregress fails on small samples
            return {'down_beta': 1.0, 'up_beta': 1.0, 'conservative_beta': 1.0}


    # If there IS enough data, calculate beta for each regime
    try:
        # Check for constant market returns in sub-samples
        if down_market_days['market'].std() < 1e-6: # Added variance check for down market
            down_beta = 0.0
        else:
            down_beta = linregress(down_market_days['market'], down_market_days['stock']).slope
            
        if up_market_days['market'].std() < 1e-6: # Added variance check for up market
            up_beta = 0.0
        else:
            up_beta = linregress(up_market_days['market'], up_market_days['stock']).slope
        
        # The conservative beta is the one with the larger magnitude, representing the bigger risk.
        conservative_beta = down_beta if abs(down_beta) > abs(up_beta) else up_beta
        
        return {'down_beta': down_beta, 'up_beta': up_beta, 'conservative_beta': conservative_beta}

    except Exception as e:
        logging.error(f"Error in regime beta calculation, falling back. Error: {e}")
        # Fallback in case of an unexpected error during regression
        if df['market'].std() < 1e-6: # Added variance check for fallback
            beta = 0.0
        else:
            beta = linregress(df['market'], df['stock']).slope
        return {'down_beta': beta, 'up_beta': beta, 'conservative_beta': beta}
def display_macro_regime_analysis():
    """
    Orchestrates and displays the entire macro volatility regime analysis.
    This function encapsulates the logic from the standalone regime analysis app.
    """
    st.header("📈 S&P 500 Macro Volatility Regime Analysis")
    st.info("This section analyzes the S&P 500 returns to identify the current market regime (e.g., Low, Medium, High Volatility). Understanding the macro 'weather' is crucial before selecting individual stocks.")

    # --- Configuration ---
    data_start_date = "2007-01-01"
    data_end_date = datetime.today().strftime('%Y-%m-%d')
    n_hidden_states = 3  # HMM states
    k_regimes = 3      # MSAR regimes

    # --- Step 1: Fetch Data and Compute All Models ---
    with st.spinner("Fetching S&P 500 data and fitting GARCH, HMM, and MSAR models..."):
        # Use the renamed function to avoid conflicts
        sp500_daily_full_data = get_macro_data('^GSPC', data_start_date, data_end_date)
        
        if sp500_daily_full_data is None or sp500_daily_full_data.empty: # Added robustness check
            st.error("No valid S&P 500 data for analysis. Please check '^GSPC' ticker or date range.")
            return

        rt_sp500 = calculate_returns(sp500_daily_full_data)
        
        if rt_sp500 is None or rt_sp500.empty:
            st.error("No valid S&P 500 returns data for analysis.")
            return

        garch_results_sp500, garch_message = fit_garch_model(rt_sp500)
        garch_cond_vol_series_spx = garch_results_sp500.conditional_volatility if garch_results_sp500 else None

        # Check if GARCH result is valid before passing to HMM
        if garch_cond_vol_series_spx is None or garch_cond_vol_series_spx.empty or garch_cond_vol_series_spx.isnull().all(): # Added robustness check
            st.warning("GARCH conditional volatility could not be computed. Skipping HMM analysis.")
            hmm_results, hmm_message = (None, None, None, None, None, None), "Skipped due to GARCH failure."
        else:
            hmm_results, hmm_message = fit_hmm_gaussian(garch_cond_vol_series_spx, n_hidden_states)
        
        hmm_states, hmm_means, hmm_sigmas, _, hmm_post_probs_df, _ = hmm_results

        msar_results_sp500, msar_message = fit_msar_model(rt_sp500, k_reg=k_regimes, order_ar=1, trend_msar='n')
        msar_post_probs_df, msar_states = (None, None)
        if msar_results_sp500 and hasattr(msar_results_sp500, 'smoothed_marginal_probabilities'): # Added hasattr check
            msar_post_probs_df = msar_results_sp500.smoothed_marginal_probabilities
            if msar_post_probs_df is not None:
                msar_states = msar_post_probs_df.idxmax(axis=1).values

    # --- Step 2: Display Current Volatility Regime Table ---
    st.subheader("Current Market Regime & Implications")
    regime_table = create_regime_table(hmm_states, hmm_means, hmm_sigmas, hmm_post_probs_df, msar_states, msar_post_probs_df, rt_sp500)
    if regime_table is not None and not regime_table.empty:
        st.dataframe(regime_table, use_container_width=True)
    else:
        st.error("Could not generate regime table: one or more models failed to converge or data was insufficient.") # Improved error message

    # --- Step 3: Display Charts and Detailed Results in Expander ---
    with st.expander("View Detailed Charts and Model Diagnostics"):
        st.subheader("Visualizations")
        # HMM plot
        if hmm_post_probs_df is not None and garch_cond_vol_series_spx is not None: # Added check for garch_cond_vol_series_spx
            st.write("#### HMM Volatility Regime Probabilities")
            plot_vol_regimes_prob(
                hmm_post_probs_df.index, garch_cond_vol_series_spx, hmm_post_probs_df,
                "HMM on GARCH Conditional Volatility", y_axis_label="GARCH Cond. Vol (Daily %)"
            )
        # MSAR plot
        if msar_post_probs_df is not None:
            st.write("#### MSAR Volatility Regime Probabilities")
            plot_vol_regimes_prob(
                msar_post_probs_df.index, rt_sp500, msar_post_probs_df,
                "MSAR on S&P 500 Returns", y_axis_label="S&P 500 Daily Returns (%)"
            )

        st.subheader("Model Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Return Distribution & Tests")
            return_dist_stats(rt_sp500)
            test_dist(rt_sp500, 'mean')
            test_dist(rt_sp500, 'normal')
        with col2:
            st.write("#### GARCH Model Results")
            if garch_results_sp500:
                st.text(garch_results_sp500.summary().as_text())
            else:
                st.error(garch_message)       
# Helper function required by the main function
def nearest_psd_matrix(matrix):
    """
    Ensure a matrix is positive semi-definite using SVD for numerical stability.
    """
    try:
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        # Symmetrize the matrix
        matrix = (matrix + matrix.T) / 2
        
        # Perform SVD
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Reconstruct with non-negative singular values
        s = np.maximum(s, 1e-10)
        psd_matrix = U @ np.diag(s) @ Vt
        
        # Symmetrize again to ensure exact symmetry after reconstruction
        psd_matrix = (psd_matrix + psd_matrix.T) / 2
        return psd_matrix
    except Exception as e:
        # Fallback to identity matrix on error
        logging.error(f"Error in PSD correction: {e}")
        return np.eye(len(matrix))

# --------------------------------------------------------------------------------------
# --- CORRECTED FUNCTION ---------------------------------------------------------------
# --------------------------------------------------------------------------------------
@st.cache_data
def fetch_turbulence_data():
    """
    Fetches live data for economic growth, inflation, sectors, and currencies.
    This version is corrected to be robust against yfinance download failures.
    """
    end_date = datetime.now()
    start_date = "1990-01-01"
    empty_df = pd.DataFrame()

    # 1. Economic Data from FRED
    try:
        fred_tickers = {
            'GNPC96': 'Economic Growth', # Real Gross National Product, Quarterly
            'CPIAUCSL': 'Inflation'       # Consumer Price Index, Monthly
        }
        econ_df = web.DataReader(list(fred_tickers.keys()), 'fred', "1947-01-01", end_date)
        econ_df = econ_df.rename(columns=fred_tickers)
        econ_growth = econ_df[['Economic Growth']].dropna().pct_change() * 100
        inflation = econ_df[['Inflation']].dropna().pct_change() * 100
    except Exception as e:
        logging.error(f"Failed to fetch FRED data: {e}")
        # Return 4 empty dataframes to match the expected output signature
        return empty_df, empty_df, empty_df, empty_df

    # 2. S&P 500 Sector Data (Robust Download)
    sector_tickers = list(sector_etf_map.values())
    raw_sector_data = yf.download(sector_tickers, start=start_date, end=end_date, progress=False)
    
    # Check if download failed or returned malformed data before accessing columns
    if raw_sector_data.empty or 'Adj Close' not in raw_sector_data.columns:
        logging.error("yfinance download for sector tickers failed or returned invalid data.")
        st.error("Failed to download sector ETF data. Turbulence analysis will be skipped.")
        return empty_df, empty_df, empty_df, empty_df # Match output signature
    sector_prices = raw_sector_data['Adj Close'].dropna()

    # 3. G-10 Currency Data (Robust Download)
    currency_tickers = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'CHF=X', 'CAD=X', 'AUDUSD=X', 'NZDUSD=X', 'SEK=X', 'NOK=X']
    raw_currency_data = yf.download(currency_tickers, start=start_date, end=end_date, progress=False)

    # Check if download failed before accessing columns
    if raw_currency_data.empty or 'Adj Close' not in raw_currency_data.columns:
        logging.error("yfinance download for currency tickers failed or returned invalid data.")
        st.error("Failed to download currency data. Turbulence analysis will be skipped.")
        return empty_df, empty_df, empty_df, empty_df # Match output signature
    currency_prices = raw_currency_data['Adj Close'].dropna()

    return econ_growth.reset_index().rename(columns={'DATE':'Date'}), \
           inflation.reset_index().rename(columns={'DATE':'Date'}), \
           sector_prices.reset_index(), \
           currency_prices.reset_index()
# --------------------------------------------------------------------------------------
# --- END OF CORRECTION ----------------------------------------------------------------
# --------------------------------------------------------------------------------------

def compute_turbulence(price_df, years=3, alpha=0.01):
    """
    Computes financial turbulence (Mahalanobis distance) from a price dataframe.
    Adapted from the original script to be more robust.
    """
    # Ensure 'Date' is the index
    if 'Date' in price_df.columns:
        price_df = price_df.set_index('Date')

    returns = price_df.pct_change().dropna()
    distances = []
    days_in_year = 252
    lookback_window = years * days_in_year

    if len(returns) < lookback_window:
        st.warning(f"Not enough data for {years}-year turbulence calculation. Need {lookback_window} days, have {len(returns)}.")
        return pd.DataFrame({'Turbulence': []}, index=[])

    for i in range(lookback_window, len(returns)):
        # Historical window of returns
        past_returns = returns.iloc[i - lookback_window : i]
        mu = past_returns.mean().values
        y = returns.iloc[i].values

        try:
            # Using Ledoit-Wolf for a more stable covariance estimate
            cov_estimator = LedoitWolf().fit(past_returns)
            inv_sig = np.linalg.inv(cov_estimator.covariance_)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            sigma = past_returns.cov().values
            inv_sig = np.linalg.inv(sigma + np.eye(sigma.shape[0]) * alpha)

        d = (y - mu).T @ inv_sig @ (y - mu)
        distances.append(d)

    turbulence_df = pd.DataFrame(
        {'Turbulence': distances},
        index=returns.index[lookback_window:]
    )
    return turbulence_df

def data_smoothing(turbulence_df):
    """
    Performs data smoothing by taking square root and monthly average.
    Adapted to use pandas resampling for cleaner code.
    """
    if turbulence_df.empty:
        return pd.DataFrame(columns=['Date', 'Turbulence'])
        
    # Take the square root of turbulence
    sqrt_turbulence = np.sqrt(turbulence_df['Turbulence'])

    # Compute monthly mean turbulence
    monthly_turbulence = sqrt_turbulence.resample('M').mean().to_frame()
    monthly_turbulence = monthly_turbulence.rename(columns={"Turbulence": "Turbulence"})
    
    return monthly_turbulence.reset_index().rename(columns={'index':'Date'})

def fit_hmm_and_plot(df, title):
    """
    Fits a 2-state HMM, identifies regimes, and creates an interactive plot.
    Replaces the original script's manual implementation with hmmlearn and Plotly.
    """
    if df.empty or df['Turbulence'].isnull().all():
        st.warning(f"No valid data to fit HMM for {title}.")
        return None, None, None

    y = df['Turbulence'].values.reshape(-1, 1)

    # Fit HMM
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(y)

    # Identify Event vs. Normal regimes (Event regime has higher mean/volatility)
    state_means = model.means_.flatten()
    event_index = np.argmax(state_means)
    normal_index = np.argmin(state_means)
    
    # Get smoothed probabilities
    smoothed_probs = model.predict_proba(y)
    df['Event'] = smoothed_probs[:, event_index]

    # Extract regime characteristics
    persistence = model.transmat_.diagonal()
    event_regime = [persistence[event_index], model.means_[event_index][0], np.sqrt(model.covars_[event_index][0][0])]
    normal_regime = [persistence[normal_index], model.means_[normal_index][0], np.sqrt(model.covars_[normal_index][0][0])]

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Event'] * 100,
        name='Probability of Event Regime',
        marker_color='rgba(239, 85, 59, 0.6)'
    ))
    fig.update_layout(
        title=f'HMM-Detected Probability of High-Turbulence Regime in {title}',
        yaxis_title='Probability (%)',
        template='plotly_dark',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% Threshold")

    return df[['Date', 'Event']], event_regime, normal_regime, fig

# --------------------------------------------------------------------------------------
# --- CORRECTED FUNCTION ---------------------------------------------------------------
# --------------------------------------------------------------------------------------
def display_turbulence_and_regime_analysis():
    """
    Main orchestrator function to run and display the entire analysis in Streamlit.
    (UPDATED to handle data failures and remove broken summary table)
    """
    st.header("🔥 Financial Turbulence & Macro Regime Detection")
    st.info("This analysis uses the Mahalanobis distance to measure financial turbulence. Hidden Markov (HMM) and Markov-Switching Autoregressive (MSAR) models then identify the probability of being in a high-turbulence 'event' regime.")

    with st.spinner("Fetching live macroeconomic and market data..."):
        econ_growth, inflation, sector_prices, currency_prices = fetch_turbulence_data()

    # --- ADDED: Graceful handling of data download failure ---
    if sector_prices.empty or currency_prices.empty: # Added robustness check
        st.warning("Could not proceed with turbulence analysis due to missing market data.")
        return # Stop the function here if data is unavailable

    with st.spinner("Calculating and smoothing financial turbulence..."):
        sector_turbulence = compute_turbulence(sector_prices, years=10)
        currency_turbulence = compute_turbulence(currency_prices, years=3)
        sector_turbulence_monthly = data_smoothing(sector_turbulence)
        currency_turbulence_monthly = data_smoothing(currency_turbulence)

    st.subheader("Regime Detection on Equity Turbulence")
    st.markdown("Comparing HMM (distribution-based) vs. MSAR (parameter-based) models.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # HMM Analysis
        # Note: fit_hmm_and_plot expects 'Date' in index or a column. data_smoothing outputs 'Date' column.
        hmm_results_df, event_regime, normal_regime, hmm_fig = fit_hmm_and_plot(sector_turbulence_monthly.copy(), "Equity Turbulence (HMM)")
        if hmm_fig:
            st.plotly_chart(hmm_fig, use_container_width=True)

    with col2:
        # MSAR Analysis
        msar_fig, msar_summary = fit_msar_and_plot(sector_turbulence_monthly.copy(), "Equity Turbulence (MSAR)")
        if msar_fig:
            st.plotly_chart(msar_fig, use_container_width=True)

    if msar_summary is not None and not msar_summary.empty: # Added check for empty summary
        st.write("MSAR Detected Regime Parameters (for Equity Turbulence):")
        st.dataframe(msar_summary.style.format("{:.4f}"))
        
    # --- REMOVED: Broken summary table section ---
    # The following code was removed because the 'results' variable was never defined,
    # which would have caused a NameError.
# --------------------------------------------------------------------------------------
# --- END OF CORRECTION ----------------------------------------------------------------
# --------------------------------------------------------------------------------------
def calculate_robust_hedge_weights(
    core_portfolio_returns, 
    hedge_instrument_returns, 
    lambda_uncertainty=0.5,
    market_factor_for_regime='SPY' ### CHANGE: Added parameter to define the market regime
):
    """
    Calculates optimal hedge weights using a tracking error minimization framework
    that accounts for beta estimation uncertainty (regularization) and is calibrated
    to DOWN-MARKET regimes for maximum robustness.
    """
    # 1. Align data and drop NaNs
    common_index = core_portfolio_returns.index.intersection(hedge_instrument_returns.index)
    if len(common_index) < 60:
        logging.warning("Insufficient data for robust hedging. Returning zero weights.")
        return pd.Series(0.0, index=hedge_instrument_returns.columns)
        
    y = core_portfolio_returns.loc[common_index]
    X = hedge_instrument_returns.loc[common_index]

    ### CHANGE: Define regimes based on the market factor
    if market_factor_for_regime not in X.columns:
        logging.error(f"Market factor {market_factor_for_regime} not in hedge instruments. Falling back.")
        down_market_index = X.index # Fallback to using all data
    else:
        market_returns = X[market_factor_for_regime]
        down_market_index = market_returns[market_returns < 0].index
        if len(down_market_index) < 30: # Ensure enough data for stable estimates
            logging.warning("Insufficient down-market days for regime hedge. Using full sample.")
            down_market_index = X.index

    # Slice the data to only include down-market days for beta calculation
    y_down = y.loc[down_market_index]
    X_down = X.loc[down_market_index]

    # 2. Estimate Betas (B matrix) and Uncertainty using ONLY the down-market data
    betas = []
    beta_variances = []

    for stock in y.columns:
        # For each stock, regress its returns against ALL hedge instruments using down-market data
        model = Ridge(alpha=0.1, fit_intercept=True).fit(X_down, y_down[stock])
        betas.append(model.coef_)
        
        # Estimate uncertainty from the residuals of this down-market regression
        residuals = y_down[stock] - model.predict(X_down)
        beta_variances.append(np.var(residuals))

    B = np.array(betas)
    Omega_beta = np.diag(beta_variances)

    # 3. Construct Components for the Optimal Hedge Formula
    # IMPORTANT: The covariance matrices should still be calculated on the FULL dataset
    # to capture the overall risk structure, not just the down-market structure.
    Sigma_hedge = X.cov().values
    
    regularizer = B.T @ Omega_beta @ B
    robust_Sigma = Sigma_hedge + (lambda_uncertainty * regularizer)
    
    combined_df = pd.concat([y, X], axis=1)
    full_cov_matrix = combined_df.cov()
    Sigma_core_hedge = full_cov_matrix.loc[y.columns, X.columns].values
    sum_covariances = np.sum(Sigma_core_hedge, axis=0)
    
    # 4. Solve for Optimal Hedge Weights
    try:
        inv_robust_Sigma = np.linalg.inv(nearest_psd_matrix(robust_Sigma))
        optimal_hedge_weights = -1 * inv_robust_Sigma @ sum_covariances
        
        return pd.Series(optimal_hedge_weights, index=X.columns)
        
    except np.linalg.LinAlgError:
        logging.error("Robust hedging failed due to singular matrix. Returning zero weights.")
        return pd.Series(0.0, index=X.columns)


@st.cache_data
def fetch_macro_data(start_date="2018-01-01"):
    """
    Fetches key macroeconomic time-series data from the FRED database.
    This version uses a default start date and calculates the end date automatically.
    """
    try:
        # Calculate end_date inside the function
        end_date = datetime.now()
        
        # Fetch 10-Year Treasury Yield (Interest Rates)
        ten_year_yield = pdr.DataReader('DGS10', 'fred', start_date, end_date) # Explicitly used pdr.DataReader
        # Fetch St. Louis Fed Financial Stress Index (centered at 0)
        stress_index = pdr.DataReader('STLFSI3', 'fred', start_date, end_date) # Explicitly used pdr.DataReader
        
        macro_df = pd.concat([ten_year_yield, stress_index], axis=1)
        macro_df.columns = ['Interest_Rate', 'Stress_Index']
        
        # Forward-fill to handle non-trading days, then back-fill any initial NaNs
        macro_df = macro_df.ffill().bfill()
        
        logging.info("Successfully fetched macroeconomic data.")
        return macro_df
    except Exception as e:
        logging.error(f"Failed to fetch macro data: {e}")
        # Return a dummy dataframe on failure to prevent crashes
        date_range = pd.date_range(start=start_date, end=datetime.now())
        return pd.DataFrame(0, index=date_range, columns=['Interest_Rate', 'Stress_Index'])

@st.cache_data
def fetch_turbulence_data():
    """
    Fetches live data for economic growth, inflation, sectors, and currencies.
    This version is corrected to be robust against yfinance download failures.
    """
    end_date = datetime.now()
    start_date = "1990-01-01"
    empty_df = pd.DataFrame()

    # 1. Economic Data from FRED
    try:
        fred_tickers = {
            'GNPC96': 'Economic Growth', # Real Gross National Product, Quarterly
            'CPIAUCSL': 'Inflation'       # Consumer Price Index, Monthly
        }
        # Explicitly set FRED as source
        econ_df = pdr.DataReader(list(fred_tickers.keys()), 'fred', "1947-01-01", end_date) # Used pdr.DataReader
        econ_df = econ_df.rename(columns=fred_tickers)
        econ_growth = econ_df[['Economic Growth']].dropna().pct_change() * 100
        inflation = econ_df[['Inflation']].dropna().pct_change() * 100
    except Exception as e:
        logging.error(f"Failed to fetch FRED data: {e}")
        # Return 4 empty dataframes to match the expected output signature
        return empty_df, empty_df, empty_df, empty_df

    # 2. S&P 500 Sector Data (Robust Download)
    sector_tickers = list(sector_etf_map.values())
    raw_sector_data = yf.download(sector_tickers, start=start_date, end=end_date, progress=False)
    
    # Check if download failed or returned malformed data before accessing columns
    if raw_sector_data.empty or 'Adj Close' not in raw_sector_data.columns: # Added robustness checks
        logging.error("yfinance download for sector tickers failed or returned invalid data.")
        st.error("Failed to download sector ETF data. Turbulence analysis will be skipped.")
        return empty_df, empty_df, empty_df, empty_df # Match output signature
    sector_prices = raw_sector_data['Adj Close'].dropna()

    # 3. G-10 Currency Data (Robust Download)
    currency_tickers = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'CHF=X', 'CAD=X', 'AUDUSD=X', 'NZDUSD=X', 'SEK=X', 'NOK=X']
    raw_currency_data = yf.download(currency_tickers, start=start_date, end=end_date, progress=False)

    # Check if download failed before accessing columns
    if raw_currency_data.empty or 'Adj Close' not in raw_currency_data.columns: # Added robustness checks
        logging.error("yfinance download for currency tickers failed or returned invalid data.")
        st.error("Failed to download currency data. Turbulence analysis will be skipped.")
        return empty_df, empty_df, empty_df, empty_df # Match output signature
    currency_prices = raw_currency_data['Adj Close'].dropna()

    return econ_growth.reset_index().rename(columns={'DATE':'Date'}), \
           inflation.reset_index().rename(columns={'DATE':'Date'}), \
           sector_prices.reset_index(), \
           currency_prices.reset_index()
def _run_analysis_on_subset(_data_subset, _all_possible_metrics, _reverse_metric_map):
    """
    Helper function to run the core factor stability analysis on a given subset of data.
    This encapsulates the logic of the original run_factor_stability_analysis function.
    """
    if _data_subset.empty or len(_data_subset.index.unique()) < 252: # Need at least a year of data
        logging.warning("Data subset is too small for meaningful stability analysis. Returning empty results.")
        return {metric: 0.0 for metric in _all_possible_metrics}, pd.DataFrame()

    time_horizons = {
        "1M": "Return_21d", "3M": "Return_63d",
        "6M": "Return_126d", "12M": "Return_252d",
    }
    valid_metric_cols = [c for c in _data_subset.columns if pd.api.types.is_numeric_dtype(_data_subset[c]) and 'Return' not in c and c not in ['Ticker', 'Name', 'Score']]
    stability_results = {}

    # Group data by date for cross-sectional analysis
    grouped_data = _data_subset.groupby(_data_subset.index)

    for horizon_label, target_column in time_horizons.items():
        if target_column in _data_subset.columns:
            historical_pure_returns = []
            # Iterate through time to perform point-in-time cross-sectional regressions
            for date, group in grouped_data:
                pure_returns_today = calculate_pure_returns(group, valid_metric_cols, target=target_column)
                if not pure_returns_today.empty:
                    historical_pure_returns.append(pure_returns_today)
            
            if historical_pure_returns:
                stability_df = analyze_coefficient_stability(historical_pure_returns)
                stability_results[horizon_label] = stability_df

    auto_weights, rationale_df = aggregate_stability_and_set_weights(
        stability_results, _all_possible_metrics, _reverse_metric_map
    )
    
    return auto_weights, rationale_df         
# --- Deep Dive Data Functions (for UI) ---
@st.cache_data
def fetch_and_organize_deep_dive_data(_ticker_symbol):
    try:
        ticker = yf.Ticker(_ticker_symbol)
        info = ticker.info
        hist_10y = ticker.history(period="10y", auto_adjust=True, interval="1d").tz_localize(None) # Added tz_localize(None)
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        if not info: return {"Error": f"Could not retrieve info for {_ticker_symbol}."}
        price_data = {
            'Price': info.get('currentPrice', info.get('regularMarketPrice')), 'Change': info.get('regularMarketChange'),
            'Change (%)': info.get('regularMarketChangePercent', 0) * 100, 'Day Low': info.get('dayLow'),
            'Day High': info.get('dayHigh'), 'Year High': info.get('fiftyTwoWeekHigh'), 'Year Low': info.get('fiftyTwoWeekLow'),
            '50-Day Avg': info.get('fiftyDayAverage'), '200-Day Avg': info.get('twoHundredDayAverage'),
            'Exchange': info.get('exchange'), 'Volume': info.get('volume'), 'Avg Volume': info.get('averageVolume'),
            'Open': info.get('open'), 'Previous Close': info.get('previousClose'), 'Market Cap': info.get('marketCap'),
            'Shares Outstanding': info.get('sharesOutstanding'), 'Beta': info.get('beta'),
            'Enterprise Value': info.get('enterpriseValue')
        }
        perf_data = {}
        if not hist_10y.empty:
            close_prices = hist_10y['Close']
            periods = {'1D': 1, '5D': 5, '1M': 21, '3M': 63, '6M': 126, 'YTD': None, '1Y': 252, '3Y': 252*3, '5Y': 252*5, '10Y': 252*10, 'Max': len(close_prices)-1}
            for name, p in periods.items():
                try:
                    if name == 'YTD':
                        current_year_prices = close_prices[close_prices.index.year == datetime.now().year] # Get prices for current year
                        if not current_year_prices.empty: # Added check for empty current_year_prices
                            ytd_start_price = current_year_prices.iloc[0]
                            perf_data[name] = (close_prices.iloc[-1] / ytd_start_price - 1) * 100 if ytd_start_price > 0 else np.nan
                        else: perf_data[name] = np.nan
                    elif p is not None and len(close_prices) > p and p > 0:
                        start_price = close_prices.iloc[-(p+1)]
                        perf_data[name] = (close_prices.iloc[-1] / start_price - 1) * 100 if start_price > 0 else np.nan
                    elif p is None and name == 'Max' and len(close_prices) > 1:
                        start_price = close_prices.iloc[0]
                        perf_data[name] = (close_prices.iloc[-1] / start_price - 1) * 100 if start_price > 0 else np.nan
                    else: perf_data[name] = np.nan
                except IndexError: perf_data[name] = np.nan
        ratios_ttm = {
            'P/E Ratio (TTM)': info.get('trailingPE'), 'Forward P/E Ratio': info.get('forwardPE'),
            'P/S Ratio (TTM)': info.get('priceToSalesTrailing12Months'), 'P/B Ratio (TTM)': info.get('priceToBook'),
            'EV/Revenue (TTM)': info.get('enterpriseToRevenue'), 'EV/EBITDA (TTM)': info.get('enterpriseToEbitda'),
            'Earnings Yield (TTM)': (1 / info.get('trailingPE')) * 100 if info.get('trailingPE') and info.get('trailingPE') != 0 else np.nan,
            'FCF Yield (TTM)': (info.get('freeCashflow', 0) / info.get('marketCap', 1)) * 100 if info.get('marketCap') and info.get('marketCap') != 0 else np.nan,
            'Dividend Yield (TTM)': info.get('dividendYield', 0) * 100,
            'Payout Ratio (TTM)': info.get('payoutRatio'), 'Current Ratio (TTM)': info.get('currentRatio'),
            'Quick Ratio (TTM)': info.get('quickRatio'), 'Debt/Equity (TTM)': info.get('debtToEquity'),
            'Return on Equity (ROE, TTM)': info.get('returnOnEquity', 0) * 100,
            'Return on Assets (ROA, TTM)': info.get('returnOnAssets', 0) * 100,
            'Gross Margin (TTM)': info.get('grossMargins', 0) * 100,
            'Operating Margin (TTM)': info.get('operatingMargins', 0) * 100,
            'Profit Margin (TTM)': info.get('profitMargins', 0) * 100,
        }
        def statement_to_df(df):
            if df is None or df.empty: return pd.DataFrame({"Data Not Available": []})
            df_display = df.copy()
            df_display.index.name = "Metric"
            df_display.columns = [d.strftime('%Y-%m-%d') for d in df_display.columns]
            # FIX: Replaced deprecated .applymap() with .map()
            return df_display.map(lambda x: f'{x:,.0f}' if isinstance(x, (int, float)) else x)
        return {
            "Price Data": price_data, "Performance": perf_data, "Key Ratios (TTM)": ratios_ttm,
            "Income Statement": statement_to_df(financials), "Balance Sheet": statement_to_df(balance_sheet),
            "Cash Flow": statement_to_df(cashflow),
        }
    except Exception as e: return {"Error": f"An error occurred: {e}"}
# Place this in SECTION 1 with other function definitions
# Place this in SECTION 1 with other function definitions

def plot_factor_exposure_breakdown(portfolio_betas):
    """
    Creates an interactive bar chart showing the portfolio's exposure to different risk factors.
    """
    df = portfolio_betas.reset_index()
    df.columns = ['Factor', 'Beta Exposure']
    df = df.sort_values('Beta Exposure', key=abs, ascending=True)

    colors = ['#00CC96' if val >= 0 else '#EF553B' for val in df['Beta Exposure']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Factor'],
        x=df['Beta Exposure'],
        orientation='h',
        marker_color=colors,
        hoverinfo='y+x',
        hovertemplate='<b>Factor:</b> %{y}<br><b>Exposure (Beta):</b> %{x:.3f}<extra></extra>'
    ))
    
    # Add a zero line for reference
    fig.add_shape(type="line", x0=0, y0=-0.5, x1=0, y1=len(df)-0.5,
                  line=dict(color="rgba(255, 255, 255, 0.5)", width=2))
    
    fig.update_layout(
        title_text='Systematic Risk Exposure Breakdown',
        xaxis_title='Portfolio Beta to Factor',
        yaxis_title=None,
        template='plotly_dark',
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=True, gridwidth=0.1, gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(showgrid=False)
    )
    return fig
def calculate_portfolio_factor_betas(portfolio_ts, factor_returns_df):
    """
    Calculates the portfolio's beta exposure to a set of factors using regression.
    """
    if portfolio_ts.empty or factor_returns_df.empty: # Added robustness checks
        return pd.Series(0.0, index=factor_returns_df.columns, name="Factor Betas")

    common_idx = portfolio_ts.index.intersection(factor_returns_df.index)
    aligned_portfolio_ts = portfolio_ts.loc[common_idx]
    aligned_factor_returns = factor_returns_df.loc[common_idx]
    
    if len(aligned_portfolio_ts) < 20 or aligned_factor_returns.empty or aligned_factor_returns.shape[1] == 0: # Added length check
        return pd.Series(0.0, index=factor_returns_df.columns, name="Factor Betas")

    # Filter out columns from aligned_factor_returns that have zero variance
    X_filtered = aligned_factor_returns.loc[:, aligned_factor_returns.std() > 1e-6] # Added filtering
    
    if X_filtered.empty: # Added check for empty filtered factors
        return pd.Series(0.0, index=factor_returns_df.columns, name="Factor Betas")

    model = Ridge(alpha=0.1).fit(X_filtered, aligned_portfolio_ts)
    
    # Map coefficients back to original factor_returns_df columns
    full_betas = pd.Series(0.0, index=factor_returns_df.columns)
    for i, col in enumerate(X_filtered.columns):
        full_betas[col] = model.coef_[i]
        
    return full_betas

def get_benchmark_metrics(benchmark_ticker="SPY", period="3y"):
    """
    Calculates key risk/return metrics for a benchmark ETF.
    """
    try:
        hist = yf.Ticker(benchmark_ticker).history(period=period, auto_adjust=True, interval="1d").tz_localize(None) # Added auto_adjust, interval, tz_localize
        if hist.empty or 'Close' not in hist.columns: # Added robustness check
            raise ValueError(f"No valid data for benchmark {benchmark_ticker}")

        returns = hist['Close'].pct_change(fill_method=None).dropna() # FIX: fill_method=None
        
        if returns.empty: # Added robustness check
            return {'Volatility': np.nan, 'Sharpe Ratio': np.nan}

        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        
        return {'Volatility': volatility, 'Sharpe Ratio': sharpe_ratio}
    except Exception as e:
        logging.error(f"Could not get benchmark metrics for {benchmark_ticker}: {e}")
        return {'Volatility': np.nan, 'Sharpe Ratio': np.nan}       
def winsorize_returns(returns_dict, lookback_T=126, d_max=6.0):
    """
    Winsorizes returns based on the robust z-score method described in
    "The Elements of Quantitative Investing".

    d_{i,t} = |log(1 + r_it)| / median(|log(1 + r_{i,t-1})|, ..., |log(1 + r_{i,t-T})|)

    Args:
        returns_dict (dict): Dictionary where keys are tickers and values are pd.Series of log returns.
        lookback_T (int): The lookback period (T) for the rolling median.
        d_max (float): The threshold. Returns whose score exceeds this are capped.

    Returns:
        dict: A new dictionary with the winsorized log return series.
    """
    winsorized_dict = {}
    total_winsorized_points = 0

    for ticker, log_returns in returns_dict.items():
        if log_returns.empty or len(log_returns) < lookback_T:
            winsorized_dict[ticker] = log_returns
            continue

        # Use simple returns for the formula's r_it component
        simple_returns = np.expm1(log_returns)
        
        # Calculate the absolute log returns for the median calculation
        abs_log_returns = log_returns.abs()
        
        # Calculate the rolling median denominator from the formula
        rolling_median_denom = abs_log_returns.rolling(window=lookback_T, min_periods=20).median().shift(1)
        
        
        # Avoid division by zero and fill NaNs robustly
        rolling_median_denom.replace(0, np.nan, inplace=True)
        # FIX: Replaced deprecated .fillna(method=...) with .ffill()
        rolling_median_denom.ffill(inplace=True)
        # Backfill any remaining NaNs at the beginning of the series
        rolling_median_denom.fillna(abs_log_returns.median(), inplace=True)
        
        # Calculate the d_it score for each point in time
        d_it = abs_log_returns / rolling_median_denom
        
        # Identify outliers
        outliers_mask = d_it > d_max
        
        if outliers_mask.any():
            total_winsorized_points += outliers_mask.sum()
            
            # Create a copy to modify
            winsorized_returns_series = log_returns.copy()
            
            # For each outlier, calculate the capped value
            # Capped |log(1+r)| = d_max * median(...)
            cap_value = d_max * rolling_median_denom[outliers_mask]
            
            # Preserve the original sign of the outlier return
            signed_cap = np.sign(log_returns[outliers_mask]) * cap_value
            
            # Apply the cap
            winsorized_returns_series[outliers_mask] = signed_cap
            
            winsorized_dict[ticker] = winsorized_returns_series
        else:
            # No outliers, just use the original series
            winsorized_dict[ticker] = log_returns
            
    logging.info(f"Winsorization complete. Capped {total_winsorized_points} outlier data points across all tickers.")
    return winsorized_dict
def display_deep_dive_data(ticker_symbol):
    data = fetch_and_organize_deep_dive_data(ticker_symbol)
    if "Error" in data:
        st.error(data["Error"])
        return
    for section, content in data.items():
        with st.expander(f"**{section}**", expanded=(section == "Key Ratios (TTM)")):
            if isinstance(content, dict):
                df = pd.DataFrame.from_dict(content, orient='index', columns=['Value'])
                df.index.name = 'Metric'
                df['Value'] = df['Value'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
                st.dataframe(df, use_container_width=True)
            elif isinstance(content, pd.DataFrame):
                st.dataframe(content, use_container_width=True)

# --- Advanced Metric & Data Fetching Functions ---
@lru_cache(maxsize=None)
def fetch_etf_history(ticker, period="3y"):
    history = yf.Ticker(ticker).history(period=period, auto_adjust=True, interval="1d")
    if history.empty or 'Close' not in history.columns: raise ValueError(f"No valid data for {ticker}")
    history.index = history.index.tz_localize(None)
    history.dropna(subset=['Close'], inplace=True) # This inplace is safe, not on a chained copy
    if history['Close'].eq(0).any(): raise ValueError(f"Zero Close prices for {ticker}")
    return history

@st.cache_data
def fetch_all_etf_histories(_etf_list, period="3y"):
    etf_histories = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_etf = {executor.submit(fetch_etf_history, etf, period): etf for etf in _etf_list}
        for future in tqdm(as_completed(future_to_etf), total=len(_etf_list), desc="Fetching ETF Histories"):
            etf = future_to_etf[future]
            try: etf_histories[etf] = future.result()
            except Exception as e: logging.error(f"Failed to fetch ETF history for {etf}: {e}")
    return etf_histories

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ticker_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="3y", auto_adjust=True, interval="1d").tz_localize(None)
    info = ticker.info
    financials = ticker.financials
    balancesheet = ticker.balance_sheet
    cashflow = ticker.cashflow
    quarterly_financials = ticker.quarterly_financials
    quarterly_balancesheet = ticker.quarterly_balance_sheet
    quarterly_cashflow = ticker.quarterly_cashflow
    return ticker, history, info, financials, balancesheet, cashflow, quarterly_financials, quarterly_balancesheet, quarterly_cashflow

# --- Quantitative Functions ---

def calculate_mahalanobis_metrics(returns, cov_matrix, periods=252):
    """
    Calculate Mahalanobis distance and MALV for precision matrix evaluation.
    """
    try:
        cov_matrix = nearest_psd_matrix(cov_matrix)
        precision_matrix = np.linalg.inv(cov_matrix)
        mahalanobis_distances = []
        for t in range(len(returns)):
            r_t = returns.iloc[t].values.reshape(-1, 1)
            distance = np.sqrt(r_t.T @ precision_matrix @ r_t).item()
            mahalanobis_distances.append(distance)
        mahalanobis_distances = np.array(mahalanobis_distances)
        malv = np.var(mahalanobis_distances**2)
        return malv, mahalanobis_distances
    except Exception as e:
        logging.error(f"Error in Mahalanobis calculation: {e}")
        return np.nan, []

def calculate_idiosyncratic_variance(returns_df, factor_returns_df, betas):
    """
    Calculate idiosyncratic variance for each asset.
    """
    try:
        idio_vars = {}
        # Align returns and factors
        common_index = returns_df.index.intersection(factor_returns_df.index)
        returns_df_aligned = returns_df.loc[common_index]
        factor_returns_df_aligned = factor_returns_df.loc[common_index]

        if returns_df_aligned.empty or factor_returns_df_aligned.empty: # Added robustness checks
            return pd.Series(0.0, index=returns_df.columns, name='IdioVariance')

        # Filter out factor columns that have zero variance for regression stability
        X_filtered = factor_returns_df_aligned.loc[:, factor_returns_df_aligned.std() > 1e-6] # Added filtering

        if X_filtered.empty: # If all factors have zero variance, idiosyncratic variance is total variance
            for ticker in returns_df_aligned.columns:
                idio_vars[ticker] = np.var(returns_df_aligned[ticker]) * 252
            return pd.Series(idio_vars, name='IdioVariance').fillna(0)

        for ticker in returns_df_aligned.columns:
            X = X_filtered.values
            y = returns_df_aligned[ticker].values
            
            # Simple regression to find residuals
            # Use Ridge for better stability if factors are highly correlated,
            # but LinearRegression is fine for residuals.
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            idio_vars[ticker] = np.var(residuals) * 252  # Annualized
        
        return pd.Series(idio_vars, name='IdioVariance').fillna(0)
    except Exception as e:
        logging.error(f"Error in idiosyncratic variance calculation: {e}")
        return pd.Series(0.0, index=returns_df.columns, name='IdioVariance')
def calculate_fmp_weights(returns_df, new_factor_returns, cov_matrix, existing_factors_returns=None):
    """
    Calculates CORRECTED Factor-Mimicking Portfolio (FMP) weights.
    The weights are calculated as w ∝ Σ⁻¹ * β, which maximizes factor exposure
    for a given level of portfolio variance.
    """
    try:
        tickers = returns_df.columns
        cov_matrix_psd = nearest_psd_matrix(cov_matrix)

        # Check for singularity after PSD adjustment
        if np.linalg.cond(cov_matrix_psd) > 1e10: # Added singularity check
            logging.warning("FMP: Covariance matrix is ill-conditioned after PSD. Returning equal weights.")
            return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)

        precision_matrix = np.linalg.inv(cov_matrix_psd)
        
        common_idx = returns_df.index.intersection(new_factor_returns.index)
        if existing_factors_returns is not None:
            common_idx = common_idx.intersection(existing_factors_returns.index)

        aligned_returns = returns_df.loc[common_idx]
        aligned_new_factor = new_factor_returns.loc[common_idx]

        if existing_factors_returns is not None and not existing_factors_returns.empty:
            aligned_existing_factors = existing_factors_returns.loc[common_idx]
            # Ensure aligned_existing_factors has non-zero variance columns for regression
            X_existing_filtered = aligned_existing_factors.loc[:, aligned_existing_factors.std() > 1e-6] # Added filtering
            if not X_existing_filtered.empty:
                model = LinearRegression().fit(X_existing_filtered, aligned_new_factor)
                ortho_factor_series = aligned_new_factor - pd.Series(model.predict(X_existing_filtered), index=common_idx)
            else: # If existing factors are constant, new factor is already orthogonal
                ortho_factor_series = aligned_new_factor
        else:
            ortho_factor_series = aligned_new_factor

        # Ensure ortho_factor_series has variance for regression
        if ortho_factor_series.std() < 1e-6: # Added check for constant factor
            logging.warning("FMP: Orthogonalized factor has no variance. Returning equal weights.")
            return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)


        # Estimate the betas of assets to the (orthogonalized) factor
        betas = []
        for ticker in aligned_returns.columns:
            # Use Ridge regression for more stable beta estimates
            model = Ridge(alpha=0.1).fit(ortho_factor_series.values.reshape(-1, 1), aligned_returns[ticker].values)
            betas.append(model.coef_[0])
        B = np.array(betas) # This is now a 1D array

        # The correct FMP formula is w ∝ Σ⁻¹ * β
        raw_weights = precision_matrix @ B
        
        # Normalize the raw weights to sum to 1 (for a long-only portfolio interpretation)
        # We take the absolute value to handle potential negative weights from the calculation
        abs_weights = np.abs(raw_weights)
        if abs_weights.sum() > 1e-9:
            final_weights = abs_weights / abs_weights.sum()
        else: # Fallback if all weights are zero
            final_weights = np.ones(len(tickers)) / len(tickers)

        return pd.Series(final_weights, index=tickers)

    except Exception as e:
        logging.error(f"Error in FMP calculation: {e}")
        return pd.Series(np.ones(len(returns_df.columns)) / len(returns_df.columns), index=returns_df.columns)
# --- CORRECTED PLOTTING FUNCTION 1 ---
def plot_sector_concentration(sector_counts):
    """
    Creates a colorful and interactive horizontal bar chart for sector concentration using Plotly.
    """
    df = sector_counts.reset_index()
    df.columns = ['Sector', 'Count']
    
    colors = px.colors.qualitative.Plotly
    num_colors = len(colors)
    sector_color_map = {sector: colors[i % num_colors] for i, sector in enumerate(df['Sector'])}
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Sector'],
        x=df['Count'],
        orientation='h',
        marker_color=[sector_color_map[s] for s in df['Sector']],
        text=df['Count'],
        textposition='outside',
        hoverinfo='y+x',
        hovertemplate='<b>Sector:</b> %{y}<br><b>Count:</b> %{x}<extra></extra>'
    ))

    fig.update_layout(
        title_text='Sector Concentration in Long Book',
        xaxis_title='Number of Stocks',
        yaxis_title=None,
        template='plotly_dark',
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            autorange="reversed",
            showgrid=False
        )
    )
    return fig

# --- CORRECTED PLOTTING FUNCTION 2 ---
def plot_factor_correlations(factor_correlations):
    """
    Creates an interactive bar chart for factor correlations, coloring bars by sign (positive/negative).
    """
    df = factor_correlations.reset_index()
    df.columns = ['Factor', 'Correlation']
    
    colors = ['#00CC96' if val >= 0 else '#EF553B' for val in df['Correlation']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Factor'],
        x=df['Correlation'],
        orientation='h',
        marker_color=colors,
        text=[f'{c:.3f}' for c in df['Correlation']],
        textposition='outside',
        hoverinfo='y+x',
        hovertemplate='<b>Factor ETF:</b> %{y}<br><b>Correlation:</b> %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title_text='Long Book Correlation to Factor ETFs',
        xaxis_title='Correlation Coefficient',
        yaxis_title=None,
        template='plotly_dark',
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        yaxis=dict(
            autorange="reversed",
            showgrid=False
        )
    )
    return fig
def calculate_information_metrics(forecasted_alphas_ts, portfolio_returns_ts, benchmark_returns_ts):
    """
    Calculate Information Coefficient (IC) and Information Ratio (IR).
    
    Args:
        forecasted_alphas_ts (pd.Series): Time series of forecasted alpha (e.g., from alpha_weights).
        portfolio_returns_ts (pd.Series): Time series of portfolio's realized returns.
        benchmark_returns_ts (pd.Series): Time series of benchmark's realized returns (e.g., SPY).

    Returns:
        tuple: (Information Coefficient, Information Ratio) or (np.nan, np.nan).
    """
    try:
        # --- Robustly prepare series for concatenation ---
        series_to_concat = []
        if not forecasted_alphas_ts.empty:
            series_to_concat.append(forecasted_alphas_ts.rename('alpha_forecast'))
        if not portfolio_returns_ts.empty:
            series_to_concat.append(portfolio_returns_ts.rename('portfolio_returns'))
        if not benchmark_returns_ts.empty:
            series_to_concat.append(benchmark_returns_ts.rename('benchmark_returns'))

        if len(series_to_concat) < 2: # Need at least two series to calculate correlation or active returns
            logging.warning("Insufficient non-empty series for IC/IR calculation after filtering.")
            return np.nan, np.nan
            
        # Align and concatenate
        aligned_df = pd.concat(series_to_concat, axis=1).dropna()
        
        # Explicitly ensure numeric dtypes, coercing errors
        for col in aligned_df.columns:
            aligned_df[col] = pd.to_numeric(aligned_df[col], errors='coerce')
        aligned_df = aligned_df.dropna() # Drop NaNs introduced by coercion

        if len(aligned_df) < 20: 
            logging.warning("Insufficient data for IC/IR calculation after alignment and coercion (need >=20 points).")
            return np.nan, np.nan
        
        # Ensure all expected columns are present after concat/dropna/coercion
        if 'alpha_forecast' not in aligned_df.columns or \
           'portfolio_returns' not in aligned_df.columns or \
           'benchmark_returns' not in aligned_df.columns:
            logging.warning("One or more essential columns ('alpha_forecast', 'portfolio_returns', 'benchmark_returns') missing after alignment. Skipping IC/IR.")
            return np.nan, np.nan


        alpha_forecast = aligned_df['alpha_forecast']
        portfolio_returns = aligned_df['portfolio_returns']
        benchmark_returns = aligned_df['benchmark_returns']

        # Information Coefficient (Spearman for robustness)
        # Ensure both series have variance for correlation
        if alpha_forecast.std() < 1e-9 or portfolio_returns.std() < 1e-9:
            ic = 0.0 # No variability, no correlation
        else:
            ic = alpha_forecast.corr(portfolio_returns, method='spearman')
        
        # Information Ratio (IR)
        active_returns = portfolio_returns - benchmark_returns
        
        if active_returns.std() == 0: # Check for constant active returns
            ir = 0.0 # No variability, so no active risk. IR is 0.
        else:
            mean_active_return = active_returns.mean() * 252 # Annualize mean active return
            tracking_error = active_returns.std() * np.sqrt(252) # Annualize tracking error
            ir = mean_active_return / tracking_error if tracking_error > 0 else np.nan
        
        return ic, ir
    except Exception as e:
        logging.error(f"Error in calculate_information_metrics: {e}")
        return np.nan, np.nan
def decompose_portfolio_risk(portfolio_returns_df, weights_df, factor_returns_df, factor_cov_matrix):
    """
    Decomposes the portfolio's total variance into systematic (factor) and specific (idiosyncratic) risk.
    Based on the principles of the Fundamental Law of Active Management.

    Args:
        portfolio_returns_df (pd.DataFrame): DataFrame of the portfolio's constituent stock returns.
        weights_df (pd.DataFrame): DataFrame with 'Ticker' and 'Weight' columns.
        factor_returns_df (pd.DataFrame): DataFrame of returns for the risk factors.
        factor_cov_matrix (pd.DataFrame): The covariance matrix of the factor returns.

    Returns:
        tuple: A tuple containing (total_variance, systematic_variance, specific_variance).
    """
    try:
        p_weights = weights_df.set_index('Ticker')['Weight']
        
        # 1. Calculate the total variance of the portfolio directly from its return series
        # Ensure returns are aligned and only valid tickers from weights_df are used
        aligned_returns = portfolio_returns_df.reindex(columns=p_weights.index).dropna(how='all')
        if aligned_returns.empty: # Added robustness check
            logging.warning("Aligned portfolio returns are empty for risk decomposition.")
            return np.nan, np.nan, np.nan

        # Reindex weights to aligned_returns.columns and fill NA with 0
        final_p_weights = p_weights.reindex(aligned_returns.columns).fillna(0)
        
        # Ensure weights sum to 1 to represent a portfolio
        if final_p_weights.sum() == 0: # Added robustness check
            logging.warning("Portfolio weights sum to zero for risk decomposition.")
            return np.nan, np.nan, np.nan
        final_p_weights = final_p_weights / final_p_weights.sum() # Normalize weights

        portfolio_ts = (aligned_returns * final_p_weights).sum(axis=1)
        total_variance = portfolio_ts.var() * 252  # Annualized

        # 2. Calculate each stock's beta exposure to the factors
        common_idx = aligned_returns.index.intersection(factor_returns_df.index)
        aligned_stock_returns = aligned_returns.loc[common_idx]
        aligned_factor_returns = factor_returns_df.loc[common_idx]
        
        # Added robustness checks for data length for regression
        if aligned_stock_returns.empty or aligned_factor_returns.empty or aligned_stock_returns.shape[0] < aligned_factor_returns.shape[1] + 2:
            logging.warning("Insufficient data for stable beta estimation in risk decomposition.")
            return total_variance, 0.0, total_variance # Assume all risk is specific if betas can't be estimated

        # Filter out factor columns that have zero variance for regression stability
        X_filtered = aligned_factor_returns.loc[:, aligned_factor_returns.std() > 1e-6]
        if X_filtered.empty: # Added robustness check
            logging.warning("All factor returns have zero variance. Assuming all risk is specific.")
            return total_variance, 0.0, total_variance


        betas_df = pd.DataFrame(index=aligned_stock_returns.columns, columns=X_filtered.columns)
        for ticker in aligned_stock_returns.columns:
            model = Ridge(alpha=0.1).fit(X_filtered, aligned_stock_returns[ticker])
            betas_df.loc[ticker] = model.coef_
        
        # Map betas back to the full set of factor columns, filling 0 for dropped ones
        full_betas_df = pd.DataFrame(0.0, index=betas_df.index, columns=factor_returns_df.columns)
        full_betas_df.loc[betas_df.index, X_filtered.columns] = betas_df

        # 3. Calculate Systematic (Factor) Variance: (w*B)' * F * (w*B)
        # w*B gives the weighted beta exposure of the portfolio to each factor
        portfolio_betas = (final_p_weights.values @ full_betas_df.values)
        
        # Ensure factor_cov_matrix is PSD
        factor_cov_matrix_psd = nearest_psd_matrix(factor_cov_matrix)

        systematic_variance = portfolio_betas.T @ factor_cov_matrix_psd @ portfolio_betas

        # 4. Calculate Specific (Idiosyncratic) Variance
        specific_variance = total_variance - systematic_variance
        
        # Ensure non-negative specific variance
        if specific_variance < 0:
            specific_variance = 0.0
            systematic_variance = total_variance - specific_variance # Recalculate if specific was negative

        return total_variance, systematic_variance, specific_variance

    except Exception as e:
        logging.error(f"Error in risk decomposition: {e}")
        return np.nan, np.nan, np.nan
# --- All Individual Metric Calculation Functions ---
def calculate_garch_volatility(returns, window=252, dist='t'):
    if returns.empty or len(returns) < window or returns.isna().all(): return np.nan
    try:
        # GARCH model expects non-zero mean and some variance to fit
        scaled_returns = returns.dropna() * 100 # Scale returns for better numerical stability in GARCH
        if len(scaled_returns) < 5 or scaled_returns.std() < 1e-6: return np.nan
        
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist=dist)
        res = model.fit(disp='off', last_obs=None, options={'maxiter': 500})
        cond_vol = res.conditional_volatility.iloc[-1] / 100 # Unscale volatility
        return cond_vol * np.sqrt(252)
    except Exception as e: # Explicitly capture exception
        logging.warning(f"GARCH for {returns.name if isinstance(returns, pd.Series) else 'N/A'} failed: {e}. Falling back to simple historical vol.") # Added logging
        return returns[-window:].std() * np.sqrt(252)
        
@lru_cache(maxsize=1024)
def calculate_returns_cached(ticker, periods_tuple):
    periods = list(periods_tuple)
    try:
        history = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
        if history.empty or len(history) < max(p for p in periods if p is not None): return {f"Return_{p}d": np.nan for p in periods}
        returns = {}
        for period in periods:
            if len(history) > period:
                returns[f"Return_{period}d"] = (history['Close'].iloc[-1] / history['Close'].iloc[-(period+1)] - 1) * 100
            else:
                returns[f"Return_{period}d"] = np.nan
        return returns
    except Exception: return {f"Return_{p}d": np.nan for p in periods}

def calculate_log_log_utility(returns):
    if returns.empty or returns.isna().all(): return np.nan
    try:
        # Clip to ensure log1p doesn't get values too close to -1 for stability
        # and ensure inner log1p operates on positive values
        safe_returns = returns.clip(lower=-0.9999) # Added clipping
        log_1p_returns = np.log1p(safe_returns) # log(1 + r)
        
        positive_log_returns = log_1p_returns[log_1p_returns > 0]
        if positive_log_returns.empty: return np.nan
        
        log_log_utility = np.mean(np.log1p(positive_log_returns))
        return log_log_utility if np.isfinite(log_log_utility) else np.nan
    except Exception as e: # Added specific exception and logging
        logging.warning(f"Error in calculate_log_log_utility: {e}")
        return np.nan

def calculate_log_log_sharpe(returns, window=252, risk_free_rate=0.04):
    if returns.empty or len(returns.dropna()) < window: return np.nan # Added dropna and better length check
    try:
        safe_returns = returns[-window:].clip(lower=-0.9999).dropna() # Added dropna
        if len(safe_returns) < window/2: return np.nan # Ensure enough data after dropna
        
        log_returns = np.log1p(safe_returns)
        log_log_returns = log_returns[log_returns > 0]
        
        if log_log_returns.empty or log_log_returns.isna().all(): return np.nan
        
        # Annualize
        mean_return = log_log_returns.mean() * 252
        std_return = log_log_returns.std() * np.sqrt(252)
        
        # Convert annual risk-free rate to daily for comparison if needed, though for Sharpe, it's consistent
        # daily_rf = risk_free_rate / 252 # This seems to be comparing annual mean to daily rf. Better to annualize RF.
        annual_rf = risk_free_rate # Corrected to annual RF for annualized mean_return

        if std_return is None or std_return == 0 or pd.isna(std_return): return np.nan
        
        # Using annual_rf directly as mean_return is also annualized
        return (mean_return - annual_rf) / std_return
    except Exception as e: # Added specific exception and logging
        logging.warning(f"Error in calculate_log_log_sharpe: {e}")
        return np.nan

def calculate_volatility_autocorrelation(returns, window=252):
    if returns.empty or len(returns.dropna()) < window: return np.nan # Added dropna and better length check
    try:
        # Use absolute returns or squared returns for volatility persistence
        squared_returns = (returns[-window:]**2).dropna()
        if len(squared_returns) < 2: return np.nan
        return squared_returns.autocorr(lag=1)
    except Exception as e: # Added specific exception and logging
        logging.warning(f"Error in calculate_volatility_autocorrelation: {e}")
        return np.nan

def calculate_stop_loss_impact(returns, stop_loss_level=-0.04):
    if returns.empty: return np.nan
    return (returns < stop_loss_level).mean()

def calculate_ar_coefficient(returns, lags=1, window=252):
    if returns.empty or returns.isna().all(): return np.nan
    effective_window = min(window, len(returns))
    if effective_window < 20: return np.nan # Need enough data for regression

    # Prepare data for AR(1) regression
    y_series = returns[-effective_window:].dropna()
    X_series = y_series.shift(lags).dropna() # Lagged returns
    y_series = y_series.reindex(X_series.index) # Align y
    
    if len(X_series) < 10: return np.nan # Need enough aligned data points
    
    try:
        # Check for zero variance in X_series to prevent linregress failure
        if X_series.std() < 1e-6: # Added check for constant X_series
            return 0.0 # No variability in lagged returns, so coefficient is 0
        
        slope, _, _, p_value, _ = linregress(X_series, y_series)
        # Only return coefficient if statistically significant and finite
        return slope if p_value < 0.1 and np.isfinite(slope) else 0.0
    except Exception as e: # Catch specific exception
        logging.warning(f"Error in calculate_ar_coefficient: {e}") # Added logging
        return 0.0

def breakout(price, lookback=20, smooth=5):
    if len(price) < lookback: return np.nan
    roll_max = price.rolling(lookback, min_periods=lookback//2).max()
    roll_min = price.rolling(lookback, min_periods=lookback//2).min()
    roll_mean = (roll_max + roll_min) / 2.0
    output = 40.0 * ((price - roll_mean) / (roll_max - roll_min))
    smoothed_output = output.ewm(span=smooth, min_periods=smooth//2).mean()
    scaled_output = (smoothed_output + 40.0) / 80.0
    return scaled_output.iloc[-1] if not scaled_output.empty else np.nan
def calculate_volatility_adjusted_z_score(full_prices_series, ticker="Unknown", metric="Z-score", sector=None):
    """
    Calculates a robust, volatility-adjusted Z-score for a price or relative strength series.

    The Z-score measures how far the current price (log-transformed) is from its rolling median,
    scaled by an adaptive dispersion measure. The dispersion is adjusted by recent volatility
    relative to historical volatility to make the score 'volatility-adjusted'.

    Args:
        full_prices_series (pd.Series): The full history of the price or relative strength series.
                                        This should ideally be the longest available series.
        ticker (str): Ticker symbol for logging.
        metric (str): Name of the metric for logging.
        sector (str): Sector for applying a heuristic adjustment.

    Returns:
        float: The volatility-adjusted Z-score, or np.nan if calculation fails.
    """
    if full_prices_series.empty or full_prices_series.isna().any() or (full_prices_series <= 0).any():
        logging.error(f"Invalid price data for {metric} calculation (Ticker: {ticker}): empty, contains NaN, or non-positive values")
        return np.nan

    min_data_for_vol_and_zscore = 252 # Minimum days needed for a stable annualized volatility and meaningful Z-score
    min_data_for_median_mad = 63 # Minimum days for robust median/MAD for the z-score window

    if len(full_prices_series) < min_data_for_vol_and_zscore:
        logging.error(f"Insufficient data length for {metric} calculation (Ticker: {ticker}): {len(full_prices_series)} < {min_data_for_vol_and_zscore} days")
        return np.nan

    # Calculate log returns for volatility measures
    daily_log_returns = np.log(full_prices_series / full_prices_series.shift(1)).dropna()
    
    if daily_log_returns.empty or len(daily_log_returns) < min_data_for_vol_and_zscore:
        logging.error(f"Insufficient valid return data for {metric} calculation (Ticker: {ticker}) after dropping NaNs (need {min_data_for_vol_and_zscore} days).")
        return np.nan

    # 1. Calculate historical volatility (longer term, more stable)
    historical_vol = daily_log_returns.std() * np.sqrt(252) # Annualized historical volatility from the entire series
    
    # 2. Define the current observation window for recent volatility
    recent_vol_window = 63 # approx 3 months
    if len(daily_log_returns) < recent_vol_window:
        logging.warning(f"Not enough recent data for current volatility for {ticker} ({len(daily_log_returns)} days). Using available length for recent vol.")
        current_vol = daily_log_returns.std() * np.sqrt(252) # Fallback to full series vol if not enough recent
    else:
        current_vol = daily_log_returns.tail(recent_vol_window).std() * np.sqrt(252) # Annualized recent volatility

    # Handle zero volatilities
    if historical_vol == 0 or np.isnan(historical_vol):
        vol_factor = 1.0
    else:
        vol_factor = current_vol / historical_vol
    
    # Adaptive window for Z-score calculation (for median/MAD of log prices)
    # This window influences how far back we look for the 'mean' (median_y) and 'std' (MAD)
    # It adapts based on vol_factor: higher volatility -> larger window for more robust stats; lower vol -> smaller, more responsive window.
    # Min 126 days (6 months), Max 504 days (2 years)
    adaptive_z_score_window = int(min(max(126, 252 * (1 + vol_factor)), 504))
    
    # Ensure the window does not exceed the available data length of the full_prices_series
    actual_z_score_window = min(len(full_prices_series), adaptive_z_score_window)

    if actual_z_score_window < min_data_for_median_mad:
        logging.warning(f"Insufficient data for Z-score median/MAD calculation for {ticker} (actual window {actual_z_score_window} < {min_data_for_median_mad}). Returning NaN.")
        return np.nan

    logging.info(f"Calculating {metric} for {ticker} with adaptive Z-score window {actual_z_score_window} days (vol_factor: {vol_factor:.2f})")

    # Use the adaptive window for log-transformed prices to calculate median and MAD
    y = np.log(full_prices_series.tail(actual_z_score_window)).values
    median_y = np.median(y)
    mad = np.median(np.abs(y - median_y))

    if mad == 0 or np.isnan(mad):
        logging.warning(f"Zero or NaN MAD in {metric} calculation for {ticker}")
        return np.nan

    # The robust Z-score calculation:
    # (Current log price - Median log price) / (Robust dispersion * Volatility scaling * Sector scaling)
    # 0.6745 is the factor to convert MAD to standard deviation for normal distribution.
    robust_z = 0.6745 * (y[-1] - median_y) / (mad * vol_factor)
    
    return robust_z

def recalculate_relative_z_scores(top_15_df, etf_histories, period="3y", window=252, min_window=200):
    """
    Recalculates relative Z-scores for a list of stocks against their benchmark ETFs.
    Modified to pass the full relative strength series to calculate_volatility_adjusted_z_score.
    """
    relative_z_scores = []
    for idx, row in top_15_df.iterrows():
        ticker = row['Ticker']
        best_factor = row['Best_Factor']
        sector = row['Sector']
        try:
            logging.info(f"Recalculating Z-Score for {ticker}, Best_Factor: {best_factor}")
            
            history = yf.Ticker(ticker).history(period=period, auto_adjust=True, interval="1d").tz_localize(None)
            etf_history = etf_histories.get(best_factor)

            if not history.empty and etf_history is not None and not etf_history.empty:
                etf_history = etf_history.copy()
                etf_history.index = etf_history.index.tz_localize(None)
                
                # Align the full series, then calculate the relative strength
                common_index_full = history.index.intersection(etf_history.index)
                
                if len(common_index_full) >= min_window: # Ensure enough data for a meaningful Z-score
                    aligned_stock_full = history['Close'][common_index_full]
                    aligned_etf_full = etf_history['Close'][common_index_full]
                    
                    if (aligned_etf_full == 0).any():
                        logging.warning(f"Zero prices found in ETF {best_factor} for {ticker}, cannot calculate relative strength.")
                        relative_z_scores.append(np.nan)
                        continue

                    full_relative_strength = aligned_stock_full / aligned_etf_full
                    
                    if not full_relative_strength.isna().any() and (full_relative_strength > 0).all() and np.isfinite(full_relative_strength).all():
                        # Pass the FULL relative strength series to the Z-score function
                        z_score = calculate_volatility_adjusted_z_score(full_relative_strength, ticker=ticker, sector=sector)
                        relative_z_scores.append(z_score)
                        logging.info(f"Calculated Z-Score for {ticker} vs {best_factor}: {z_score:.4f}")
                    else:
                        relative_z_scores.append(np.nan)
                        logging.warning(f"Invalid full relative strength data for {ticker} vs {best_factor}")
                else:
                    relative_z_scores.append(np.nan)
                    logging.warning(f"Insufficient data for {ticker} vs {best_factor}: {len(common_index_full)} days")
            else:
                relative_z_scores.append(np.nan)
                logging.warning(f"Empty history for {ticker} or {best_factor}")
        except Exception as e:
            relative_z_scores.append(np.nan)
            logging.error(f"Error recalculating Z-Score for {ticker}: {str(e)}")
            
    return relative_z_scores

def calculate_portfolio_relative_z_score(weighted_df, etf_histories, best_etf, period="3y", window=252, min_window=200):
    """
    Calculates the relative Z-score of the entire portfolio against its best-correlated ETF.
    Modified to pass the full relative strength series to calculate_volatility_adjusted_z_score.
    """
    portfolio_prices = None

    if 'Weight' not in weighted_df.columns or weighted_df['Weight'].sum() == 0:
        logging.warning("Invalid weights for portfolio Z-score calculation.")
        return np.nan, best_etf

    for idx, row in weighted_df.iterrows():
        ticker = row['Ticker']
        weight = row['Weight']
        try:
            history = yf.Ticker(ticker).history(period=period, auto_adjust=True, interval="1d").tz_localize(None)
            if history.empty or 'Close' not in history.columns or len(history) < 2:
                continue

            # Rebase individual stock prices to 100 at the start of its history for weighted sum
            # We need sufficient history for rebasing, otherwise `iloc[0]` might fail or be misleading.
            # Use `min_window` as a heuristic minimum for rebasing stability.
            if len(history) < min_window:
                logging.debug(f"Skipping {ticker} for portfolio Z-score: insufficient history ({len(history)} < {min_window})")
                continue

            rebased_prices = 100 * (history['Close'] / history['Close'].iloc[0])

            if portfolio_prices is None:
                portfolio_prices = rebased_prices * weight
            else:
                # Align indices when adding, fill NaNs with 0
                portfolio_prices = portfolio_prices.add(rebased_prices * weight, fill_value=0)

        except Exception as e:
            logging.error(f"Error getting history for {ticker} in portfolio Z-score calc: {e}")
            continue

    if portfolio_prices is None or portfolio_prices.empty:
        logging.error("Failed to construct portfolio price series for Z-score calculation.")
        return np.nan, best_etf

    portfolio_prices = portfolio_prices.dropna()
    if portfolio_prices.empty:
        logging.error("Portfolio price series is empty after dropping NaNs.")
        return np.nan, best_etf

    etf_prices_df = etf_histories.get(best_etf)
    if etf_prices_df is None or etf_prices_df.empty or len(etf_prices_df) < 2:
        logging.warning(f"No history for best ETF '{best_etf}' in portfolio Z-score calculation.")
        return np.nan, best_etf

    # Rebase ETF prices to 100 at its history start
    etf_prices = 100 * (etf_prices_df['Close'] / etf_prices_df['Close'].iloc[0])

    # Align portfolio and ETF prices (take the common period for the full relative strength calculation)
    # CRITICAL FIX: Removed [-window:] slicing here. Pass the FULL common history.
    common_index_full = portfolio_prices.index.intersection(etf_prices.index)

    if len(common_index_full) < min_window: # Ensure minimum history for Z-score calculation
        logging.warning(f"Insufficient overlap for full portfolio vs {best_etf}: {len(common_index_full)} days (need at least {min_window})")
        return np.nan, best_etf

    aligned_portfolio_full = portfolio_prices.loc[common_index_full]
    aligned_etf_full = etf_prices.loc[common_index_full]

    if (aligned_etf_full == 0).any(): 
        logging.warning(f"Zero prices found in ETF {best_etf}, cannot calculate relative strength for portfolio.")
        return np.nan, best_etf

    full_relative = aligned_portfolio_full / aligned_etf_full

    if not full_relative.isna().any() and (full_relative > 0).all() and np.isfinite(full_relative).all():
        # CRITICAL FIX: Removed `period=len(common_index)` as `calculate_volatility_adjusted_z_score` no longer uses it.
        # Pass the FULL `full_relative` series.
        z_score = calculate_volatility_adjusted_z_score(full_relative, ticker="Portfolio", sector=None) # Passed sector=None
        logging.info(f"Portfolio Z-Score vs {best_etf}: {z_score:.4f}")
        return z_score, best_etf

    logging.warning("Invalid portfolio full relative data after alignment.")
    return np.nan, best_etf
def calculate_piotroski_f_score(financials, balancesheet, cashflow, total_assets, roa, net_income):
    score = 0
    try:
        # ROA > 0
        if pd.notna(roa) and roa > 0: score += 1
        
        # Operating Cash Flow > 0
        op_cash_flow = get_value(cashflow, ['Operating Cash Flow', 'TotalCashFromOperatingActivities']) # Added alternative key
        if pd.notna(op_cash_flow) and op_cash_flow > 0: score += 1
        
        # OCF > Net Income (Accruals)
        if pd.notna(op_cash_flow) and pd.notna(net_income) and op_cash_flow > net_income: score += 1
        
        # Long Term Debt decrease
        long_term_debt_curr = get_value(balancesheet, ['Long Term Debt', 'LongTermDebt']) # Added alternative key
        long_term_debt_prev = get_value(balancesheet, ['Long Term Debt', 'LongTermDebt'], 1) # Added alternative key
        if pd.notna(long_term_debt_curr) and pd.notna(long_term_debt_prev) and long_term_debt_curr <= long_term_debt_prev: score += 1
        
        # Current Ratio increase
        current_assets_curr = get_value(balancesheet, ['Total Current Assets', 'CurrentAssets']) # Added alternative key
        current_liabilities_curr = get_value(balancesheet, ['Total Current Liabilities', 'CurrentLiabilities']) # Added alternative key
        current_ratio_curr = current_assets_curr / current_liabilities_curr if pd.notna(current_assets_curr) and pd.notna(current_liabilities_curr) and current_liabilities_curr != 0 else np.nan
        
        current_assets_prev = get_value(balancesheet, ['Total Current Assets', 'CurrentAssets'], 1) # Added alternative key
        current_liabilities_prev = get_value(balancesheet, ['Total Current Liabilities', 'CurrentLiabilities'], 1) # Added alternative key
        current_ratio_prev = current_assets_prev / current_liabilities_prev if pd.notna(current_assets_prev) and pd.notna(current_liabilities_prev) and current_liabilities_prev != 0 else np.nan
        
        if pd.notna(current_ratio_curr) and pd.notna(current_ratio_prev) and current_ratio_curr > current_ratio_prev: score += 1
        
        # Gross Margin increase
        gross_profit_curr = get_value(financials, ['Gross Profit', 'GrossProfit']) # Added alternative key
        total_revenue_curr = get_value(financials, ['Total Revenue', 'TotalRevenue']) # Added alternative key
        gross_margin_curr = gross_profit_curr / total_revenue_curr if pd.notna(gross_profit_curr) and pd.notna(total_revenue_curr) and total_revenue_curr != 0 else np.nan
        
        gross_profit_prev = get_value(financials, ['Gross Profit', 'GrossProfit'], 1) # Added alternative key
        total_revenue_prev = get_value(financials, ['Total Revenue', 'TotalRevenue'], 1) # Added alternative key
        gross_margin_prev = gross_profit_prev / total_revenue_prev if pd.notna(gross_profit_prev) and pd.notna(total_revenue_prev) and total_revenue_prev != 0 else np.nan
        
        if pd.notna(gross_margin_curr) and pd.notna(gross_margin_prev) and gross_margin_curr > gross_margin_prev: score += 1
        
        # Asset Turnover increase
        total_revenue_curr = get_value(financials, ['Total Revenue', 'TotalRevenue']) # Added alternative key
        total_assets_curr = total_assets # Already provided as argument
        asset_turnover_curr = total_revenue_curr / total_assets_curr if pd.notna(total_revenue_curr) and pd.notna(total_assets_curr) and total_assets_curr != 0 else np.nan
        
        total_revenue_prev = get_value(financials, ['Total Revenue', 'TotalRevenue'], 1) # Added alternative key
        total_assets_prev = get_value(balancesheet, ['Total Assets', 'TotalAssets'], 1) # Added alternative key
        asset_turnover_prev = total_revenue_prev / total_assets_prev if pd.notna(total_revenue_prev) and pd.notna(total_assets_prev) and total_assets_prev != 0 else np.nan
        
        if pd.notna(asset_turnover_curr) and pd.notna(asset_turnover_prev) and asset_turnover_curr > asset_turnover_prev: score += 1

        # Share Issuances / Dilution (should be 0 for no new shares issued)
        # Using CommonStockSharesOutstanding from balancesheet, index 0 vs index 1
        shares_curr = get_value(balancesheet, ['Common Stock Shares Outstanding', 'CommonStockSharesOutstanding'])
        shares_prev = get_value(balancesheet, ['Common Stock Shares Outstanding', 'CommonStockSharesOutstanding'], 1)
        if pd.notna(shares_curr) and pd.notna(shares_prev) and shares_curr <= shares_prev: score += 1
        
    except Exception as e: # Added specific exception and logging
        logging.warning(f"Error calculating Piotroski F-Score: {e}")
    return score

def calculate_lo_modified_variance(sub_series, q):
    n = len(sub_series)
    if n < 2: return np.nan
    mean_val = np.mean(sub_series)
    if np.allclose(sub_series, mean_val): return 0.0
    sample_var = np.sum((sub_series - mean_val)**2) / n
    if sample_var < 0: return np.nan
    if q <= 0 or q >= n: return sample_var
    try:
        autocovariances = smt.acovf(sub_series, adjusted=False, fft=True, nlag=q)[1:]
        if len(autocovariances) != q: return sample_var
    except Exception: return sample_var
    autocovariance_sum = 0.0
    for j in range(q):
        weight = 1.0 - ((j + 1) / (q + 1.0))
        autocovariance_sum += weight * autocovariances[j]
    modified_var = sample_var + 2.0 * autocovariance_sum
    return max(0.0, modified_var)

def calculate_hurst_lo_modified(series, min_n=10, max_n=None, q_method='auto'):
    if isinstance(series, pd.Series): series = series.values
    series = series[~np.isnan(series)]
    N = len(series)
    if max_n is None: max_n = N // 2
    max_n = min(max_n, N - 1)
    min_n = max(2, min_n) # Ensure min_n is at least 2 for diff/variance
    if N < 20 or min_n >= max_n: return np.nan, pd.DataFrame() # Not enough data or invalid range
    
    # Generate unique intervals (scales)
    ns = np.unique(np.geomspace(min_n, max_n, num=20, dtype=int))
    ns = [n_val for n_val in ns if n_val >= min_n] # Filter to ensure >= min_n
    if not ns: return np.nan, pd.DataFrame() # No valid intervals

    rs_values, valid_ns = [], []
    for n in ns:
        if n < 2: continue # Cannot compute R/S for n<2
        
        q = 0
        if isinstance(q_method, int): q = max(0, min(q_method, n - 1))
        elif q_method == 'auto' and n > 10: q = max(0, min(int(np.floor(1.1447 * (n**(1/3)))), n - 1))
        
        rs_chunk = []
        num_chunks = N // n
        if num_chunks == 0: continue # No full chunks of size n
        
        for i in range(num_chunks):
            chunk = series[i*n : (i+1)*n]
            if len(chunk) < 2: continue # Chunk too small
            
            mean = np.mean(chunk)
            if np.allclose(chunk, mean, rtol=1e-8, atol=1e-10): continue # Constant chunk

            mean_adjusted = chunk - mean
            cum_dev = np.cumsum(mean_adjusted)
            R = np.ptp(cum_dev) # Range of cumulative deviations
            
            if pd.isna(R) or R < 0: continue # Invalid range

            modified_var = calculate_lo_modified_variance(chunk, q)
            if pd.isna(modified_var) or modified_var < 1e-12: continue # Invalid or zero variance

            S_q = np.sqrt(modified_var)
            rs = R / S_q
            if not pd.isna(rs) and rs >= 0: rs_chunk.append(rs)
        
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
            valid_ns.append(n)
    
    if len(valid_ns) < 3: return np.nan, pd.DataFrame() # Need at least 3 points for a robust regression
    results_df = pd.DataFrame({'interval': valid_ns, 'rs_mean': rs_values}).dropna()
    if len(results_df) < 3: return np.nan, pd.DataFrame() # After dropping NaNs

    try:
        # Log-log regression to find Hurst exponent
        # Ensure log values are finite
        log_interval = np.log(results_df['interval'])
        log_rs_mean = np.log(results_df['rs_mean'])
        
        if not np.isfinite(log_interval).all() or not np.isfinite(log_rs_mean).all(): # Added check for finite log values
            logging.warning("Non-finite log values for Hurst regression.") # Added logging
            return np.nan, pd.DataFrame()

        hurst, _, _, _, _ = linregress(log_interval, log_rs_mean)
        return hurst, results_df
    except Exception as e: # Catch specific exception
        logging.warning(f"Error in Hurst regression: {e}. Returning NaN.") # Added logging
        return np.nan, pd.DataFrame()

# Add this to SECTION 1 with your other functions
@st.cache_data
def get_normalized_prices(_returns_dict):
    """
    Calculates volatility-normalized prices for all tickers.
    """
    # Create a DataFrame of prices from log returns
    all_prices_df = pd.DataFrame({
        ticker: np.exp(rets.cumsum()) * 100  # Start all at 100
        for ticker, rets in _returns_dict.items()
    })
    
    # Calculate rolling 3-month volatility
    all_vols_df = all_prices_df.pct_change().rolling(window=63, min_periods=20).std()
    
    # Normalize price by its volatility
    normalized_prices = (all_prices_df / all_vols_df).ffill().bfill()
    return normalized_prices
def calculate_relative_carry(results_df):
    """
    Calculates the relative carry for each stock against its sector median.
    Rx_t = Cx_t - median(C_sector_t)
    """
    if 'Carry' not in results_df.columns or 'Sector' not in results_df.columns: # Added robustness check
        st.warning("Cannot calculate Relative Carry: Missing 'Carry' or 'Sector' column. Setting to 0.0")
        results_df['Relative_Carry'] = 0.0
        return results_df

    # Calculate the median carry for each sector
    # Ensure 'Carry' column is numeric and handle NaNs gracefully
    results_df['Carry_Numeric'] = pd.to_numeric(results_df['Carry'], errors='coerce') # Added explicit numeric conversion
    
    if results_df['Carry_Numeric'].isnull().all(): # Added check for all NaN carry
        st.warning("All 'Carry' values are NaN. Cannot calculate Relative Carry. Setting to 0.0")
        results_df['Relative_Carry'] = 0.0
        return results_df

    # Group by sector and transform to get the median for each stock's sector
    sector_median_carry = results_df.groupby('Sector')['Carry_Numeric'].transform('median')
    
    # The final signal is the stock's carry minus its sector's median carry
    results_df['Relative_Carry'] = results_df['Carry_Numeric'] - sector_median_carry
    
    # Clean up the temporary column
    results_df = results_df.drop(columns=['Carry_Numeric'])

    return results_df
def simulate_historical_pure_returns(pure_returns_today):
    """
    SIMULATES a history of past Pure Factor Return tables.
    """
    if pure_returns_today is None:
        return []
    
    historical_data = []
    for i in range(12): # Simulate 12 past "monthly" runs
        noise = np.random.normal(0, 0.5, len(pure_returns_today))
        drift = (12 - i) / 12 * 0.1
        simulated_series = pure_returns_today + noise + drift
        historical_data.append(simulated_series)
        
    historical_data.append(pure_returns_today)
    return historical_data


def analyze_coefficient_stability(historical_data):
    """
    Analyzes the stability of factor coefficients over time to find the most robust factors.
    """
    if not historical_data:
        return pd.DataFrame()

    # Ensure all series in historical_data have the same index for concatenation
    # Reindex all to the union of all indices or a common, filtered set
    all_indices = pd.Index([])
    for s in historical_data:
        all_indices = all_indices.union(s.index)
    
    # Pad shorter series with NaN if their factor was not present at that time
    df = pd.concat([s.reindex(all_indices) for s in historical_data], axis=1)
    df.columns = [f'run_{i}' for i in range(len(df.columns))]
    
    stability_metrics = pd.DataFrame(index=df.index)
    stability_metrics['mean_coeff'] = df.mean(axis=1)
    stability_metrics['std_coeff'] = df.std(axis=1)
    
    # Handle std_coeff being zero to prevent division by zero in Sharpe
    stability_metrics['sharpe_ratio_coeff'] = stability_metrics['mean_coeff'] / (stability_metrics['std_coeff'].replace(0, np.nan)) # Replaced +1e-6 with replace(0,np.nan)
    
    # For pct_positive, we need to handle NaNs correctly. Use .count() as total, not len(df.columns)
    # as some factors might not be present in all runs.
    stability_metrics['pct_positive'] = (df > 0).sum(axis=1) / df.count(axis=1).replace(0, np.nan) # Used df.count(axis=1).replace(0, np.nan)
    stability_metrics['pct_positive'].fillna(0, inplace=True) # Fill NaNs (e.g. if factor was never present)

    # Sort by absolute Sharpe to find most robust factors (positive or negative)
    return stability_metrics.sort_values(by='sharpe_ratio_coeff', key=abs, ascending=False).fillna(0) # Fill NaNs after sorting for display

def set_weights_from_stability(stability_df, all_metrics, reverse_metric_map):
    """
    Sets final portfolio weights based on the factor's stability (Coefficient Sharpe Ratio).
    """
    if stability_df.empty or 'sharpe_ratio_coeff' not in stability_df.columns:
        return {metric: 0.0 for metric in all_metrics}, pd.DataFrame()

    scores = stability_df['sharpe_ratio_coeff'].abs()
    
    total_score = scores.sum()
    if total_score == 0:
        return {metric: 0.0 for metric in all_metrics}, stability_df # Return zero weights and original df

    final_weights = (scores / total_score) * 100
    
    final_weights_dict = {metric: 0.0 for metric in all_metrics}
    for short_name, weight in final_weights.items():
        long_name = METRIC_NAME_MAP.get(short_name, short_name)
        if long_name in all_metrics: # Ensure that `long_name` is one of the keys in `default_weights` (which `all_metrics` is based on)
            final_weights_dict[long_name] = weight
            
    # Add 'Final_Weight' to stability_df based on short_name (its index)
    stability_df['Final_Weight'] = final_weights
    stability_df.fillna(0, inplace=True) # This inplace is safe
    
    return final_weights_dict, stability_df

def check_multicollinearity(X, characteristics, vif_threshold=10):
    """
    Checks for multicollinearity and removes offending features.
    This version is more robust against perfect multicollinearity.
    """
    if X.empty or X.shape[1] < 2:
        return characteristics

    # 1. Start with a clean slate
    X_clean = X.copy().replace([np.inf, -np.inf], np.nan)
    
    # 2. Drop columns that are entirely NaN after cleaning
    X_clean.dropna(axis=1, how='all', inplace=True)
    
    # 3. Fill remaining NaNs using the median
    for col in X_clean.columns:
        if X_clean[col].isnull().any():
            X_clean[col].fillna(X_clean[col].median(), inplace=True)

    # 4. Remove columns with zero or near-zero variance (constants)
    # This is a key step to prevent VIF errors.
    non_zero_var_cols = [col for col in X_clean.columns if X_clean[col].var() > 1e-9]
    X_filtered = X_clean[non_zero_var_cols]
    
    if X_filtered.shape[1] < 2:
        return X_filtered.columns.tolist()

    # 5. Remove one of each pair of highly correlated features
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)] # Slightly lower threshold
    X_filtered.drop(columns=to_drop, inplace=True)
    
    if X_filtered.shape[1] < 2:
        return X_filtered.columns.tolist()

    # 6. Iteratively calculate VIF and remove the worst offender until all are below the threshold.
    # This is more robust than a single pass.
    final_characteristics = X_filtered.columns.tolist()
    while True:
        vif_data = pd.DataFrame()
        try:
            vif_data["feature"] = X_filtered.columns
            vif_data["VIF"] = [variance_inflation_factor(X_filtered.values, i) for i in range(X_filtered.shape[1])]
        except Exception as e:
            logging.error(f"Error during VIF calculation: {e}. Returning current set of features.")
            return final_characteristics
        
        max_vif = vif_data['VIF'].max()
        if max_vif < vif_threshold:
            break
        
        # Remove the feature with the highest VIF and repeat
        feature_to_remove = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
        X_filtered.drop(columns=[feature_to_remove], inplace=True)
        final_characteristics.remove(feature_to_remove)
        
        if X_filtered.shape[1] < 2:
            break
            
    return final_characteristics

def calculate_portfolio_factor_correlations(weighted_df, etf_histories, period="3y", min_days=240):
    """
    Calculates the correlation of a weighted portfolio's returns against a list of ETF returns.
    """
    logging.info("Starting robust portfolio factor correlations calculation")
    correlations = pd.Series(dtype=float)

    portfolio_returns = None
    if 'Weight' not in weighted_df.columns or weighted_df.empty:
        logging.warning("Weighted DataFrame is empty or missing 'Weight' column.")
        return correlations

    for idx, row in weighted_df.iterrows():
        ticker = row['Ticker']
        weight = row['Weight']
        try:
            history = yf.Ticker(ticker).history(period=period, auto_adjust=True, interval="1d").tz_localize(None)
            if history.empty or 'Close' not in history.columns:
                continue

            # FIX: Set fill_method to None to resolve FutureWarning
            returns = history['Close'].pct_change(fill_method=None).dropna()

            if returns.empty:
                continue

            if portfolio_returns is None:
                portfolio_returns = returns * weight
            else:
                portfolio_returns = portfolio_returns.add(returns * weight, fill_value=0)
        except Exception as e:
            logging.error(f"Error fetching history for {ticker} in factor correlation: {e}")
            continue

    if portfolio_returns is None or portfolio_returns.empty:
        logging.error("Failed to compute portfolio returns for factor correlation.")
        return correlations

    portfolio_returns = portfolio_returns.dropna()

    for etf, etf_history in etf_histories.items():
        if etf_history.empty or 'Close' not in etf_history.columns:
            continue
        try:
            # FIX: Set fill_method to None to resolve FutureWarning
            etf_returns = etf_history['Close'].pct_change(fill_method=None).dropna()

            if etf_returns.empty:
                continue
            
            common_index = portfolio_returns.index.intersection(etf_returns.index)
            if len(common_index) < min_days:
                continue
                
            aligned_portfolio = portfolio_returns.loc[common_index]
            aligned_etf = etf_returns.loc[common_index]

            # Ensure there is variance in both series for correlation
            if aligned_portfolio.std() < 1e-6 or aligned_etf.std() < 1e-6: # Added variance check
                corr = 0.0
            else:
                corr = aligned_portfolio.corr(aligned_etf)
            
            if np.isfinite(corr):
                correlations[etf] = corr
                
        except Exception as e:
            logging.error(f"Error calculating correlation for ETF {etf}: {e}")
            continue
            
    if correlations.empty:
        logging.warning("No valid correlations computed, falling back to SPY")
        spy_history = etf_histories.get('SPY')
        if spy_history is not None and not spy_history.empty:
            # FIX: Set fill_method to None to resolve FutureWarning
            spy_returns = spy_history['Close'].pct_change(fill_method=None).dropna()
            common_index = portfolio_returns.index.intersection(spy_returns.index)
            if len(common_index) >= min_days:
                if portfolio_returns.loc[common_index].std() < 1e-6 or spy_returns.loc[common_index].std() < 1e-6: # Added variance check
                    corr = 0.0
                else:
                    corr = portfolio_returns.loc[common_index].corr(spy_returns.loc[common_index])
                correlations['SPY'] = corr if np.isfinite(corr) else 0.5
            else:
                correlations['SPY'] = 0.5
    
    return correlations.sort_values(ascending=False)

def aggregate_stability_and_set_weights(stability_results, all_metrics, reverse_metric_map):
    """
    Aggregates stability metrics from multiple time horizons and sets final portfolio weights.
    """
    if not stability_results:
        return {metric: 0.0 for metric in all_metrics}, pd.DataFrame()

    all_factors = set()
    for horizon, df in stability_results.items():
        all_factors.update(df.index)

    agg_df = pd.DataFrame(index=list(all_factors))
    agg_df['avg_sharpe_coeff'] = 0.0
    agg_df['consistency_score'] = 0.0
    agg_df['horizons_present'] = 0

    for factor in agg_df.index:
        sharpes = []
        for horizon, df in stability_results.items():
            if factor in df.index:
                sharpes.append(df.loc[factor, 'sharpe_ratio_coeff'])
        
        if not sharpes:
            continue
            
        avg_sharpe = np.mean(sharpes)
        agg_df.loc[factor, 'avg_sharpe_coeff'] = avg_sharpe
        agg_df.loc[factor, 'horizons_present'] = len(sharpes)

        if avg_sharpe != 0:
            sign_of_avg = np.sign(avg_sharpe)
            # Count how many individual sharpes have the same sign as the average
            same_sign_count = sum(1 for s in sharpes if np.sign(s) == sign_of_avg and not np.isclose(s, 0)) # Added np.isclose(s, 0) for robustness
            agg_df.loc[factor, 'consistency_score'] = same_sign_count / len(sharpes)
        else:
            agg_df.loc[factor, 'consistency_score'] = 0.0

    # Calculate Final_Score ensuring non-negative values for weighting
    agg_df['Final_Score'] = agg_df['avg_sharpe_coeff'].abs() * (agg_df['consistency_score'] ** 2)
    agg_df = agg_df.sort_values('Final_Score', ascending=False).fillna(0)
    
    total_score = agg_df['Final_Score'].sum()
    if total_score > 0:
        agg_df['Final_Weight'] = (agg_df['Final_Score'] / total_score) * 100
    else:
        agg_df['Final_Weight'] = 0.0

    final_weights_dict = {metric: 0.0 for metric in all_metrics}
    for short_name, row in agg_df.iterrows():
        long_name = METRIC_NAME_MAP.get(short_name, short_name)
        if long_name in all_metrics: # Ensure it maps to a valid metric in default_weights (i.e. all_metrics)
            final_weights_dict[long_name] = row['Final_Weight']
            
    return final_weights_dict, agg_df   

def calculate_pure_returns(df, characteristics, target='Return_252d', vif_threshold=5, use_pca=True, pca_variance_threshold=0.95):
    """
    Calculates pure factor returns using a robust cross-sectional regression with PCA.
    """
    if df.empty or target not in df.columns or df[target].isnull().all(): # Added df.empty check
        return pd.Series(dtype=float, name="PureReturns")
    
    y = pd.to_numeric(df[target], errors='coerce')
    valid_characteristics = [col for col in characteristics if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    X = df[valid_characteristics].copy().replace([np.inf, -np.inf], np.nan)
    
    valid_indices = y.dropna().index
    X, y = X.loc[valid_indices], y.loc[valid_indices]
    if X.empty or y.empty or len(y) < 20:
        logging.warning(f"Insufficient data for pure returns calculation: {len(y)} samples.")
        return pd.Series(dtype=float, name="PureReturns")

    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
        # Apply log1p only if values are consistently positive and large
        if X[col].min() >= 0 and X[col].quantile(0.75) > 1000: 
            X[col] = np.log1p(X[col])
        # Add small noise to constant columns to avoid issues with scaling/regression
        if X[col].var() < 1e-8: 
            X[col] = X[col] + np.random.normal(0, 1e-4, len(X))
    
    final_characteristics = check_multicollinearity(X, valid_characteristics, vif_threshold) # Refined check_multicollinearity
    if not final_characteristics:
        logging.warning("No valid characteristics left after VIF check.")
        return pd.Series(dtype=float, name="PureReturns")
    
    X = X[final_characteristics]
    
    # Check if X is still valid after filtering
    if X.empty or X.shape[1] == 0: # Added check for empty X after filtering
        logging.warning("No valid features left for scaling after multicollinearity check.")
        return pd.Series(dtype=float, name="PureReturns")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        model = Ridge(alpha=1.0, solver='auto')
        
        if use_pca and X_scaled.shape[1] > 1: # Only use PCA if multiple features # Added condition for PCA
            pca = PCA(n_components=pca_variance_threshold)
            X_pca = pca.fit_transform(X_scaled)
            model.fit(X_pca, y)
            original_space_coefs = pca.inverse_transform(model.coef_.reshape(1, -1))[0]
        else:
            model.fit(X_scaled, y)
            original_space_coefs = model.coef_

        # Ensure scaler_scale has the correct length and prevent division by zero
        scaler_scale = scaler.scale_
        if len(scaler_scale) != len(original_space_coefs): # Added length check
            logging.warning("Scaler scale length mismatch. Cannot unscale coefficients accurately.")
            unscaled_coefs = original_space_coefs
        else:
            scaler_scale[scaler_scale < 1e-8] = 1e-8 # Prevent division by near-zero
            unscaled_coefs = original_space_coefs / scaler_scale
        
        # Clip coefficients to prevent extreme values, which can happen with noisy data
        return pd.Series(unscaled_coefs, index=final_characteristics, name="PureReturns").clip(lower=-10.0, upper=10.0)
        
    except Exception as e:
        logging.error(f"Pure returns regression failed: {e}")
        return pd.Series(dtype=float, name="PureReturns")
# We will use a simplified vol calculation here for demonstration.
# --- ADD THIS NEW FUNCTION to SECTION 1 ---
def calculate_industry_relative_factors(df):
    """
    Calculates industry-relative factors after all tickers have been processed.
    This is a post-processing step.
    """
    st.write("Calculating industry-relative factors...")
    if 'Sector' not in df.columns or df.empty:
        return df

    # List of base factors that have an industry-relative version
    # Example: We calculated 'TobinQ', now we calculate 'IndRel_TobinQ'
    factors_to_adjust = [
        'TobinQ', 'SusGrwRate', 'CashCycle', 'DA', 'WCAccruals', 'ShareChg',
        # Add the base name for every 'IndRel_' factor you implement
    ]

    for factor in factors_to_adjust:
        ind_rel_factor = f"IndRel_{factor}"
        if factor in df.columns and ind_rel_factor in df.columns:
            # Calculate as: Value - Industry Median
            industry_median = df.groupby('Sector')[factor].transform('median')
            df[ind_rel_factor] = df[factor] - industry_median
    
    st.write("Industry-relative calculations complete.")
    return df

# --- IN YOUR `main()` FUNCTION, ADD THIS CALL ---
# Find this line:
# results_df, failed_tickers, returns_dict = process_tickers(tickers, etf_histories, sector_etf_map)

# And ADD the new function call immediately after it:
@lru_cache(maxsize=1024)
def calculate_returns_cached(ticker, periods_tuple):
    periods = list(periods_tuple)
    try:
        # We need a longer period to ensure we can calculate 252-day returns,
        # but also sufficient for shorter periods. "2y" is a good balance.
        # Max period requested is 252 days, so 2 years (approx 504 days) of history is sufficient.
        history = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
        if history.empty or len(history) < max(p for p in periods if p is not None):
            return {f"Return_{p}d": np.nan for p in periods}
        
        returns = {}
        for period in periods:
            if period is not None and len(history) > period:
                # Calculate simple percentage return (current / past - 1)
                returns[f"Return_{period}d"] = (history['Close'].iloc[-1] / history['Close'].iloc[-(period+1)] - 1) * 100
            else:
                returns[f"Return_{period}d":] = np.nan # Corrected slicing for assignment
        return returns
    except Exception as e:
        logging.error(f"Error calculating returns for {ticker} for periods {periods_tuple}: {e}")
        return {f"Return_{p}d": np.nan for p in periods}


def process_single_ticker(ticker_symbol, etf_histories, sector_etf_map):
    """
    Processes a single ticker to calculate a wide range of fundamental, technical, and advanced quantitative factors.
    This function is designed to be called concurrently for multiple tickers.
    
    It fetches annual and quarterly financial statements, price history, and company info.
    It then computes over 100 distinct metrics and returns them in a structured list,
    along with the ticker's log returns for further portfolio analysis.
    
    Factors that cannot be calculated due to missing data or because they require external
    (e.g., analyst estimate) data sources are gracefully handled and set to np.nan.
    """
    try:
        # Step 1: Fetch all necessary data for the ticker
        _, history, info, financials, balancesheet, cashflow, quarterly_financials, quarterly_balancesheet, quarterly_cashflow = fetch_ticker_data(ticker_symbol)

        # Initial validation: If core data is missing, we can't proceed.
        if history.empty or not info:
            failed_data = {col: np.nan for col in columns}
            failed_data['Ticker'] = ticker_symbol
            failed_data['Name'] = f"{ticker_symbol} (Failed to fetch history or info)"
            return [failed_data.get(col) for col in columns], pd.Series(dtype=float)

        # Step 2: Initialize data dictionary and basic info
        data = {col: np.nan for col in columns} # Initialize all columns with NaN
        data['Ticker'] = ticker_symbol
        data['Name'] = info.get('longName', 'N/A')
        data['Sector'] = info.get('sector', 'Unknown')
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        shares_outstanding = info.get('sharesOutstanding')
        market_cap = info.get('marketCap')

        # Step 3: Pre-extract a wide range of financial data points (annual) for multiple years
        # This centralizes data fetching and makes subsequent calculations cleaner.
        # Year 0 is the most recent year, Year 1 is the year before, etc.
        revenue = get_value(financials, ['Total Revenue', 'TotalRevenue']) or 0
        gross_profit = get_value(financials, ['Gross Profit', 'GrossProfit']) or 0
        net_income = get_value(financials, ['Net Income', 'NetIncome']) or 0
        operating_income = get_value(financials, ['Operating Income', 'OperatingIncome']) or 0
        interest_expense = get_value(financials, ['Interest Expense', 'InterestExpense']) or 0
        ebit = get_value(financials, ['Ebit', 'EBIT']) or 0
        total_assets = get_value(balancesheet, ['Total Assets', 'TotalAssets']) or 0
        total_liabilities = get_value(balancesheet, ['Total Liabilities', 'TotalLiab']) or 0
        intangibles = (get_value(balancesheet, ['Intangible Assets', 'IntangibleAssets']) or 0) + (get_value(balancesheet, ['Goodwill']) or 0)
        total_equity = get_value(balancesheet, ['Total Stockholder Equity', 'TotalStockholderEquity']) or 0
        current_assets = get_value(balancesheet, ['Total Current Assets', 'CurrentAssets']) or 0
        current_liabilities = get_value(balancesheet, ['Total Current Liabilities', 'CurrentLiabilities']) or 0
        inventory = get_value(balancesheet, ['Inventory']) or 0
        cogs = get_value(financials, ['Cost Of Revenue', 'CostOfRevenue']) or 0
        operating_cash_flow = get_value(cashflow, ['Operating Cash Flow', 'TotalCashFromOperatingActivities']) or 0
        capex = get_value(cashflow, ['Capital Expenditure', 'CapitalExpenditures']) or 0
        depreciation = get_value(cashflow, ['Depreciation And Amortization', 'Depreciation']) or 0
        dividends_paid = get_value(cashflow, ['Dividends Paid', 'DividendsPaid']) or 0
        buybacks = get_value(cashflow, ['Repurchase Of Capital Stock', 'RepurchaseOfStock']) or 0
        fcf = operating_cash_flow + capex if pd.notna(operating_cash_flow) and pd.notna(capex) else np.nan
        
        # Historical values for growth calculations
        revenue_y1 = get_value(financials, ['Total Revenue', 'TotalRevenue'], 1)
        revenue_y3 = get_value(financials, ['Total Revenue', 'TotalRevenue'], 3)
        total_assets_y1 = get_value(balancesheet, ['Total Assets', 'TotalAssets'], 1)
        eps_y0 = get_value(financials, ['Diluted EPS'], 0)
        eps_y3 = get_value(financials, ['Diluted EPS'], 3)
        shares_y0 = get_value(balancesheet, ['Common Stock Shares Outstanding'], 0)
        shares_y1 = get_value(balancesheet, ['Common Stock Shares Outstanding'], 1)
        
        # Step 4: Calculate Factors - Grouped by Category
        
        # --- A. Existing Core Factors (Recalculated with pre-fetched data for consistency) ---
        data['Market_Cap'] = market_cap
        data['Dividend_Yield'] = info.get('dividendYield', 0) * 100
        data['PE_Ratio'] = info.get('trailingPE')
        data['EPS_Diluted'] = info.get('trailingEps')
        data['Insider_Ownership_Ratio'] = info.get('heldPercentInsiders', 0) * 100
        data['Institutional_Ownership_Ratio'] = info.get('heldPercentInstitutions', 0) * 100
        data['Audit_Risk'], data['Board_Risk'], data['Compensation_Risk'], data['Shareholder_Rights_Risk'], data['Overall_Risk'] = \
            info.get('auditRisk',np.nan), info.get('boardRisk',np.nan), info.get('compensationRisk',np.nan), info.get('shareHolderRightsRisk',np.nan), info.get('overallRisk',np.nan)

        data['Current_Ratio'] = _safe_division(current_assets, current_liabilities)
        data['Quick_Ratio'] = _safe_division(current_assets - inventory, current_liabilities)
        data['Debt_Ratio'] = _safe_division(total_liabilities, total_assets)
        data['Liabilities_to_Equity'] = _safe_division(total_liabilities, total_equity)
        data['Gross_Profit_Margin'] = _safe_division(gross_profit, revenue) * 100
        data['Operating_Margin'] = _safe_division(operating_income, revenue) * 100
        data['Net_Profit_Margin'] = _safe_division(net_income, revenue) * 100
        data['ROA'] = _safe_division(net_income, total_assets) * 100
        data['ROE'] = _safe_division(net_income, total_equity) * 100
        data['PS_Ratio'] = _safe_division(market_cap, revenue)
        data['FCF_Yield'] = _safe_division(fcf, market_cap) * 100
        data['Sales_Per_Share'] = _safe_division(revenue, shares_outstanding)
        data['FCF_Per_Share'] = _safe_division(fcf, shares_outstanding)
        data['Asset_Turnover'] = _safe_division(revenue, total_assets)
        data['CapEx_to_DepAmor'] = _safe_division(abs(capex), depreciation)
        data['Dividends_to_FCF'] = _safe_division(abs(dividends_paid), fcf) if fcf > 0 else np.nan
        data['Interest_Coverage'] = _safe_division(ebit, abs(interest_expense))
        data['Inventory_Turnover'] = _safe_division(cogs, inventory)
        data['Share_Buyback_to_FCF'] = _safe_division(abs(buybacks), fcf) if fcf > 0 else np.nan
        data['Dividends_Plus_Buyback_to_FCF'] = _safe_division(abs(dividends_paid) + abs(buybacks), fcf) if fcf > 0 else np.nan
        data['Earnings_Yield'] = _safe_division(data['EPS_Diluted'], current_price) * 100
        data['FCF_to_Net_Income'] = _safe_division(fcf, net_income) if net_income > 0 else np.nan
        data['Tangible_Book_Value'] = total_assets - intangibles - total_liabilities
        data['Return_On_Tangible_Equity'] = _safe_division(net_income, data['Tangible_Book_Value']) * 100
        data['Earnings_Growth_Rate_5y'] = info.get('earningsGrowth', np.nan) * 100
        data['Revenue_Growth_Rate_5y'] = info.get('revenueGrowth', np.nan) * 100
        data['Piotroski_F-Score'] = calculate_piotroski_f_score(financials, balancesheet, cashflow, total_assets, data['ROA'], net_income)
        nopat = operating_income * (1 - 0.25) if pd.notna(operating_income) else np.nan
        invested_capital = total_assets - current_liabilities if pd.notna(total_assets) and pd.notna(current_liabilities) else np.nan
        data['ROIC'] = _safe_division(nopat, invested_capital) * 100
        data['Cash_ROIC'] = _safe_division(fcf, invested_capital) * 100
        
        # --- B. NEWLY IMPLEMENTED FACTORS ---
        
        # B.1. Size Factors
        data['LogMktCap'] = np.log(market_cap) if market_cap and market_cap > 0 else np.nan
        data['LogMktCapCubed'] = data['LogMktCap'] ** 3 if pd.notna(data['LogMktCap']) else np.nan
        data['LogAssets'] = np.log(total_assets) if total_assets and total_assets > 0 else np.nan

        # B.2. Quality & Solvency Factors
        data['Altman_ZScore'] = _calculate_altman_z_score(info, financials, balancesheet, current_price)
        data['LTDE'] = _safe_division(long_term_debt_y0, total_equity)
        data['CashRatio'] = _safe_division(cash_and_equiv_y0, current_liabilities)
        payout_ratio = info.get('payoutRatio', np.nan)
        data['SusGrwRate'] = (data['ROE'] / 100 * (1 - payout_ratio)) * 100 if pd.notna(data['ROE']) and pd.notna(payout_ratio) else np.nan
        data['AstGrwth'] = _safe_division(total_assets - total_assets_y1, abs(total_assets_y1)) * 100
        data['FinLev'] = _safe_division(total_assets, total_equity)
        
        # B.3. Growth & Stability Factors
        data['3YAvgAnnSalesGrw'] = _calculate_annualized_growth(revenue, revenue_y3, 3)
        data['3YAvgAnnEPSGrw'] = _calculate_annualized_growth(eps_y0, eps_y3, 3)
        data['ShareChg'] = _safe_division(shares_y0 - shares_y1, abs(shares_y1)) * 100

        # --- C. Time-Series, Price, Volume & Technical Indicators ---
        log_returns = pd.Series(dtype=float) # Initialize
        if not history.empty and 'Close' in history.columns and len(history) > 1:
            log_returns = np.log(history['Close'] / history['Close'].shift(1)).dropna()
            daily_returns = history['Close'].pct_change().dropna()
            
            # C.1. Your Original Advanced Metrics
            data['GARCH_Vol'] = calculate_garch_volatility(log_returns)
            data['AR_Coeff'] = calculate_ar_coefficient(log_returns)
            data['Log_Log_Utility'] = calculate_log_log_utility(log_returns)
            data['Log_Log_Sharpe'] = calculate_log_log_sharpe(log_returns)
            data['Vol_Autocorr'] = calculate_volatility_autocorrelation(log_returns)
            data['Stop_Loss_Impact'] = calculate_stop_loss_impact(log_returns)
            hurst, _ = calculate_hurst_lo_modified(log_returns)
            data['Hurst_Exponent'] = hurst
            data['Trend'] = breakout(history['Close'])
            if len(history) >= 90:
                data['Dollar_Volume_90D'] = (history['Volume'] * history['Close']).rolling(90).mean().iloc[-1]
            if len(history) > 252:
                data['Momentum'] = (history['Close'].iloc[-1] / history['Close'].iloc[-252] - 1) * 100

            # C.2. NEWLY ADDED Technical and Price-based Factors
            data['14DayRSI'] = _calculate_rsi(history['Close'], window=14)
            macd_line, _ = _calculate_macd(history['Close'], fast_window=12, slow_window=26)
            data['10DMACD'] = macd_line # Note: this is a standard 12/26 MACD
            stoch_k, _ = _calculate_stochastic_oscillator(history['High'], history['Low'], history['Close'], k_window=14)
            data['20DStochastic'] = stoch_k # Note: this is a standard 14-period stochastic
            
            data['AnnVol1M'] = daily_returns.tail(21).std() * np.sqrt(BUSINESS_DAYS_IN_YEAR) * 100
            data['AnnVol12M'] = daily_returns.tail(252).std() * np.sqrt(BUSINESS_DAYS_IN_YEAR) * 100
            
            if len(history) > 22: data['PM1M'] = (history['Close'].iloc[-1] / history['Close'].iloc[-22] - 1) * 100
            if len(history) > 126: data['PM6M'] = (history['Close'].iloc[-1] / history['Close'].iloc[-126] - 1) * 100
            if len(history) > 189: data['PM9M'] = (history['Close'].iloc[-1] / history['Close'].iloc[-189] - 1) * 100
            if len(history) > 252: data['PM12M1M'] = (history['Close'].iloc[-22] / history['Close'].iloc[-252] - 1) * 100
            
            dollar_volume_series = history['Close'] * history['Volume']
            data['Amihud'] = _calculate_amihud_illiquidity(dollar_volume_series, daily_returns)
            
            if len(history) >= 260:
                data['PrcTo52WH'] = _safe_division(history['Close'].iloc[-1], history['Close'].tail(260).max())
                data['PrcTo260DL'] = _safe_division(history['Close'].iloc[-1], history['Close'].tail(260).min())

            # C.3. Beta and Relative Strength (Your Original Logic)
            spy_hist = etf_histories.get('SPY')
            if spy_hist is not None and not spy_hist.empty:
                spy_returns = np.log(spy_hist['Close'] / spy_hist['Close'].shift(1)).dropna()
                beta_data = calculate_regime_aware_betas(log_returns, spy_returns)
                if isinstance(beta_data, dict):
                    data['Beta_to_SPY'] = beta_data.get('conservative_beta')
                    data['Beta_Down_Market'] = beta_data.get('down_beta')
                    data['Beta_Up_Market'] = beta_data.get('up_beta')
                    data['Beta60M'] = beta_data.get('conservative_beta') # Approximation
            
            rolling_correlations = {}
            for etf, etf_history in etf_histories.items():
                if etf_history is not None and not etf_history.empty:
                    etf_returns = np.log(etf_history['Close'] / etf_history['Close'].shift(1)).dropna()
                    common_idx = log_returns.index.intersection(etf_returns.index)
                    if len(common_idx) > 90:
                        corr = log_returns.loc[common_idx].corr(etf_returns.loc[common_idx]) if log_returns.loc[common_idx].std() > 1e-6 and etf_returns.loc[common_idx].std() > 1e-6 else 0.0
                        if pd.notna(corr): rolling_correlations[etf] = corr
            
            if rolling_correlations:
                best_factor_ticker = max(rolling_correlations, key=lambda k: abs(rolling_correlations.get(k, 0)))
                data['Best_Factor'] = best_factor_ticker
                data['Correlation_Score'] = rolling_correlations.get(best_factor_ticker)
                best_etf_hist = etf_histories.get(best_factor_ticker)
                if best_etf_hist is not None:
                    common_idx_z = history.index.intersection(best_etf_hist.index)
                    if len(common_idx_z) > 200:
                        if not (best_etf_hist['Close'][common_idx_z] <= 0).any():
                            relative_strength = history['Close'][common_idx_z] / best_etf_hist['Close'][common_idx_z]
                            data['Relative_Z_Score'] = calculate_volatility_adjusted_z_score(relative_strength, ticker=ticker_symbol, sector=data['Sector'])
        
        # --- D. Final Calculations & Placeholders ---
        returns_perf = calculate_returns_cached(ticker_symbol, tuple([5, 10, 21, 63, 126, 252]))
        data.update({f"Return_{p}d": returns_perf.get(f"Return_{p}d") for p in [5, 10, 21, 63, 126, 252]})

        data['Growth'] = data.get('Sales_Growth_YOY')
        data['Q_Score'] = min(pd.to_numeric(data.get('Quick_Ratio', 0), errors='coerce') / 5.0, 1.0) if pd.notna(data.get('Quick_Ratio')) and data.get('Quick_Ratio', 0) > 0 else 0.0
        data['Coverage_Score'] = min(pd.to_numeric(data.get('Interest_Coverage', 0), errors='coerce') / 10.0, 1.0) if pd.notna(data.get('Interest_Coverage')) and data.get('Interest_Coverage', 0) > 0 else 0.0
        
        vision_score = 0
        if data.get('Sector') == 'Technology' and pd.notna(net_income) and net_income < 0: vision_score += 5
        if pd.notna(data.get('Sales_Growth_TTM')) and data.get('Sales_Growth_TTM') > 20: vision_score += 3
        data['Vision'] = min(vision_score / 8.0, 1.0)
        
        pe_val, ps_val = data.get('PE_Ratio'), data.get('PS_Ratio')
        if pd.notna(pe_val) and pd.notna(ps_val) and (pe_val + ps_val) > 0:
            data['Value_Factor'] = min(1 / ((pe_val + ps_val) / 2.0), 1.0)
        else: data['Value_Factor'] = 0.0

        profit_metrics = [data.get('ROE'), data.get('ROIC'), data.get('Net_Profit_Margin')]
        valid_profit_metrics = [m for m in profit_metrics if pd.notna(m)]
        data['Profitability_Factor'] = min(np.mean(valid_profit_metrics) / 100.0, 1.0) if valid_profit_metrics else 0.0
        
        data['Carry'] = (pd.to_numeric(data.get('Earnings_Yield', 0), errors='coerce') + pd.to_numeric(data.get('FCF_Yield', 0), errors='coerce')) / 2.0
        data['Earnings_Price_Ratio'] = data['Earnings_Yield'] / 100 if pd.notna(data.get('Earnings_Yield')) else np.nan
        data['Book_to_Market_Ratio'] = _safe_division(total_equity, market_cap)

        # Placeholders for factors that require post-processing or external (non-yfinance) data
        data['IndRel_TobinQ'] = np.nan # TODO: Implement as post-processing step
        data['ROEStddev20Q'] = np.nan # TODO: Implement using quarterly data history
        data['EQ-style'], data['HG-style'], data['Sz-style'], data['AE-style'], data['PM-style'], data['Vol-style'] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        data['EPSEstDispFY1C'], data['EPSEstDispFY2C'], data['AdjEPSNumRevFY1C'], data['AdjEPSNumRevFY2C'] = np.nan, np.nan, np.nan, np.nan

        # --- E. Finalize & Format Output ---
        result_list = [data.get(col) for col in columns]
        return result_list, log_returns

    except Exception as e:
        logging.error(f"Critical error processing {ticker_symbol}: {e}", exc_info=True)
        failed_data = {col: np.nan for col in columns}
        failed_data['Ticker'] = ticker_symbol
        failed_data['Name'] = f"{ticker_symbol} (Processing Error)"
        return [failed_data.get(col) for col in columns], pd.Series(dtype=float)
@st.cache_data
def process_tickers(_tickers, _etf_histories, _sector_etf_map):
    results, returns_dict, failed_tickers = [], {}, []
    
    # --- CORE FIX: Initialize returns_dict with all tickers ---
    for ticker in _tickers:
        returns_dict[ticker] = pd.Series(dtype=float)

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(process_single_ticker, ticker, _etf_histories, _sector_etf_map): ticker for ticker in _tickers}
        for future in tqdm(as_completed(future_to_ticker), total=len(_tickers), desc="Processing All Ticker Metrics"):
            ticker = future_to_ticker[future]
            try:
                result, returns = future.result()
                
                if result and pd.notna(result[1]): 
                    results.append(result)
                    if returns is not None and not returns.empty:
                        returns_dict[ticker] = returns
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                logging.error(f"Failed to process {ticker} in future: {e}")
                failed_tickers.append(ticker)
            
    if not results:
        return pd.DataFrame(columns=columns), failed_tickers, returns_dict 
    
    results_df = pd.DataFrame(results, columns=columns)
    
    numeric_cols = [c for c in columns if c not in ['Ticker', 'Name', 'Sector', 'Best_Factor', 'Risk_Flag']]
    results_df[numeric_cols] = results_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    for col in results_df.select_dtypes(include=np.number).columns:
        if results_df[col].isna().all():
            results_df[col] = 0.0
        else:
            median_val = results_df[col].median()
            results_df[col] = results_df[col].fillna(median_val)

        if results_df[col].var() < 1e-8:
            results_df[col] += np.random.normal(0, 0.01, len(results_df))
    
    # --- DEFRAGMENTATION FIX ---
    # Create a clean, non-fragmented copy before returning.
    # The .infer_objects() call is a good practice to ensure optimal data types.
    defragmented_df = results_df.infer_objects().copy()
    
    return defragmented_df, failed_tickers, returns_dict
@st.cache_data
def run_factor_stability_analysis(_results_df, _all_possible_metrics, _reverse_metric_map):
    """
    Encapsulates the entire computationally expensive factor stability analysis 
    into a single, cacheable function. This prevents re-running this heavy logic 
    on every Streamlit script refresh, solving the infinite loading loop.
    """
    # --- MODIFICATION START ---
    # These map to the display names in METRIC_NAME_MAP
    time_horizons = {
        "1W": "Return_5d",
        "2W": "Return_10d",
        "1M": "Return_21d",
        "3M": "Return_63d",
        "6M": "Return_126d",
        "12M": "Return_252d",
    }
    # --- MODIFICATION END ---
    
    # Filter out metrics that are not numeric or are "Score" related
    valid_metric_cols = [c for c in _results_df.columns if pd.api.types.is_numeric_dtype(_results_df[c]) and 'Return' not in c and c not in ['Ticker', 'Name', 'Score']]
    stability_results = {}

    logging.info("Starting cached factor stability analysis...")
    for horizon_label, target_column_short_name in time_horizons.items(): # Renamed target_column
        if target_column_short_name in _results_df.columns:
            logging.info(f"Calculating pure returns for {horizon_label} horizon (Target: {target_column_short_name})...") # Added target name to log
            
            # Use the short name for the target column in calculate_pure_returns
            pure_returns_today = calculate_pure_returns(_results_df, valid_metric_cols, target=target_column_short_name)
            
            if not pure_returns_today.empty:
                # Simulate a list of Series, each representing pure factor returns for a date
                historical_pure_returns = simulate_historical_pure_returns(pure_returns_today)
                stability_df = analyze_coefficient_stability(historical_pure_returns)
                stability_results[horizon_label] = stability_df
            else:
                logging.warning(f"Pure returns calculation failed for {horizon_label}, skipping stability analysis.")
    
    logging.info("Aggregating stability results...")
    auto_weights, rationale_df = aggregate_stability_and_set_weights(
        stability_results, _all_possible_metrics, _reverse_metric_map
    )
    
    return auto_weights, rationale_df, stability_results

# The user's provided `robust_vol_calc` is more sophisticated and could be swapped in.
@st.cache_data
def get_all_prices_and_vols(_tickers, _returns_dict):
    if not _returns_dict: # Added robustness check for empty returns_dict
        return pd.DataFrame() # Return empty if no returns

    all_prices_df = pd.DataFrame({t: np.exp(rets.cumsum()) for t, rets in _returns_dict.items() if not rets.empty})
    
    if all_prices_df.empty: # Added robustness check for empty prices_df
        return pd.DataFrame()
        
    # --- MODIFICATION START ---
    # Calculate EWMA volatility for each ticker
    all_vols_df = pd.DataFrame()
    for ticker in all_prices_df.columns:
        # Calculate daily percentage returns (not log returns here for vol calc)
        daily_pct_returns = all_prices_df[ticker].pct_change(fill_method=None).dropna() # FIX: fill_method=None
        all_vols_df[ticker] = calculate_ewma_volatility(daily_pct_returns, span=63, annualize=False) # Daily vol
    
    # Normalize by dividing price by its daily (non-annualized) volatility
    normalized_prices = (all_prices_df / all_vols_df).replace([np.inf, -np.inf], np.nan).dropna() # Added replace inf/nan
    # --- MODIFICATION END ---
    
    return normalized_prices

def calculate_cs_mean_reversion(results_df, normalized_prices, horizon=126, ewma_span=32):
    """
    Calculates the cross-sectional mean reversion signal.
    Dx_t = [Nx_t - Nx_tau] - [NA_t - NA_tau]
    Forecast = -EWMA(Dx_t, span)
    This version is corrected for robustness against insufficient data.
    """
    # --- ROBUSTNESS CHECK 1: Ensure there is enough data for the calculation ---
    if normalized_prices.empty or len(normalized_prices) < horizon + 5: # Add a small buffer
        logging.warning(f"Insufficient data for CS Mean Reversion (need >{horizon} days, have {len(normalized_prices)}). Setting signal to 0.")
        # Directly create the column and fill it with zeros
        results_df['CS_Mean_Reversion'] = 0.0
        return results_df

    # Align normalized_prices columns to tickers in results_df
    valid_tickers_in_prices = [t for t in results_df['Ticker'].unique() if t in normalized_prices.columns]
    if not valid_tickers_in_prices:
        logging.warning("No common tickers between results_df and normalized_prices. Setting CS Mean Reversion to 0.")
        results_df['CS_Mean_Reversion'] = 0.0
        return results_df

    normalized_prices_aligned = normalized_prices[valid_tickers_in_prices]

    # Get the sector for each ticker, ensuring alignment
    sector_map = results_df.set_index('Ticker')['Sector'].reindex(valid_tickers_in_prices)
    
    # Calculate the price change over the horizon for each stock
    # Ensure diff operation considers sufficient previous data
    instrument_change = normalized_prices_aligned.diff(horizon)
    
    # Calculate the average price change for each sector (asset class)
    # Use .groupby then .transform to map back to original DataFrame shape
    asset_class_change = instrument_change.groupby(sector_map, axis=1).transform('mean')

    # Calculate the outperformance (Disequilibrium)
    outperformance = instrument_change - asset_class_change
    
    # The final forecast is the negative EWMA of this outperformance
    forecast = -outperformance.ewm(span=ewma_span, min_periods=max(1, ewma_span//2)).mean()

    # --- ROBUSTNESS CHECK 2: Ensure the forecast calculation was successful ---
    if forecast.empty or forecast.iloc[-1].isnull().all():
        logging.warning("CS Mean Reversion calculation resulted in no valid signals. Setting to 0.")
        results_df['CS_Mean_Reversion'] = 0.0
        return results_df

    # Get the latest forecast for each ticker
    latest_forecast = forecast.iloc[-1].rename('CS_Mean_Reversion_Val') # Renamed to avoid direct conflict
    
    # --- THE DEFINITIVE FIX: Use .map() to safely create the column ---
    # This guarantees the 'CS_Mean_Reversion' column is created.
    # Tickers not in `latest_forecast` will get NaN.
    results_df['CS_Mean_Reversion'] = results_df['Ticker'].map(latest_forecast)
    
    # Now that the column is guaranteed to exist, this .fillna() is safe.
    results_df['CS_Mean_Reversion'] = results_df['CS_Mean_Reversion'].fillna(0.0)
    
    return results_df
def calculate_ideal_weights(factor_rankings, etf_histories, df, pure_returns=None, rank_threshold=1000, max_weight=15.0, pure_return_weight=0.8):
    """
    Calculates dynamic 'ideal' weights for the metric sliders.
    """
    all_metrics = list(default_weights.keys())
    long_to_short_map = REVERSE_METRIC_NAME_MAP

    if factor_rankings.empty or 'Avg_Rank' not in factor_rankings.columns:
        factor_rankings = pd.DataFrame(index=[long_to_short_map.get(m, m) for m in all_metrics],
                                      columns=['Avg_Rank']).fillna(1.0)

    valid_factors = factor_rankings[factor_rankings['Avg_Rank'] < rank_threshold].copy()
    if valid_factors.empty:
        valid_factors = factor_rankings.copy()
        valid_factors['Avg_Rank'] = valid_factors['Avg_Rank'].fillna(1.0) + np.random.uniform(0.1, 0.5, len(valid_factors))

    valid_factors['Inverse_Corr_Rank'] = 1 / valid_factors['Avg_Rank'].replace(0, 1e-6)
    total_inverse_corr_rank = valid_factors['Inverse_Corr_Rank'].sum() or 1e-6

    pure_return_ranks = {}
    if pure_returns is not None and not pure_returns.empty and pure_returns.abs().sum() > 0:
        pure_return_ranks_series = pure_returns.abs().rank(ascending=False, na_option='bottom')
        pure_return_ranks = {metric: 1 / rank if not pd.isna(rank) and rank > 0 else 1e-6
                            for metric, rank in pure_return_ranks_series.items()}
        total_inverse_pure_rank = sum(pure_return_ranks.values()) or 1e-6
    else:
        total_inverse_pure_rank = len(all_metrics)
        pure_return_ranks = {long_to_short_map.get(m, m): 1.0 + np.random.uniform(0.1, 0.5) for m in all_metrics}

    weights = {}
    for long_name in all_metrics:
        short_name = long_to_short_map.get(long_name, long_name)
        corr_rank_weight = (valid_factors.loc[short_name, 'Inverse_Corr_Rank'] / total_inverse_corr_rank * 100) if short_name in valid_factors.index else 1.0
        pure_rank_weight = (pure_return_ranks.get(short_name, 1.0) / total_inverse_pure_rank * 100) if total_inverse_pure_rank > 0 else 1.0
        weights[long_name] = (1 - pure_return_weight) * corr_rank_weight + pure_return_weight * pure_rank_weight

    excess_weight = 0.0
    for metric in weights:
        if weights[metric] > max_weight:
            excess_weight += weights[metric] - max_weight
            weights[metric] = max_weight

    non_capped_sum = sum(w for w in weights.values() if w < max_weight and w > 0)
    if non_capped_sum > 0:
        for metric in weights:
            if weights[metric] < max_weight and weights[metric] > 0:
                weights[metric] += (weights[metric] / non_capped_sum) * excess_weight

    total_weight = sum(weights.values())
    if total_weight > 0:
        norm_factor = 100.0 / total_weight
        for metric in weights:
            weights[metric] *= norm_factor
    else:
        weights = {metric: 100.0 / len(all_metrics) for metric in all_metrics}

    cluster_fig = None 
    etf_clusters = {etf: [etf] for etf in etf_histories.keys()}
    return weights, cluster_fig, etf_clusters

def analyze_factor_correlations(df, returns_dict):
    """
    Analyzes factor correlations with robust data cleaning.
    """
    if df.empty:
        logging.error("analyze_factor_correlations received empty DataFrame")
        st.error("No valid tickers for correlation analysis")
        return df, pd.DataFrame(), pd.DataFrame(columns=['Avg_Rank']).fillna(1.0) # Return empty df for factor_rankings

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return df, pd.DataFrame(), pd.DataFrame(columns=['Avg_Rank']).fillna(1.0)
    
    valid_data = df[numeric_cols].copy()
    
    # FIX: Replaced inplace=True with direct reassignment
    valid_data = valid_data.replace([np.inf, -np.inf], np.nan)
    valid_data = valid_data.dropna(axis=1, how='all') # Drop columns that are all NaN
    
    if valid_data.empty: # Check if empty after dropping all-NaN columns
        logging.warning("No valid data for correlation after NaN/inf cleaning. Returning empty results.")
        return df, pd.DataFrame(), pd.DataFrame(columns=['Avg_Rank']).fillna(1.0)


    for col in valid_data.columns:
        median_val = valid_data[col].median()
        # FIX: Replaced inplace=True with direct reassignment
        valid_data[col] = valid_data[col].fillna(median_val)
        
        if valid_data[col].var() < 1e-6: # Add small noise to constant columns
            valid_data[col] = valid_data[col] + np.random.normal(0, 0.01, len(valid_data))

    if valid_data.empty or len(valid_data) < 2:
        logging.warning("Insufficient valid data for correlation. Returning empty results.")
        return df, pd.DataFrame(), pd.DataFrame(columns=['Avg_Rank']).fillna(1.0) # Ensure empty factor_rankings has a 'Metrics' column if needed later

    try:
        # Ensure that valid_data has at least 2 columns before calculating correlation matrix
        if valid_data.shape[1] < 2:
            logging.warning("Only one valid numeric column left for correlation. Cannot compute matrix.")
            # Changed the index to reflect the single remaining column
            return df, pd.DataFrame(), pd.DataFrame(index=[valid_data.columns[0]], columns=['Avg_Rank']).fillna(1.0) 


        corr_matrix = np.corrcoef(valid_data.T, rowvar=True)
        corr_matrix = np.where(np.isfinite(corr_matrix), corr_matrix, 0.0)
        
        valid_metrics = valid_data.columns.tolist()
        correlation_results = pd.DataFrame(corr_matrix, index=valid_metrics, columns=valid_metrics)

        abs_corrs = correlation_results.abs()
        ranks_dict = {col: abs_corrs[col].rank(ascending=False, na_option='bottom') for col in valid_metrics}
        ranks = pd.concat(ranks_dict, axis=1)
        ranks['Avg_Rank'] = ranks.mean(axis=1, skipna=True)
        factor_rankings = ranks[['Avg_Rank']].copy()
        factor_rankings['Metrics'] = factor_rankings.index # Add 'Metrics' column for consistency if needed by other functions

        return df, correlation_results, factor_rankings

    except Exception as e:
        logging.error(f"Error in analyze_factor_correlations after cleaning: {str(e)}")
        return df, pd.DataFrame(), pd.DataFrame(columns=['Avg_Rank']).fillna(1.0)
def calculate_correlation_matrix(tickers, returns_dict, window=90):
    """
    Calculates a robust, positive semi-definite correlation and covariance matrix.
    """
    n = len(tickers)
    if n == 0 or not returns_dict:
        # Return empty DFs with correct index/columns if no tickers
        return pd.DataFrame(index=tickers, columns=tickers), pd.DataFrame(index=tickers, columns=tickers)

    # Filter returns_dict to only include provided tickers
    filtered_returns_dict = {t: returns_dict[t] for t in tickers if t in returns_dict and not returns_dict[t].empty}
    
    if not filtered_returns_dict: # If no valid returns for any ticker
        identity = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        return identity, identity

    returns_df = pd.DataFrame(filtered_returns_dict)
    
    # Reindex to ensure all tickers are present, filling missing with 0 for matrix calculation
    returns_df = returns_df.reindex(columns=tickers).fillna(0.0)

    aligned_returns = returns_df.tail(window)
    
    # FIX: Replaced inplace=True with direct reassignment
    aligned_returns = aligned_returns.dropna(axis=1, how='all') # Drop columns that are all NaN in the window
    # aligned_returns = aligned_returns.fillna(0.0) # Already handled by reindex and .fillna(0.0) above

    valid_tickers = aligned_returns.columns.tolist()
    if len(valid_tickers) < 2: # Need at least 2 for covariance
        identity = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        return identity, identity

    try:
        lw = LedoitWolf()
        lw.fit(aligned_returns)
        
        cov_matrix_values = lw.covariance_ * 252 # Annualize
        
        # Create full covariance matrix (including tickers that were dropped due to NaNs)
        cov_matrix_full = pd.DataFrame(np.eye(n) * np.mean(np.diag(cov_matrix_values)), index=tickers, columns=tickers)
        cov_matrix_full.loc[valid_tickers, valid_tickers] = cov_matrix_values
        
        vols = np.sqrt(np.diag(cov_matrix_values))
        vols[vols < 1e-8] = 1.0 # Prevent division by near-zero vol
        
        # Ensure outer product is correctly shaped for correlation
        corr_matrix_values = cov_matrix_values / np.outer(vols, vols)
        
        corr_matrix_full = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        corr_matrix_full.loc[valid_tickers, valid_tickers] = corr_matrix_values
        
        final_corr = pd.DataFrame(nearest_psd_matrix(corr_matrix_full.values), index=tickers, columns=tickers)

    except Exception as e:
        logging.error(f"Ledoit-Wolf estimation failed: {e}. Falling back to identity matrix for correlation and cov.")
        identity = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        return identity, identity

    return final_corr, cov_matrix_full

def calculate_weights(returns_df, method="equal", cov_matrix=None, factor_returns=None, betas=None):
    """
    Calculate portfolio weights with various methods.
    """
    n_assets = len(returns_df.columns)
    tickers = returns_df.columns
    
    if n_assets == 0: # Added robustness for empty returns_df
        return pd.Series(dtype=float)

    try:
        if method == "equal":
            return pd.Series(np.ones(n_assets) / n_assets, index=tickers)
        
        elif method == "inv_vol":
            vols = returns_df.std() * np.sqrt(252)
            # Handle zero or NaN volatilities
            inv_vols = 1 / vols.replace([0, np.nan], 1e-6) # Replace 0/NaN with small positive to avoid inf/nan
            if inv_vols.sum() == 0: # If all became 1e-6, ensure uniform distribution
                 return pd.Series(np.ones(n_assets) / n_assets, index=tickers)
            return inv_vols / inv_vols.sum()

        elif method == "log_log_sharpe":
            mu = returns_df.mean() * 252 # Annualized mean for objective
            if cov_matrix is None:
                cov_matrix = returns_df.cov() * 252
            cov_matrix = nearest_psd_matrix(cov_matrix)
            
            # Ensure cov_matrix is not singular for optimization
            if np.linalg.cond(cov_matrix) > 1e10: # Added singularity check
                logging.warning("Log Log Sharpe: Covariance matrix is ill-conditioned. Falling back to equal weights.")
                return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

            w = cp.Variable(n_assets)
            
            gamma = cp.Parameter(nonneg=True, value=0.5) # Risk aversion parameter
            max_weight = 0.15 # Max individual stock weight
            
            # Objective: Maximize expected return minus risk penalty
            objective = cp.Maximize(mu.values @ w - gamma * cp.quad_form(w, cov_matrix))
            constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS)
            
            # Check for successful solution
            if w.value is not None and w.value.sum() > 1e-6:
                 final_weights = pd.Series(w.value, index=tickers)
                 # Re-normalize just in case numerical issues led to sum slightly off 1
                 return final_weights / final_weights.sum()
            else:
                 logging.warning("Log Log Sharpe optimization failed or returned zero weights. Falling back to equal weights.")
                 return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

        elif method == "fmp":
            if factor_returns is None or cov_matrix is None:
                raise ValueError("Factor returns and covariance matrix required for FMP.")
            weights = calculate_fmp_weights(returns_df, factor_returns, cov_matrix)
            return pd.Series(weights, index=tickers)

        elif method == "alpha_orthogonal":
            if betas is None or returns_df is None:
                raise ValueError("Betas and returns required for alpha_orthogonal.")
            
            alpha = returns_df.mean() * 252 # Annualized alpha for objective
            w = cp.Variable(n_assets)
            
            # Ensure betas DataFrame is aligned and numeric
            aligned_betas = betas.reindex(tickers).fillna(0.0)
            
            # Filter beta columns for non-zero variance before using in constraint
            valid_beta_cols = aligned_betas.loc[:, aligned_betas.std() > 1e-6].columns # Added filtering
            if valid_beta_cols.empty: # Added check for empty valid betas
                logging.warning("Alpha Orthogonal: All betas are zero or constant. Cannot enforce orthogonality. Falling back to equal weights.")
                return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

            objective = cp.Maximize(alpha.values @ w)
            # Ensure the constraint uses only valid beta columns
            constraints = [cp.sum(w) == 1, w >= 0, aligned_betas[valid_beta_cols].values.T @ w == 0] # Used valid_beta_cols
            
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS)
            
            if w.value is not None and w.value.sum() > 1e-6:
                final_weights = pd.Series(w.value, index=tickers)
                return final_weights / final_weights.sum()
            else:
                logging.warning("Alpha Orthogonal optimization failed or returned zero weights. Falling back to equal weights.")
                return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

        else:
            raise ValueError(f"Unknown weighting method: {method}")
    except Exception as e:
        logging.error(f"Error in weight calculation for method '{method}': {e}")
        return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

def display_ma_deviation(history):
    st.subheader("Price Deviation from Moving Averages")

    if history.empty or len(history) < 200: # Added history.empty check
        st.warning("Not enough data for Moving Average analysis (requires 200 days).")
        return

    try:
        price = history['Close'].iloc[-1]
        
        # Ensure enough data points for rolling windows
        if len(history) < 20: ma20 = np.nan # Added conditional checks
        else: ma20 = history['Close'].rolling(window=20).mean().iloc[-1]
        
        if len(history) < 50: ma50 = np.nan # Added conditional checks
        else: ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        
        if len(history) < 200: ma200 = np.nan # Added conditional checks
        else: ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        
        if len(history) < 20: std20 = np.nan # Added conditional checks
        else: std20 = history['Close'].rolling(window=20).std().iloc[-1]

        if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma200) or pd.isna(std20) or std20 == 0: # Added std20 == 0 check
             st.info("Could not calculate all moving average components or standard deviation is zero (constant price).") # Changed to info
             return

        upper_band = ma20 + 2 * std20
        lower_band = ma20 - 2 * std20

        if price > upper_band:
            status = "Overbought"
            status_color = "red"
        elif price < lower_band:
            status = "Oversold"
            status_color = "green"
        else:
            status = "Neutral"
            status_color = "darkgray"

        # Dynamically adjust range for gauge for better visualization
        min_range = min(ma200, lower_band, price) * 0.9 # Dynamic range adjustment
        max_range = max(ma200, upper_band, price) * 1.1 # Dynamic range adjustment

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=price,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{status}</b>", 'font': {'size': 20}},
            number={'prefix': "$", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [min_range, max_range], 'tickwidth': 1, 'tickcolor': "darkblue"}, # Used dynamic range
                'bar': {'color': status_color, 'thickness': 0.3}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                'steps': [{'range': [min_range, ma50], 'color': 'rgba(0, 255, 0, 0.2)'}, {'range': [ma50, max_range], 'color': 'rgba(255, 0, 0, 0.2)'}], # Used dynamic range
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.9, 'value': price}
            }
        ))
        # Add MA values as annotations
        fig.add_annotation( # Added annotations instead of trace
            x=0.05, y=0.05, text=f"MA20: ${ma20:.2f}", showarrow=False, font=dict(size=10, color="white"), align="left", xref="paper", yref="paper"
        )
        fig.add_annotation(
            x=0.35, y=0.05, text=f"MA50: ${ma50:.2f}", showarrow=False, font=dict(size=10, color="white"), align="left", xref="paper", yref="paper"
        )
        fig.add_annotation(
            x=0.65, y=0.05, text=f"MA200: ${ma200:.2f}", showarrow=False, font=dict(size=10, color="white"), align="left", xref="paper", yref="paper"
        )

        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying MA deviation chart: {e}")

def get_regression_metrics(history, period=126):
    if history.empty or len(history) < period: return np.nan, "N/A" # Added history.empty check
    
    # Ensure prices are positive for log transformation
    if (history['Close'].iloc[-period:] <= 0).any(): # Added non-positive price check
        return np.nan, "N/A"

    y = np.log(history['Close'].iloc[-period:])
    x = np.arange(len(y))
    
    # Check for constant y values to avoid linregress errors
    if y.std() < 1e-6: # Added variance check
        return 0.0, "NEUTRAL"
        
    slope, intercept, _, p_value, _ = linregress(x, y)
    
    # Check for valid linregress output
    if not np.isfinite(slope) or not np.isfinite(p_value): # Added finite check
        return np.nan, "N/A"

    predicted_y = slope * x + intercept
    residuals = y - predicted_y
    std_dev_from_reg = np.std(residuals)
    
    trend_str = "NEUTRAL"
    if p_value < 0.05: # Statistical significance
        # Economic significance threshold (e.g. slope must be visibly non-zero)
        if slope > 0.0005: trend_str = "Positive" 
        elif slope < -0.0005: trend_str = "Negative"
    
    return std_dev_from_reg, trend_str

def get_daily_risk_range(history):
    if history.empty or len(history) < 15: return np.nan, np.nan, np.nan, np.nan
    
    last_close = history['Close'].iloc[-1]
    prev_close = history['Close'].iloc[-2] if len(history) >= 2 else last_close # Added check for prev_close
    
    high, low, close = history['High'], history['Low'], history['Close']
    
    # Ensure all series have enough data for calculation
    if len(high) < 2 or len(low) < 2 or len(close) < 2: # Added check for length of series
        return np.nan, np.nan, np.nan, np.nan

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr_df = pd.concat([tr1, tr2, tr3], axis=1)
    tr = tr_df.max(axis=1).dropna() # Drop NaNs before EWMA

    if tr.empty: # Added check for empty tr
        return np.nan, np.nan, np.nan, np.nan

    atr = tr.ewm(span=14, adjust=False, min_periods=max(1, 14//2)).mean().iloc[-1] # Added min_periods for robustness
    
    if pd.isna(atr): # Added check for NaN atr
        return np.nan, np.nan, np.nan, np.nan

    risk_low, risk_high = last_close - atr, last_close + atr
    pct_change = (last_close - prev_close) / prev_close * 100 if prev_close != 0 else np.nan # Handled prev_close == 0
    return risk_low, risk_high, last_close, pct_change

def display_momentum_bar(ticker_symbol, history):
    st.subheader("Dual-Scale Momentum (14-Day | 14-Hour)")
    
    # --- 14-Day RSI ---
    rsi_14d = 50.0 # Default if not enough data
    if len(history) >= 15:
        delta_d = history['Close'].diff()
        gain_d = (delta_d.where(delta_d > 0, 0)).rolling(window=14, min_periods=1).mean() # Added min_periods=1
        loss_d = (-delta_d.where(delta_d < 0, 0)).rolling(window=14, min_periods=1).mean() # Added min_periods=1
        
        if not pd.isna(loss_d.iloc[-1]) and loss_d.iloc[-1] != 0:
            rs_d = gain_d.iloc[-1] / loss_d.iloc[-1]
            rsi_14d = 100 - (100 / (1 + rs_d))
        elif not pd.isna(loss_d.iloc[-1]) and loss_d.iloc[-1] == 0 and gain_d.iloc[-1] > 0:
            rsi_14d = 100 # Pure gains, RSI is 100
        elif not pd.isna(gain_d.iloc[-1]) and gain_d.iloc[-1] == 0 and loss_d.iloc[-1] > 0: # Added pure losses case
            rsi_14d = 0 # Pure losses, RSI is 0

    # --- 14-Hour RSI ---
    rsi_14h, has_hourly = 50.0, False
    try:
        hourly_hist = yf.Ticker(ticker_symbol).history(period="60d", interval="1h", auto_adjust=True).tz_localize(None) # Added tz_localize(None)
        if not hourly_hist.empty and len(hourly_hist) >= 15:
            delta_h = hourly_hist['Close'].diff()
            gain_h = (delta_h.where(delta_h > 0, 0)).rolling(window=14, min_periods=1).mean() # Added min_periods=1
            loss_h = (-delta_h.where(delta_h < 0, 0)).rolling(window=14, min_periods=1).mean() # Added min_periods=1
            if not pd.isna(loss_h.iloc[-1]) and loss_h.iloc[-1] != 0:
                rs_h = gain_h.iloc[-1] / loss_h.iloc[-1]
                rsi_14h = 100 - (100 / (1 + rs_h))
            elif not pd.isna(loss_h.iloc[-1]) and loss_h.iloc[-1] == 0 and gain_h.iloc[-1] > 0:
                rsi_14h = 100
            elif not pd.isna(gain_h.iloc[-1]) and gain_h.iloc[-1] == 0 and loss_h.iloc[-1] > 0: # Added pure losses case
                rsi_14h = 0
            has_hourly = True
    except Exception as e:
        logging.warning(f"Could not fetch hourly data for {ticker_symbol}: {e}")

    col1, col2 = st.columns(2)
    col1.metric("14-Day Trend (The Bar)", f"{rsi_14d:.1f}", help="RSI > 50 is bullish.")
    if has_hourly:
        col2.metric("14-Hour Pressure (The Marker)", f"{rsi_14h:.1f}", help="RSI > 50 indicates short-term upward pressure.")
    else:
        col2.info("Hourly data not available.")

    bar_color = "#04AA6D" if rsi_14d > 50 else "#FA3F46"
    fig = go.Figure()
    fig.add_trace(go.Bar(y=['RSI'], x=[rsi_14d], orientation='h', marker_color=bar_color, marker_line_width=0, width=0.5, hoverinfo='none'))
    fig.add_shape(type="line", x0=50, y0=-0.5, x1=50, y1=0.5, line=dict(color="rgba(255, 255, 255, 0.3)", width=1))
    fig.add_shape(type="line", x0=35, y0=-0.5, x1=35, y1=0.5, line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dash"))
    fig.add_shape(type="line", x0=65, y0=-0.5, x1=65, y1=0.5, line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dash"))
    if has_hourly:
        fig.add_shape(type="line", x0=rsi_14h, y0=-0.5, x1=rsi_14h, y1=0.5, line=dict(color="white", width=3))
    fig.update_layout(xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False), yaxis=dict(showticklabels=False, showgrid=False, zeroline=False), showlegend=False, plot_bgcolor='rgba(68, 68, 68, 0.5)', paper_bgcolor='rgba(0,0,0,0)', height=40, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bar shows 14-day trend; white marker shows 14-hour pressure; dashed lines at RSI 35 and 65.")

def calculate_absolute_z_score_and_trend(daily_history):
    # This function is retained without modification as it's separate from `calculate_volatility_adjusted_z_score`
    # and its internal slicing (`.iloc[-252:]`, `.iloc[-126:]`) is specific to its definition of
    # "absolute" Z-score and trend, not to be confused with the relative Z-score's calculation logic.
    if daily_history.empty or len(daily_history) < 252: return np.nan, "NEUTRAL"
    
    # Ensure prices are positive for log transform
    if (daily_history['Close'].iloc[-252:] <= 0).any():
        return np.nan, "NEUTRAL"

    prices_log = np.log(daily_history['Close'].iloc[-252:])
    x = np.arange(len(prices_log))
    
    # Check for constant prices
    if prices_log.std() < 1e-6: # Added check for constant prices_log
        return 0.0, "NEUTRAL"

    slope, _, r_value, p_value, _ = linregress(x, prices_log)
    
    # Check for valid linregress output
    if not np.isfinite(slope) or not np.isfinite(p_value) or not np.isfinite(r_value): # Added check for finite output
        return np.nan, "NEUTRAL"

    trend_status = "NEUTRAL"
    if p_value < 0.05 and abs(r_value) > 0.4: # Statistical and economic significance for trend
        if slope > 0.0005: trend_status = "UP-TREND" # Threshold for "up"
        elif slope < -0.0005: trend_status = "DOWN-TREND" # Threshold for "down"
    
    # Absolute Z-score calculation (last 126 days)
    prices = daily_history['Close'].iloc[-126:]
    if len(prices) < 126: return np.nan, trend_status # Not enough for this part
    
    # Check for constant prices in this window
    if prices.std() < 1e-6: # Added check for constant prices
        return 0.0, trend_status # If constant, Z-score is 0

    m_trend = prices.rolling(window=126, min_periods=63).median().iloc[-1]
    mad = np.median(np.abs(prices - m_trend))
    
    if pd.isna(m_trend) or pd.isna(mad) or mad == 0: return np.nan, trend_status
    
    absolute_z_score = (prices.iloc[-1] - m_trend) / (mad / 0.6745)
    
    return absolute_z_score, trend_status

def display_signal_sigma_checklist(stock_data, daily_history):
    st.subheader("Signal Sigma Checklist")
    def display_checklist_item(label, is_passed, help_text=""):
        col1, col2 = st.columns([0.1, 0.9])
        with col1: st.markdown("✅" if is_passed else "❌")
        with col2: st.markdown(f"**{label}**", help=help_text)

    # Ensure daily_history is passed correctly for calculation
    absolute_z_score, trend_status = calculate_absolute_z_score_and_trend(daily_history)
    relative_z_score = stock_data.get('Relative_Z_Score')
    best_factor = stock_data.get('Best_Factor', 'its benchmark') # Default for display
    
    # Ensure returns are numeric and not NaN before comparison
    return_6m = pd.to_numeric(stock_data.get('Return_126d'), errors='coerce') # Added explicit numeric conversion
    return_1y = pd.to_numeric(stock_data.get('Return_252d'), errors='coerce') # Added explicit numeric conversion
    f_score = pd.to_numeric(stock_data.get('Piotroski_F-Score'), errors='coerce') # Added explicit numeric conversion
    op_leverage = pd.to_numeric(stock_data.get('Operating_Leverage'), errors='coerce') # Added explicit numeric conversion
    rd_ratio = pd.to_numeric(stock_data.get('RD_to_Gross_Profit_2Y_Avg', 0), errors='coerce') # Added explicit numeric conversion

    st.markdown(f"**Current Trend Status: `{trend_status}`**")
    st.markdown("---")
    st.markdown("#### Primary Rule (Internal Trend Timing)")
    if trend_status == "UP-TREND":
        is_above_lower_channel = pd.notna(absolute_z_score) and absolute_z_score >= -1.0
        display_checklist_item("Price is Above Lower Channel (Absolute Z > -1)", is_above_lower_channel, f"Current Absolute Z-Score: {absolute_z_score:.2f}" if pd.notna(absolute_z_score) else "N/A") # Added N/A help
    elif trend_status == "DOWN-TREND":
        is_above_m_trend = pd.notna(absolute_z_score) and absolute_z_score >= 0.0
        display_checklist_item("Price is Recovering Above M-Trend (Absolute Z > 0)", is_above_m_trend, f"Current Absolute Z-Score: {absolute_z_score:.2f}" if pd.notna(absolute_z_score) else "N/A") # Added N/A help
    else: st.info("The stock is in a NEUTRAL trend. The primary timing rules do not apply.")
    st.markdown("---")
    st.markdown("#### Universal Screening Rules (Strength & Quality)")
    is_outperforming = pd.notna(relative_z_score) and relative_z_score > 0
    display_checklist_item("Outperforming Benchmark (Relative Z > 0)", is_outperforming, f"Relative Z-Score vs {best_factor}: {relative_z_score:.2f}" if pd.notna(relative_z_score) else "N/A") # Added N/A help
    positive_6m_return = pd.notna(return_6m) and return_6m > 0
    display_checklist_item("6-Month Return is Positive", positive_6m_return, f"Current 6M Return: {return_6m:.1f}%" if pd.notna(return_6m) else "N/A") # Added N/A help
    positive_1y_return = pd.notna(return_1y) and return_1y > 0
    display_checklist_item("1-Year Return is Positive", positive_1y_return, f"Current 1Y Return: {return_1y:.1f}%" if pd.notna(return_1y) else "N/A") # Added N/A help
    strong_f_score = pd.notna(f_score) and f_score >= 5
    display_checklist_item("Piotroski F-Score >= 5", strong_f_score, f"Current: {f_score:.0f}" if pd.notna(f_score) else "N/A") # Added N/A help
    good_op_leverage = pd.notna(op_leverage) and op_leverage >= 1
    display_checklist_item("Operating Leverage >= 1", good_op_leverage, f"Current: {op_leverage:.2f}" if pd.notna(op_leverage) else "N/A") # Added N/A help
    high_rd_investment = pd.notna(rd_ratio) and rd_ratio > 0.25
    display_checklist_item("R&D / Gross Profit > 25%", high_rd_investment, f"Current: {rd_ratio*100:.1f}%" if pd.notna(rd_ratio) else "N/A") # Added N/A help
def valuation_wizard(ticker_symbol, revenue_growth_rate, gross_margin_rate, op_ex_as_percent_of_sales, share_count_growth_rate, ev_to_ebitda_multiple, tax_rate):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        last_revenue = get_value(financials, ['Total Revenue', 'TotalRevenue']) # Used get_value for robustness
        last_ebitda = info.get('ebitda') or (get_value(financials, ['EBIT']) + get_value(cashflow, ['Depreciation And Amortization', 'Depreciation'])) # Used get_value
        last_interest_expense = get_value(financials, ['Interest Expense', 'InterestExpense']) # Used get_value
        
        current_shares = info.get('sharesOutstanding')
        current_eps = info.get('trailingEps')
        
        total_debt = get_value(balance_sheet, ['Total Debt', 'TotalDebt']) or info.get('totalDebt', 0) # Used get_value
        cash = get_value(balance_sheet, ['Cash And Cash Equivalents', 'CashAndCashEquivalents']) or info.get('totalCash', 0) # Used get_value
        
        net_debt = total_debt - cash
        
        # Robustness for essential base data
        if not all(pd.notna(v) and v > 0 for v in [last_revenue, current_shares, last_ebitda]): # Check for notna and >0
            return np.nan, np.nan, "Error: Could not retrieve essential base data (revenue, shares, EBITDA)."
        if pd.isna(current_eps): return np.nan, np.nan, "Error: Could not retrieve Current EPS."
        
        debt_to_ebitda = net_debt / last_ebitda if last_ebitda > 0 else float('inf')
        if debt_to_ebitda > 10.0: # High debt to EBITDA as unreliability flag
            return np.nan, np.nan, f"Valuation Unreliable: Debt to EBITDA ratio ({debt_to_ebitda:.1f}x) is excessively high (>10x)."
        
        y5_revenue = last_revenue * ((1 + revenue_growth_rate) ** 5)
        y5_ebitda = last_ebitda * ((1 + revenue_growth_rate) ** 5)
        
        # Ensure depreciation value is safely retrieved and check for 0 for growth
        depreciation_base = info.get('depreciation', 0)
        y5_depreciation = depreciation_base * ((1 + revenue_growth_rate)**5) if depreciation_base != 0 else 0 # Avoid 0 * growth
        y5_ebit = y5_ebitda - y5_depreciation # Improved y5_ebit calculation
        
        y5_enterprise_value = y5_ebitda * ev_to_ebitda_multiple
        
        # Ensure interest expense is handled for growth and net debt for cash growth
        y5_interest_expense = (last_interest_expense * ((1 + revenue_growth_rate / 4)**5)) if last_interest_expense != 0 else 0
        y5_cash = cash * ((1 + revenue_growth_rate / 4)**5) # Assuming cash grows with a fraction of revenue growth
        y5_total_debt = total_debt # Assuming debt is not paid down in the model
        y5_net_debt = y5_total_debt - y5_cash # Recalculated net debt
        
        y5_equity_value = y5_enterprise_value - y5_net_debt
        if y5_equity_value <= 0:
            return np.nan, np.nan, "Valuation Failed: Model predicts equity value will be zero or negative (debt exceeds enterprise value)."
        
        y5_shares = current_shares * ((1 + share_count_growth_rate) ** 5)
        price_target = y5_equity_value / y5_shares if y5_shares > 0 else np.nan
        
        if price_target is np.nan:
            return np.nan, np.nan, "Valuation Failed: Share count is zero or invalid for year 5."
            
        if current_eps <= 0:
            return price_target, np.nan, "Current EPS is non-positive; Implied 5Y EPS CAGR cannot be calculated."
        else:
            y5_ebt = y5_ebit - y5_interest_expense # Used y5_interest_expense
            y5_net_income = y5_ebt * (1 - tax_rate)
            y5_eps = y5_net_income / y5_shares if y5_shares > 0 else np.nan
            if pd.notna(y5_eps) and y5_eps > 0 and current_eps > 0: # Ensure both are positive
                 eps_cagr = (y5_eps / current_eps)**(1/5) - 1
                 return price_target, eps_cagr, "Calculation successful."
            else:
                 return price_target, np.nan, "Model predicts negative or invalid EPS in Year 5."
    except Exception as e:
        return np.nan, np.nan, f"An error occurred during valuation: {e}" # More specific error message

def display_valuation_wizard(ticker_symbol):
    st.subheader("Valuation Wizard (5-Year Forecast)")
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="3y", auto_adjust=True, interval="1d").tz_localize(None) # Added auto_adjust, interval, tz_localize
        financials = ticker.financials
        info = ticker.info

        # Corrected code to address the warning
        # Ensure 'Total Revenue' exists before calling pct_change or iloc
        total_revenue_series = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else pd.Series()
        
        rev_g = (total_revenue_series.pct_change(periods=-1, fill_method=None).mean()) * 100 if not total_revenue_series.empty else 5.0 # FIX: fill_method=None
        
        gross_profit_val = financials.loc['Gross Profit'].iloc[0] if 'Gross Profit' in financials.index else 0
        total_revenue_val = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
        gm = (gross_profit_val / total_revenue_val) * 100 if total_revenue_val > 0 else 50.0
        
        op_inc = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else 0
        rev = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
        opex_r = ((rev - op_inc) / rev) * 100 if rev > 0 else 30.0
        
        hist_ev_ebitda = info.get('enterpriseToEbitda', 15)
    except Exception as e:
        st.warning(f"Could not fetch historical data for context in Valuation Wizard: {e}.") # Specific error message
        rev_g, gm, opex_r, hist_ev_ebitda = 5.0, 50.0, 30.0, 15.0

    with st.expander("Step 1: Set Your Fundamental Assumptions", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            # Ensure float conversion and default values are robust
            rev_growth = st.slider("5Y Revenue Growth Rate (%)", -10.0, 40.0, round(float(rev_g or 5.0), 1), 0.5, key=f"rev_{ticker_symbol}")
            gross_margin = st.slider("5Y Gross Profit Margin (%)", 0.0, 100.0, round(float(gm or 50.0), 1), 0.5, key=f"gm_{ticker_symbol}")
            op_ex_ratio = st.slider("Operating Expenses as % of Sales", 0.0, 100.0, round(float(opex_r or 30.0), 1), 0.5, key=f"opex_{ticker_symbol}")
        with col2:
            ev_ebitda = st.slider("Terminal EV/EBITDA Multiple", 5.0, 40.0, round(float(hist_ev_ebitda or 15.0), 1), 0.5, key=f"ev_{ticker_symbol}")
            shares_growth = st.slider("Annual Share Count Growth (%)", -5.0, 5.0, -1.0, 0.1, key=f"shares_{ticker_symbol}")
            tax_rate_input = st.slider("Effective Tax Rate (%)", 15.0, 35.0, 21.0, 0.5, key=f"tax_{ticker_symbol}")

    if st.button("Calculate Fundamental Price Target", type="primary", key=f"calc_{ticker_symbol}"):
        with st.spinner("Running model..."):
            price_target, eps_cagr, commentary = valuation_wizard(ticker_symbol, rev_growth/100.0, gross_margin/100.0, op_ex_ratio/100.0, shares_growth/100.0, ev_ebitda, tax_rate_input/100.0)
        st.subheader("Model Results")
        if price_target and np.isfinite(price_target):
            current_price = history['Close'].iloc[-1]
            upside_pct = ((price_target - current_price) / current_price) * 100 if current_price > 0 else 0
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Fundamental Price Target (5Y)", f"${price_target:,.2f}")
            res_col2.metric("Potential Upside", f"{upside_pct:.1f}%", delta=f"{upside_pct:.1f}%" if abs(upside_pct) > 0.1 else None)
            if eps_cagr and np.isfinite(eps_cagr):
                res_col3.metric("Implied 5Y EPS Growth (CAGR)", f"{eps_cagr:.2%}")
            else:
                res_col3.info(commentary)
        else:
            st.error(f"Valuation Failed. Reason: {commentary}")
# --- NEW FUNCTION: calculate_idio_variance_single_stock(...) ---
# --- NEW FUNCTION: calculate_idio_variance_single_stock(...) ---
# --- NEW FUNCTION: calculate_idio_variance_single_stock(...) ---
# --- NEW FUNCTION: calculate_idio_variance_single_stock(...) ---
def calculate_idio_variance_single_stock(returns_dict, benchmark_returns, common_window=252):
    """
    Calculates the average annualized idiosyncratic variance for single stocks in the universe.
    This is IVar_n=1 from the formulas.
    
    Args:
        returns_dict (dict): Dictionary where keys are tickers and values are pd.Series of log returns.
        benchmark_returns (pd.Series): Log returns of the benchmark (e.g., SPY).
        common_window (int): The lookback window for alignment and regression.

    Returns:
        float: Average annualized idiosyncratic variance across the universe, or np.nan.
    """
    idio_variances = []
    
    # Ensure benchmark_returns are valid and not empty
    if benchmark_returns.empty or benchmark_returns.isnull().all():
        logging.warning("Benchmark returns are empty or all NaN for idiosyncratic variance calculation. Skipping.")
        return np.nan

    # Get the index of the benchmark returns directly (should be DatetimeIndex from prior processing)
    benchmark_idx = benchmark_returns.index
    
    for ticker, stock_returns in returns_dict.items():
        # Ensure stock_returns are valid and not empty for the current ticker
        if stock_returns.empty or stock_returns.isnull().all() or len(stock_returns.dropna()) < common_window:
            logging.debug(f"Skipping {ticker}: Insufficient data for common_window ({len(stock_returns.dropna())} < {common_window}).")
            continue

        # Get the index of the stock returns directly (should be DatetimeIndex from prior processing)
        stock_idx = stock_returns.index

        # Find the intersection of indices
        intersected_idx = stock_idx.intersection(benchmark_idx)
        
        if intersected_idx.empty: # If no overlap, skip
            logging.debug(f"Skipping {ticker}: No common dates with benchmark after intersection.")
            continue
            
        # Ensure intersected_idx is sorted. Intersection usually returns sorted, but explicit sort can't hurt.
        if not intersected_idx.is_monotonic_increasing:
            intersected_idx = intersected_idx.sort_values()

        window_val = int(common_window) # Ensure window_val is integer

        # --- REPLACING .tail() WITH DIRECT SLICING AND ADDING DEBUG LOGGING ---
        logging.info(f"DEBUG: Before slicing for {ticker}. intersected_idx type: {type(intersected_idx)}. Length: {len(intersected_idx)}. Window_val: {window_val}. First 5: {intersected_idx[:5]}, Last 5: {intersected_idx[-5:]}")
        
        # Use slicing instead of .tail() - this is functionally equivalent for positive window_val
        # and more robust against potential obscure Pandas Index bugs.
        common_idx = intersected_idx[-window_val:] 

        logging.info(f"DEBUG: After slicing for {ticker}. common_idx type: {type(common_idx)}. Length: {len(common_idx)}. First 5: {common_idx[:5]}, Last 5: {common_idx[-5:]}")

        # Ensure common_idx has enough points for stable regression *after* taking the tail
        if len(common_idx) < 30: 
            logging.debug(f"Skipping {ticker}: Not enough common data points ({len(common_idx)}) after slicing for regression (min 30).")
            continue

        aligned_stock_returns = stock_returns.loc[common_idx]
        aligned_benchmark_returns = benchmark_returns.loc[common_idx].values.reshape(-1, 1)

        # Additional check for variance before regression
        if aligned_benchmark_returns.std() < 1e-6: # If benchmark returns are constant/zero
            logging.debug(f"Skipping {ticker}: Benchmark returns are constant for common_idx. Cannot regress.")
            # If benchmark is constant, residual variance is just stock's variance
            if aligned_stock_returns.std() > 1e-6:
                idio_variances.append(aligned_stock_returns.var() * 252) 
            continue

        try:
            model = LinearRegression().fit(aligned_benchmark_returns, aligned_stock_returns)
            residuals = aligned_stock_returns - model.predict(aligned_benchmark_returns)
            if np.isfinite(np.var(residuals)): # Ensure the calculated variance is finite
                idio_variances.append(np.var(residuals) * 252) # Annualize
            else:
                logging.warning(f"Idiosyncratic variance for {ticker} is non-finite after regression. Skipping.")
        except Exception as e:
            logging.warning(f"Failed to calculate idiosyncratic variance for {ticker} due to regression error: {e}. Skipping.")
            continue

    if idio_variances:
        return np.mean(idio_variances) # Return average IVar_n=1 across the universe
    logging.warning("No valid idiosyncratic variances calculated for any ticker in the universe. Returning NaN.")
    return np.nan
# --- NEW FUNCTION: calculate_terminal_wealth_metrics(...) ---
def calculate_terminal_wealth_metrics(
    portfolio_sharpe, benchmark_sharpe,
    port_idio_var_n1_avg, num_port_stocks,
    comparison_target_vol=0.20,
    benchmark_idio_var_n1_avg=None, num_bm_stocks=500,
    leverage_f=1.0, time_horizon_years=5, risk_free_rate=0.04
):
    """
    Calculates Terminal Wealth Ratio (TWR) and Probability of Losing (P(TP < TB))
    based on the provided formulas.

    Args:
        portfolio_sharpe (float): Sharpe Ratio of the portfolio.
        benchmark_sharpe (float): Sharpe Ratio of the benchmark.
        port_idio_var_n1_avg (float): Average annualized idiosyncratic variance of a single stock in the portfolio's universe.
        num_port_stocks (int): Number of stocks in the portfolio.
        comparison_target_vol (float, optional): The annualized volatility level at which both portfolios are compared for Sharpe-driven growth differences. Defaults to 0.15 (15%).
        benchmark_idio_var_n1_avg (float, optional): Average annualized idiosyncratic variance of a single stock in the benchmark's universe.
                                                     Defaults to port_idio_var_n1_avg if not provided.
        num_bm_stocks (int, optional): Number of stocks in the benchmark. Defaults to 500.
        leverage_f (float, optional): Effective leverage multiplier. Defaults to 1.0.
        time_horizon_years (int, optional): Time horizon in years. Defaults to 5.
        risk_free_rate (float, optional): Annualized risk-free rate. Defaults to 0.04.

    Returns:
        tuple: (twr, prob_losing) or (np.nan, np.nan) if inputs are invalid.
    """
    # --- Robustly convert all inputs to float/int and handle None/NaN explicitly ---
    p_sharpe = float(portfolio_sharpe) if pd.notna(portfolio_sharpe) else np.nan
    bm_sharpe = float(benchmark_sharpe) if pd.notna(benchmark_sharpe) else np.nan
    p_idio_var_n1_avg = float(port_idio_var_n1_avg) if pd.notna(port_idio_var_n1_avg) else np.nan
    n_p = int(num_port_stocks) if pd.notna(num_port_stocks) else 0 # num_port_stocks can be 0, but must be int
    comp_target_vol = float(comparison_target_vol) if pd.notna(comparison_target_vol) else np.nan
    bm_idio_var_n1_avg_final = float(benchmark_idio_var_n1_avg) if pd.notna(benchmark_idio_var_n1_avg) else p_idio_var_n1_avg
    n_bm = int(num_bm_stocks) if pd.notna(num_bm_stocks) else 500 # num_bm_stocks can be 0 or small, use 500 as safe default
    lev_f = float(leverage_f) if pd.notna(leverage_f) else np.nan
    time_h_years = int(time_horizon_years) if pd.notna(time_horizon_years) else np.nan
    rf_rate = float(risk_free_rate) if pd.notna(risk_free_rate) else np.nan


    # --- Primary Validation: Check for essential numeric inputs ---
    if not all(np.isfinite([p_sharpe, bm_sharpe, p_idio_var_n1_avg, comp_target_vol, lev_f, time_h_years])):
        logging.warning("TWR/Prob Losing: Essential numeric inputs are non-finite after conversion. Returning NaN.")
        return np.nan, np.nan
    if n_p == 0:
        logging.warning("TWR/Prob Losing: Number of portfolio stocks is zero. Returning NaN.")
        return np.nan, np.nan
    if n_bm == 0: # Avoid division by zero for benchmark stocks
        logging.warning("TWR/Prob Losing: Number of benchmark stocks is zero. Using default of 500 for calculation stability.")
        n_bm = 500
    
    # Calculate idiosyncratic variance drag difference term
    # IVar_p = (1/nP - 1/nBM) * IVar_n=1
    # Ensure n_p and n_bm are not 0 before division
    idio_variance_difference_term_unlevered = (1/n_p - 1/n_bm) * p_idio_var_n1_avg 
    
    # --- TWR Calculation ---
    # Component 1: Difference in growth rate from Sharpe Ratios at a common target volatility
    sharpe_driven_growth_diff_annual = comp_target_vol * (p_sharpe - bm_sharpe) * lev_f 
    
    # Component 2: Geometric drag difference from idiosyncratic variance
    diversification_drag_diff_term_annual = 0.5 * idio_variance_difference_term_unlevered * (lev_f**2)
    
    # Total annualized geometric growth rate difference (gp - gb)
    total_geometric_growth_diff_annual = sharpe_driven_growth_diff_annual - diversification_drag_diff_term_annual
    
    # Terminal Wealth Ratio (TWR)
    twr = np.exp(total_geometric_growth_diff_annual * time_h_years)

    # --- Probability of Losing Calculation ---
    # Prob(TW_P < TW_BM) = Φ( (1/sqrt(2)) * sqrt((1/nP - 1/nBM) * IVar_n=1 * f^2 * t) )
    prob_losing_var_term = idio_variance_difference_term_unlevered * (lev_f**2) * time_h_years
    
    if prob_losing_var_term <= 0:
        # If the diversification difference term is non-positive, it implies a diversification advantage or no penalty.
        # This means the portfolio is less likely to lose due to this specific component.
        prob_losing_z_score = -5.0 # A very low Z-score implying near-zero probability
    else:
        prob_losing_z_score = np.sqrt(prob_losing_var_term) / np.sqrt(2) 
    
    # Ensure norm is imported from scipy.stats
    prob_losing = norm.cdf(prob_losing_z_score)

    return twr, prob_losing
def get_correlated_stocks(selected_ticker, returns_dict, results_df, correlation_threshold=0.40):
    """
    Finds other tickers correlated with the selected ticker. If a threshold is provided,
    it filters by correlation first. It enriches the data and calculates a pairs score.
    """
    if selected_ticker not in returns_dict or len(returns_dict) < 2: # Added robustness check
        return pd.DataFrame()

    try:
        all_returns_df = pd.concat(returns_dict, axis=1)
    except Exception as e:
        logging.error(f"Failed to concat returns_dict: {e}"); return pd.DataFrame()

    recent_returns = all_returns_df.tail(90).fillna(0.0)
    if selected_ticker not in recent_returns.columns: return pd.DataFrame()

    correlations_to_selected = recent_returns.corr()[selected_ticker].drop(selected_ticker, errors='ignore')
    
    # --- KEY CHANGE: Conditional Filtering ---
    if correlation_threshold is not None: # User can pass None to get all correlations
        # If a threshold is given, filter for high correlation first
        corr_df = correlations_to_selected[correlations_to_selected.abs() >= correlation_threshold].to_frame()
    else:
        # Otherwise, consider all stocks
        corr_df = correlations_to_selected.to_frame()

    if corr_df.empty: return pd.DataFrame()
        
    corr_df = corr_df.rename(columns={selected_ticker: 'Correlation'})

    # Enrich with additional data
    required_cols = ['Ticker', 'Relative_Z_Score', 'PE_Ratio', 'Return_63d']
    # Check if results_df actually has these columns before trying to access
    if all(col in results_df.columns for col in required_cols):
        additional_info = results_df[required_cols].set_index('Ticker')
        # Use .join safely, ensuring only common indices are joined
        corr_df = corr_df.join(additional_info, how='left') # Changed to left join to keep all correlations

    # Calculate the Pairs Trade Score
    # Ensure selected_stock_z_score is retrieved safely
    selected_stock_data = results_df[results_df['Ticker'] == selected_ticker].set_index('Ticker')
    if not selected_stock_data.empty and 'Relative_Z_Score' in selected_stock_data.columns:
        selected_stock_z_score = selected_stock_data.loc[selected_ticker, 'Relative_Z_Score']
        corr_df['Z_Score_Divergence'] = (corr_df['Relative_Z_Score'] - selected_stock_z_score).abs()
        corr_df['Pairs_Score'] = corr_df['Correlation'].abs() * corr_df['Z_Score_Divergence']
    else:
        corr_df['Z_Score_Divergence'] = np.nan
        corr_df['Pairs_Score'] = np.nan
        logging.warning(f"Could not retrieve Relative_Z_Score for {selected_ticker} for Pairs Score calculation.")

    return corr_df

def display_stock_dashboard(ticker_symbol, results_df, returns_dict, etf_histories):
    """Orchestrator function to display the entire individual stock dashboard."""
    st.header(f"🔬 Detailed Dashboard for {ticker_symbol}")
    try:
        daily_history = yf.Ticker(ticker_symbol).history(period="3y", auto_adjust=True, interval="1d").tz_localize(None) # Added tz_localize(None)
        if daily_history.empty: # Added robustness check
            st.warning("Could not fetch detailed daily history for this ticker.")
            return
        
        # Ensure the ticker is in results_df before trying to access iloc[0]
        stock_data_row = results_df[results_df['Ticker'] == ticker_symbol]
        if stock_data_row.empty:
            st.error(f"Stock data for {ticker_symbol} not found in the processed results.")
            return
        stock_data = stock_data_row.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Error fetching data for dashboard: {e}")
        return

    if 'display_signal_sigma_checklist' in globals():
        display_signal_sigma_checklist(stock_data, daily_history)
        st.divider()

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        display_ma_deviation(daily_history)
        
        c1_tech, c2_tech, c3_tech = st.columns(3)
        with c1_tech:
            std_dev_reg, trend_str = get_regression_metrics(daily_history)
            st.metric("Std Dev From Trend", f"{std_dev_reg:.4f}")
        with c2_tech:
            st.metric("Medium-Term Trend", trend_str)
        with c3_tech:
            hurst_value = stock_data.get('Hurst_Exponent')
            if pd.notna(hurst_value):
                hurst_interpretation = "Trending" if hurst_value > 0.55 else "Mean-Reverting" if hurst_value < 0.45 else "Random"
                st.metric("Hurst Exponent", f"{hurst_value:.3f}", delta=hurst_interpretation, delta_color="off")
            else:
                st.metric("Hurst Exponent", "N/A")

        st.subheader("Daily Risk Range (ATR-based)")
        risk_low, risk_high, last_price, pct_change = get_daily_risk_range(daily_history)
        if not pd.isna(risk_low):
            c1_atr, c2_atr, c3_atr = st.columns(3)
            c1_atr.metric("Low", f"${risk_low:,.2f}")
            c2_atr.metric("Last", f"${last_price:,.2f}", f"{pct_change:.2f}%")
            c3_atr.metric("High", f"${risk_high:,.2f}")
        else:
            st.info("Not enough data for ATR calculation.")
        
        display_momentum_bar(ticker_symbol, daily_history)


    with col2:
        st.subheader(f"Actionable Peer Analysis (90d)")
        
        sort_by = st.radio(
            "Sort Peers By:",
            ('Pairs Score (Best Opportunities)', 'Correlation (Truest Peers)'),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # --- KEY CHANGE: Dynamic Function Calls and Sorting ---
        if sort_by == 'Pairs Score (Best Opportunities)':
            # Call the function WITHOUT a threshold to search the whole market
            # Added `returns_dict` to the function call as it's required for `get_correlated_stocks`
            correlated_stocks_df = get_correlated_stocks(ticker_symbol, returns_dict, results_df, correlation_threshold=None) # Passed None to search whole market
            # Sort the full results by the Pairs Score
            final_df = correlated_stocks_df.sort_values('Pairs_Score', ascending=False).head(15) # Show top 15 opportunities
        else: # Sort by Correlation
            # Call the function WITH a threshold to find only true peers
            correlated_stocks_df = get_correlated_stocks(ticker_symbol, returns_dict, results_df, correlation_threshold=0.6)
            # Sort the filtered results by Correlation
            final_df = correlated_stocks_df.sort_values('Correlation', key=abs, ascending=False).head(15) # Show top 15 peers

        if not final_df.empty:
            display_cols = [
                'Correlation', 
                'Relative_Z_Score', 
                'PE_Ratio', 
                'Return_63d',
                'Pairs_Score'
            ]
            
            st.dataframe(
                final_df[display_cols],
                use_container_width=True,
                column_config={
                    "Correlation": st.column_config.NumberColumn(format="%.2f"),
                    "Relative_Z_Score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                    "PE_Ratio": st.column_config.NumberColumn("P/E Ratio", format="%.1f"),
                    "Return_63d": st.column_config.NumberColumn("3-Mo Return", format="%.1f%%"),
                    "Pairs_Score": st.column_config.ProgressColumn(
                        "Pairs Score",
                        help="Highlights potential pairs trades. = Correlation * |Z-Score Divergence|.",
                        format="%.2f",
                        min_value=0,
                        max_value=max(final_df['Pairs_Score'].max(), 3),
                    ),
                }
            )
        else:
            st.info(f"No significant peers found based on the selected criteria.")

# --- NEW FUNCTION: apply_regime_weights(...) ---
################################################################################
# SECTION 2: MAIN APPLICATION LOGIC (CORRECTED)
################################################################################
# SECTION 2: MAIN APPLICATION LOGIC (CORRECTED WITH CACHING)
################################################################################
# SECTION 2: MAIN APPLICATION LOGIC (WITH INTERACTIVE PLOTS)
################################################################################
# SECTION 2: MAIN APPLICATION LOGIC (COMPLETE AND FINAL)
################################################################################
################################################################################
# SECTION 2: MAIN APPLICATION LOGIC (COMPLETE - UNCONSTRAINED PURE ALPHA STRATEGY)
# SECTION 2: MAIN APPLICATION LOGIC (COMPLETE, ENHANCED, AND CORRECTED)
################################################################################
# From SECTION 2: MAIN APPLICATION LOGIC
def main():
    st.title("Quantitative Portfolio Analysis")
    st.sidebar.header("Controls")
    if st.sidebar.button("Clear Cache & Re-run All", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # --- UI Controls ---
    st.sidebar.subheader("Portfolio Construction")
    weighting_method_ui = st.sidebar.selectbox(
        "Portfolio Weighting Method",
        ["Equal Weight", "Inverse Volatility", "Log Log Sharpe Optimized"]
    )
    corr_window = st.sidebar.slider("Correlation Window (days)", 30, 180, 90, 30)
    
    st.sidebar.subheader("🛡️ Portfolio Hedging")
    hedge_risks = st.sidebar.multiselect(
        "Select Risks to Hedge", options=etf_list, default=['SPY', 'QQQ']
    )
    target_net_exposure_ui = st.sidebar.slider(
        "Target Net Exposure (%)", -100.0, 100.0, 0.0, 5.0,
        help="Set your desired final market exposure. 0% is market-neutral."
    ) / 100.0
    lambda_uncertainty_ui = st.sidebar.slider(
        "Hedging Conservatism (Lambda)", 0.1, 5.0, 0.5, 0.1
    )

    # --- Data Fetching and Initial Processing ---
    with st.spinner("Fetching ETF and Macroeconomic histories..."):
        etf_histories = fetch_all_etf_histories(etf_list)
        macro_data = fetch_macro_data() # Fetch macro data, but it won't be used for weight adjustments now
    st.success("All historical data loaded.")

    with st.spinner(f"Processing {len(tickers)} tickers..."):
        results_df, failed_tickers, returns_dict = process_tickers(tickers, etf_histories, sector_etf_map)
        results_df = calculate_industry_relative_factors(results_df)
    if results_df.empty:
        st.error("Fatal Error: No tickers could be processed."); st.stop()
    st.success(f"Successfully processed {len(results_df)} tickers.")
    if failed_tickers:
        st.expander("Show Failed Tickers").warning(f"{len(failed_tickers)} tickers failed: {', '.join(failed_tickers)}")

    # --- CORRECTED ORDER OF OPERATIONS ---
    # 1. First, we must clean the raw returns.
    with st.spinner("Applying Winsorization to clean return data..."):
        winsorized_returns_dict = winsorize_returns(returns_dict, lookback_T=126, d_max=7.0)
    st.success("Return data cleaned successfully.")

    # 2. Now, with clean returns, we can generate advanced signals.
    with st.spinner("Generating advanced signals (Carry, Mean Reversion)..."):
        results_df = calculate_relative_carry(results_df)
        if winsorized_returns_dict: # Check if winsorized_returns_dict is not empty
            normalized_prices = get_all_prices_and_vols(list(winsorized_returns_dict.keys()), winsorized_returns_dict)
            results_df = calculate_cs_mean_reversion(results_df, normalized_prices)
        else:
            st.warning("Winsorized returns dictionary is empty. Skipping normalized price and CS Mean Reversion calculations.")
            results_df['CS_Mean_Reversion'] = np.nan # Ensure column exists if it was to be used
    st.success("Advanced signals generated.")

    # --- Automatic Factor Weighting ---
    st.sidebar.subheader("Factor Weighting")
    active_weights = default_weights # Initialize with defaults
    active_rationale = pd.DataFrame() # Initialize as empty
    stability_results = None # Initialize stability_results
    
    # Macro regime analysis is displayed later in Tab 3, but no longer impacts weights directly here.
    
    with st.spinner("Analyzing stability of all factors (incl. advanced)..."):
        all_possible_metrics = list(default_weights.keys())
        auto_weights, rationale_df, stability_results = run_factor_stability_analysis(
            results_df, all_possible_metrics, REVERSE_METRIC_NAME_MAP
        )
        
        if not rationale_df.empty:
            rationale_df['Signal Direction'] = np.where(rationale_df['avg_sharpe_coeff'] >= 0, 'Positive ✅', 'Inverted 🔄')
            active_weights = auto_weights # Use auto_weights directly, no regime adjustment
            active_rationale = rationale_df # Update active_rationale
        else:
            st.warning("Factor stability analysis failed, using default weights.")
            active_weights = default_weights # Fallback if stability analysis fails
            active_rationale = pd.DataFrame() # Ensure it's empty

    if not active_rationale.empty:
        with st.sidebar.expander("View Factor Model Rationale", expanded=True):
            display_rationale = active_rationale[['avg_sharpe_coeff', 'consistency_score', 'Signal Direction']].copy()
            display_rationale['Final_Weight'] = display_rationale.index.map(lambda short_name: active_weights.get(METRIC_NAME_MAP.get(short_name, short_name), 0.0))
            st.dataframe(display_rationale.loc[display_rationale['Final_Weight'] > 0.1].sort_values('Final_Weight', ascending=False), use_container_width=True)
    
    user_weights = active_weights # user_weights now reflect all-weather adjustments (or defaults)
    
    # --- Scoring Block (Unconstrained Pure Alpha) ---
    alpha_score = pd.Series(0.0, index=results_df.index)
    if not results_df.empty and user_weights: # Check if results_df is not empty AND user_weights dictionary is not empty
        for long_name, weight in user_weights.items():
            if weight > 0:
                short_name = REVERSE_METRIC_NAME_MAP.get(long_name)
                # Ensure short_name exists in results_df columns
                if short_name in results_df.columns:
                    rank_ascending = True  # Initial default: higher factor value is typically better for scoring

                    # Prioritize: Use active_rationale if available and decisive
                    if not active_rationale.empty and short_name in active_rationale.index:
                        sharpe_coeff = active_rationale.loc[short_name, 'avg_sharpe_coeff']
                        # If the Sharpe coefficient is valid and non-zero, it determines the ranking direction.
                        if pd.notna(sharpe_coeff) and abs(sharpe_coeff) > 1e-6: # Use a small epsilon to treat near-zero as zero
                            rank_ascending = (sharpe_coeff > 0)
                        # Else (if sharpe_coeff is NaN or effectively zero), rank_ascending remains True (the initial default).
                        # This means if the factor stability analysis doesn't give a clear direction, we default to higher is better.

                    # No other hardcoded `if` blocks for `rank_ascending` are used.
                    # The decision now solely rests on `active_rationale` or the initial default.

                    rank_series = results_df[short_name].rank(pct=True, ascending=rank_ascending)
                    alpha_score += rank_series.fillna(0.5) * weight
                else:
                    logging.warning(f"Metric '{long_name}' (short name '{short_name}') not found in results_df columns. Contributing neutral score.")
                    alpha_score += 0.5 * weight # Neutral contribution if factor is not found
        # Remaining part of the scoring block (outside the loop)
        def z_score_robust(series):
            if series.empty:
                return pd.Series(np.nan, index=series.index)
            median_val = series.median()
            mad_val = (series - median_val).abs().median()
            
            # Add a small epsilon to MAD to prevent division by zero for constant series
            if mad_val < 1e-6: # Check if MAD is effectively zero
                return pd.Series(0.0, index=series.index) # If no dispersion, all scores are 0
            
            # 0.6745 is a scaling factor to make MAD a consistent estimator of standard deviation
            # for normally distributed data.
            return 0.6745 * (series - median_val) / mad_val

        results_df['Score'] = z_score_robust(alpha_score)
    else:
        results_df['Score'] = np.nan # If no scores can be calculated
        st.warning("Cannot calculate Alpha Scores: results_df or user_weights are empty/invalid.")

    top_15_df = results_df.sort_values('Score', ascending=False).head(15).copy()
    top_15_tickers = top_15_df['Ticker'].tolist()

    # --- Portfolio Overview & Hedging ---
    st.header("📈 Portfolio Overview & Hedging")
    if not top_15_tickers:
        st.warning("No stocks for portfolio construction (top 15 list is empty)."); st.stop()
    
    st.subheader(f"Core Portfolio (Long Book): Weights by {weighting_method_ui}")
    # Ensure winsorized_returns_dict is not empty for portfolio_returns_df
    if not winsorized_returns_dict:
        st.error("Cannot construct portfolio returns: Winsorized returns dictionary is empty."); st.stop()
        
    # --- IMPORTANT CORRECTION: Convert log returns to simple returns for portfolio_returns_df ---
    # winsorized_returns_dict contains LOG returns. Most portfolio calculations (Sharpe, variance, etc.)
    # typically expect simple returns.
    simple_returns_dict = {
        ticker: np.expm1(log_rets) # np.expm1(x) calculates exp(x) - 1, converting log returns to simple returns
        for ticker, log_rets in winsorized_returns_dict.items()
    }
    portfolio_returns_df = pd.DataFrame(simple_returns_dict).reindex(columns=top_15_tickers).dropna(how='all')
    
    if portfolio_returns_df.empty:
        st.error("Portfolio returns DataFrame is empty after reindexing/dropping NaNs. Cannot proceed with portfolio analysis."); st.stop()

    _, cov_matrix = calculate_correlation_matrix(top_15_tickers, winsorized_returns_dict, window=corr_window) # Note: winsorized_returns_dict is LOG returns here
    method_map = {"Equal Weight": "equal", "Inverse Volatility": "inv_vol", "Log Log Sharpe Optimized": "log_log_sharpe"}
    # Pass portfolio_returns_df (simple returns) to calculate_weights
    p_weights = calculate_weights(portfolio_returns_df, method=method_map.get(weighting_method_ui, "equal"), cov_matrix=cov_matrix)
    
    if p_weights.empty:
        st.error("Portfolio weights could not be calculated. Cannot proceed with portfolio analysis."); st.stop()

    weights_df = p_weights.reset_index(); weights_df.columns = ['Ticker', 'Weight']
    weights_df = pd.merge(weights_df, top_15_df[['Ticker', 'Name', 'Sector']], on='Ticker', how='left')
    st.dataframe(weights_df[['Ticker', 'Name', 'Weight']].sort_values("Weight", ascending=False), use_container_width=True)

    with st.expander("🔬 View Long Book Risk Analysis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Average Beta Exposure")
            
            if 'Beta_to_SPY' in top_15_df.columns and not top_15_df['Beta_to_SPY'].empty and pd.notna(top_15_df['Beta_to_SPY']).any():
                avg_beta_val = top_15_df['Beta_to_SPY'].mean()
                st.metric("Average Beta (Conservative)", f"{avg_beta_val:.2f}", help="Weighted average of up and down market betas.")
            else:
                st.info("Beta to SPY not available for current portfolio.") 
            
            if 'Beta_Down_Market' in top_15_df.columns and not top_15_df['Beta_Down_Market'].empty and pd.notna(top_15_df['Beta_Down_Market']).any():
                avg_down_beta = top_15_df['Beta_Down_Market'].mean()
                st.metric("Avg. Down-Market Beta", f"{avg_down_beta:.2f}", help="Portfolio sensitivity during market declines. Lower is more defensive.")
            else:
                st.info("Beta_Down_Market column was not found or contains no valid data.") 
            
            if not top_15_df.empty and 'Sector' in top_15_df.columns:
                sector_counts = top_15_df['Sector'].value_counts()
                if not sector_counts.empty:
                    sector_fig = plot_sector_concentration(sector_counts)
                    st.plotly_chart(sector_fig, use_container_width=True)
                else:
                    st.info("No sector data available for concentration plot.")
            else:
                st.info("Top 15 DataFrame is empty or missing 'Sector' column for concentration plot.")

        with col2:
            st.subheader("Factor ETF Correlation (Portfolio DNA)")
            factor_correlations = calculate_portfolio_factor_correlations(weights_df, etf_histories)
            if not factor_correlations.empty:
                factor_fig = plot_factor_correlations(factor_correlations.head(10))
                st.plotly_chart(factor_fig, use_container_width=True)
            else:
                st.info("No factor correlations could be calculated for the portfolio.")

    # --- Hedging Calculation and Display ---
    # --- Hedging Calculation and Display ---
    hedge_weights = pd.Series(dtype=float)
    if hedge_risks and not portfolio_returns_df.empty: # Only try to hedge if there's a portfolio (simple returns)
        with st.spinner(f"Calculating robust, regime-aware hedge for {', '.join(hedge_risks)}..."):
            hedge_instrument_returns_dict = {
                etf: etf_histories[etf]['Close'].pct_change(fill_method=None).dropna() 
                for etf in hedge_risks 
                if etf in etf_histories and not etf_histories[etf].empty
            }
            if hedge_instrument_returns_dict:
                hedge_instrument_returns_df = pd.DataFrame(hedge_instrument_returns_dict)
                
                aligned_portfolio_returns_for_hedge, aligned_hedge_returns = portfolio_returns_df.align( # Use portfolio_returns_df (simple returns)
                    hedge_instrument_returns_df, join='inner', axis=0
                )
                if not aligned_portfolio_returns_for_hedge.empty and not aligned_hedge_returns.empty:
                    aligned_portfolio_returns_for_hedge = aligned_portfolio_returns_for_hedge.fillna(0.0)
                    aligned_hedge_returns = aligned_hedge_returns.fillna(0.0)
                    
                    # calculate_robust_hedge_weights returns negative weights for short positions
                    hedge_weights = calculate_robust_hedge_weights(
                        aligned_portfolio_returns_for_hedge, 
                        aligned_hedge_returns, 
                        lambda_uncertainty=lambda_uncertainty_ui
                    )
                else:
                    st.info("Not enough aligned data for hedging calculation after filtering. Skipping hedge.")
            else:
                st.info("No valid hedge instruments with history found. Skipping hedging calculation.")

    # Display section for hedging
    if not hedge_weights.empty and not portfolio_returns_df.empty:
        st.subheader("Hedge Portfolio & Final Exposure")
        
        long_exposure_amount = 1.0 # The long book is 100% of capital, represented as 1.0 notional

        current_optimal_hedge_sum = hedge_weights.sum() 

        k_scaling_factor = 1.0 # Default to 1 (apply optimal hedge directly)

        if current_optimal_hedge_sum != 0: # Avoid division by zero
            k_scaling_factor = (target_net_exposure_ui - long_exposure_amount) / current_optimal_hedge_sum
        
        scaled_hedge_weights = hedge_weights * k_scaling_factor

        total_short_exposure_magnitude = abs(min(0, scaled_hedge_weights.sum())) 
        
        final_net_exposure_calculated = long_exposure_amount + scaled_hedge_weights.sum() 
        
        hedge_display_df = scaled_hedge_weights.reset_index()
        hedge_display_df.columns = ['Hedge Instrument', 'Weight']
        
        # CRITICAL FIX: The right-hand side should refer to `hedge_display_df`, not `display_hedge_df`
        display_hedge_df = hedge_display_df[abs(hedge_display_df['Weight']) > 1e-5].copy() 
        
        display_hedge_df['Weight %'] = display_hedge_df['Weight'] * 100 
        
        if scaled_hedge_weights.sum() > 0.01 and target_net_exposure_ui > long_exposure_amount: # If hedge became a long component
             st.write("These are the **scaled NOTIONAL adjustments** for your target net exposure (negative implies short).")
        else:
             st.write("These are the **scaled short positions** required to meet your target net exposure.")

        st.dataframe(display_hedge_df[['Hedge Instrument', 'Weight %']].sort_values('Weight %'), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Long Exposure", f"{long_exposure_amount:.1%}")
        col2.metric("Short Exposure (Hedge)", f"{total_short_exposure_magnitude:.1%}") 
        col3.metric("Net Exposure", f"{final_net_exposure_calculated:.1%}")
    elif hedge_risks: 
        st.info("Hedging calculation failed or returned empty weights. Check data availability for hedge instruments.")
    else: 
        st.info("No hedge instruments selected. Portfolio has 100% long exposure.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Long Exposure", "100.0%")
        col2.metric("Short Exposure (Hedge)", "0.0%")
        col3.metric("Net Exposure", "100.0%")
    
    st.divider()

    # --- Initialize benchmark_metrics upfront ---
    benchmark_metrics = get_benchmark_metrics() 
    if not benchmark_metrics:
        benchmark_metrics = {'Volatility': np.nan, 'Sharpe Ratio': np.nan}
        st.warning("Could not fetch benchmark metrics (SPY). Performance comparisons may be incomplete.")
    
    # --- Portfolio Performance Metrics ---
    st.subheader("📊 Portfolio Strategy Performance Metrics")

    # Fetch SPY simple returns (for tracking error)
    spy_simple_returns_series = etf_histories['SPY']['Close'].pct_change(fill_method=None).dropna()
    # Convert SPY returns to LOG RETURNS for consistency with winsorized_returns_dict
    spy_log_returns_series = np.log1p(spy_simple_returns_series.clip(lower=-0.9999))

    # --- Initialize all metrics and relevant DataFrames/Series to safe empty states ---
    ic, ir, malv, twr_val, prob_losing_val, avg_idio_var_n1 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    total_vol = np.nan 
    tracking_error = np.nan 
    estimated_alpha_value = np.nan 

    final_returns = pd.Series(dtype=float) 
    lagged_forecast_ts = pd.Series(dtype=float) 
    valid_tickers_for_metrics = [] 
    
    # NEW: Initialize aligned_returns DataFrames used across sections
    aligned_returns_for_metrics = pd.DataFrame() # General purpose aligned returns
    aligned_returns_for_icir_metrics = pd.DataFrame() # For IC/IR, TE
    aligned_returns_for_twr_metrics = pd.DataFrame() # For TWR, MALV, idio_var_n1

    # NEW: Initialize factor_returns_df for the risk dashboard
    factor_returns_df = pd.DataFrame() # Initialized here, populated in dashboard section
    # Ensure portfolio_returns_df (simple returns) and p_weights are not empty/None
    if portfolio_returns_df.empty or p_weights is None or p_weights.empty:
        st.info("Portfolio returns or weights are not available for performance metrics calculation.")
    else:
        # Align portfolio_returns_df with spy_simple_returns_series (for all metrics expecting simple returns)
        common_idx_for_metrics = portfolio_returns_df.index.intersection(spy_simple_returns_series.index)
        aligned_returns_for_metrics = portfolio_returns_df.loc[common_idx_for_metrics].copy().fillna(0.0)

        if aligned_returns_for_metrics.empty:
            st.info("Aligned portfolio returns are empty for performance metrics calculation due to data alignment issues.")
        else:
            valid_tickers_for_metrics = aligned_returns_for_metrics.columns.tolist()
            if not valid_tickers_for_metrics:
                st.info("No valid tickers remain after aligning portfolio returns for metrics. Skipping performance metrics.")
            else:
                aligned_w = p_weights.reindex(valid_tickers_for_metrics).fillna(0)
                final_returns = (aligned_returns_for_metrics * aligned_w).sum(axis=1) # The portfolio's actual returns (simple returns)

                # Calculate IVar_n=1 for TWR/Prob Losing (requires log returns, use spy_log_returns_series for benchmark)
                if returns_dict and not spy_log_returns_series.empty:
                    # winsorized_returns_dict contains log returns, avg_idio_var_n1 should be based on log returns
                    avg_idio_var_n1 = calculate_idio_variance_single_stock(winsorized_returns_dict, spy_log_returns_series)
                else:
                    logging.warning("Returns dictionary or SPY log returns series is empty/invalid. Cannot calculate avg_idio_var_n1.")

                # --- Calculate alpha_weights and related forecast series ---
                if not top_15_df.empty and 'Ticker' in top_15_df.columns and 'Score' in top_15_df.columns:
                    filtered_top_15_df = top_15_df[top_15_df['Ticker'].isin(valid_tickers_for_metrics)]
                    if not filtered_top_15_df.empty:
                        aligned_scores = filtered_top_15_df.set_index('Ticker')['Score']
                        if not aligned_scores.empty and aligned_scores.sum() != 0:
                            alpha_weights = aligned_scores / aligned_scores.sum()
                            valid_forecast_tickers = list(set(aligned_returns_for_metrics.columns) & set(alpha_weights.index))
                            if valid_forecast_tickers:
                                # Forecast is based on SIMPLE returns of the portfolio constituents
                                forecast_ts_unlagged = (aligned_returns_for_metrics.loc[:, valid_forecast_tickers] * alpha_weights.loc[valid_forecast_tickers]).sum(axis=1)
                                lagged_forecast_ts = forecast_ts_unlagged.shift(1)
                            else:
                                st.info("No valid tickers for alpha forecast after alignment.")
                        else:
                            st.info("Alpha scoring could not be performed (empty or zero sum scores after filtering).")
                    else:
                        st.info("No top_15_df tickers align with valid portfolio metrics tickers for alpha scoring.")
                else:
                    st.info("top_15_df is empty or missing 'Ticker'/'Score' columns for alpha scoring.")

                # Calculate MALV if possible (requires log returns, and a DataFrame of these)
                valid_cov_tickers = list(set(valid_tickers_for_metrics) & set(cov_matrix.columns))
                if len(valid_cov_tickers) > 1:
                    # MALV calculation should use log returns for consistency (as per Mahalanobis definition)
                    # Need to construct a log_returns_df from winsorized_returns_dict
                    # Filter this by common_idx_for_metrics (which represents the intersection of portfolio and SPY simple returns)
                    # This ensures temporal alignment.
                    malv_log_returns_df = pd.DataFrame(winsorized_returns_dict).reindex(columns=valid_cov_tickers).loc[common_idx_for_metrics].dropna(how='all')
                    if not malv_log_returns_df.empty:
                        aligned_cov_matrix = cov_matrix.loc[valid_cov_tickers, valid_cov_tickers]
                        malv, _ = calculate_mahalanobis_metrics(malv_log_returns_df, aligned_cov_matrix)
                    else:
                        st.info("Insufficient valid log returns for Mahalanobis calculation after alignment.")
                else:
                    logging.warning("Insufficient valid tickers for Mahalanobis calculation.")


                # Calculate IC/IR and Tracking Error only if forecast and final returns are available
                if not lagged_forecast_ts.empty and not final_returns.empty and not spy_simple_returns_series.empty:
                    aligned_benchmark_returns_for_icir = spy_simple_returns_series.loc[final_returns.index]
                    if not aligned_benchmark_returns_for_icir.empty:
                        ic, ir = calculate_information_metrics(lagged_forecast_ts, final_returns, aligned_benchmark_returns_for_icir)

                        active_returns = final_returns - aligned_benchmark_returns_for_icir
                        if active_returns.std() > 1e-6:
                            tracking_error = active_returns.std() * np.sqrt(252)
                        else:
                            tracking_error = 0.0
                    else:
                        st.info("Aligned benchmark returns are empty for IC/IR and Tracking Error calculation.")
                else:
                    st.info("Cannot calculate Information Coefficient, Ratio, or Tracking Error (forecast, portfolio, or benchmark returns missing).")


    # --- Display Metrics (using initialized NaN values if not calculated) ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision Matrix Quality (MALV)", f"{malv:.4f}" if pd.notna(malv) else "N/A", help=f"Expected: {2/len(valid_cov_tickers):.4f}" if 'valid_cov_tickers' in locals() and valid_cov_tickers else "N/A")
    col2.metric("Information Coefficient (IC)", f"{ic:.4f}" if pd.notna(ic) else "N/A", help="Lagged correlation of alpha vs returns.")
    col3.metric("Information Ratio (IR)", f"{ir:.4f}" if pd.notna(ir) else "N/A", help="Risk-adjusted return (Sharpe).")

    # --- TWR & Probability of Losing Calculation & Display (using initialized NaN values) ---
    benchmark_sharpe = benchmark_metrics.get('Sharpe Ratio', np.nan)
    
    # All conditions for TWR/Prob Losing must be met:
    if pd.notna(ir) and pd.notna(avg_idio_var_n1) and valid_tickers_for_metrics and pd.notna(benchmark_sharpe):
        with st.sidebar.expander("TWR & Prob Losing Parameters", expanded=False):
            twr_time_horizon = st.slider("Time Horizon (Years)", 1, 30, 5, key="twr_time_horizon")
            twr_risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.5, format="%.1f%%", key="twr_risk_free_rate") / 100.0
            comparison_vol = st.slider("Comparison Target Volatility (%)", 5.0, 30.0, 20.0, 1.0, format="%.1f%%", key="comparison_vol_twr") / 100.0
            num_bm_stocks = st.number_input("Benchmark Stocks (n_BM)", min_value=100, value=500, step=100, key="num_bm_stocks")
            effective_leverage_f = 1.0 + target_net_exposure_ui # Target Net Exposure acts as overall leverage

        twr_val, prob_losing_val = calculate_terminal_wealth_metrics(
            portfolio_sharpe=ir,
            benchmark_sharpe=benchmark_sharpe,
            port_idio_var_n1_avg=avg_idio_var_n1,
            num_port_stocks=len(valid_tickers_for_metrics),
            comparison_target_vol=comparison_vol,
            num_bm_stocks=num_bm_stocks,
            leverage_f=effective_leverage_f,
            time_horizon_years=twr_time_horizon,
            risk_free_rate=twr_risk_free_rate
        )
        
        if pd.notna(twr_val) and pd.notna(prob_losing_val):
            st.subheader(f"🔮 Long-Term Wealth Projection (over {twr_time_horizon} years, at {comparison_vol*100:.1f}% Target Volatility)")
            col_twr_1, col_twr_2 = st.columns(2)
            col_twr_1.metric("Expected Terminal Wealth Ratio (TWR)", f"{twr_val:.2f}x", help="Expected compounded wealth relative to starting capital if benchmark has 1.0x. Higher is better.")
            col_twr_2.metric("Prob. of Losing to Benchmark", f"{prob_losing_val:.1%}", help="Probability your portfolio underperforms the fully diversified benchmark in geometric terms.")
        else:
            st.info("Insufficient data or invalid parameters for TWR/Probability of Losing calculation.")
    else:
        st.info("TWR/Probability of Losing cannot be calculated due to missing Information Ratio, Idiosyncratic Variance, or Benchmark Sharpe.")
    # --- Advanced Risk Dashboard ---
    # --- Advanced Risk Dashboard ---
    # --- Advanced Risk Dashboard ---
# --- Advanced Risk Dashboard ---
    # --- Advanced Risk Dashboard ---
    st.subheader("🔬 Quantitative Risk Dashboard")
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.subheader("Portfolio vs. Benchmark (SPY)")
        with st.spinner("Decomposing portfolio risk..."):
            # Initialize factor_returns_df to an empty DataFrame, will be populated if data is available
            factor_returns_df = pd.DataFrame() 
            temp_factor_returns_df = pd.DataFrame({etf: etf_histories[etf]['Close'].pct_change(fill_method=None) for etf in factor_etfs if etf in etf_histories}).dropna(how='all')
            
            if not temp_factor_returns_df.empty:
                factor_returns_df = temp_factor_returns_df.loc[:, temp_factor_returns_df.std() > 1e-6]
            
            total_var, sys_var, spec_var = np.nan, np.nan, np.nan 

            if not factor_returns_df.empty and not aligned_returns_for_metrics.empty and not weights_df.empty and aligned_returns_for_metrics.std().sum() > 1e-6: 
                factor_cov = factor_returns_df.cov() * 252 
                total_var, sys_var, spec_var = decompose_portfolio_risk(aligned_returns_for_metrics, weights_df, factor_returns_df, factor_cov)
            else:
                if factor_returns_df.empty:
                    st.info("Factor returns for risk decomposition are empty or have zero variance. Cannot decompose portfolio risk.")
                elif aligned_returns_for_metrics.empty or weights_df.empty:
                    st.info("Individual stock returns or weights for the portfolio are empty. Cannot decompose portfolio risk.")
                elif aligned_returns_for_metrics.std().sum() <= 1e-6:
                    st.info("Individual stock returns for the portfolio have zero variance (all constant). Cannot decompose portfolio risk.")
                else:
                    st.info("Could not decompose portfolio risk due to an unhandled data issue.") 

            if pd.notna(total_var) and total_var > 0:
                spec_pct = (spec_var / total_var) * 100
                st.metric("Idiosyncratic Risk %", f"{spec_pct:.1f}%" if pd.notna(spec_pct) else "N/A", f"{spec_pct - 0:.1f}% vs SPY" if pd.notna(spec_pct) else None, help="The portion of risk unique to your stock picks. SPY is 0% by definition. Higher is better.")
                total_vol = np.sqrt(total_var) 
                bench_vol = benchmark_metrics.get('Volatility', np.nan)
                st.metric("Annualized Volatility", f"{total_vol:.2%}" if pd.notna(total_vol) else "N/A", f"{(total_vol - bench_vol)/bench_vol:.1%}" if pd.notna(total_vol) and pd.notna(bench_vol) and bench_vol != 0 else None, help="Total portfolio risk. Delta shows difference vs. SPY.")
                
                # --- Re-introducing Expected Active Return (Alpha) with simplified explanation ---
                expected_alpha_bps = np.nan
                if pd.notna(ir) and pd.notna(tracking_error) and tracking_error > 0:
                    expected_alpha_bps = ir * tracking_error * 10000 # Convert to basis points
                    st.metric(
                        "Expected Active Return (Alpha)", 
                        f"{expected_alpha_bps:.2f} bp/yr", 
                        help="Your portfolio's estimated skill in generating returns above what's expected from its market risk (calculated from Information Ratio and Tracking Error)."
                    )
                    st.caption("This is the extra return your portfolio aims to achieve compared to a basic market-tracking strategy.")
                else:
                    st.info("Expected Active Return (Alpha) cannot be calculated (missing Information Ratio or Tracking Error).")

                # --- Alpha vs. Diversification Hurdle ---
                num_bm_stocks_for_hurdle = locals().get('num_bm_stocks', 500) 
                
                if pd.notna(avg_idio_var_n1) and pd.notna(ir) and ir > 0 and num_bm_stocks_for_hurdle > 0 and len(valid_tickers_for_metrics) > 0: # Ensure valid inputs
                    num_port_stocks_current = len(valid_tickers_for_metrics) 
                    
                    diversification_drag_term = (1/num_port_stocks_current - 1/num_bm_stocks_for_hurdle) * avg_idio_var_n1 / 2
                    
                    st.subheader("🎯 Alpha vs. Diversification Hurdle")
                    st.info("To geometrically outperform a diversified benchmark, your average stock-picking alpha needs to overcome a diversification drag specific to your portfolio size and stock type.")
                    
                    col_alpha_1, col_alpha_2 = st.columns(2)
                    col_alpha_1.metric("Current Portfolio Size", f"{num_port_stocks_current} stocks")
                    
                    if diversification_drag_term > 0:
                        col_alpha_2.metric("Min Alpha to Overcome Drag (Unlevered)", f"{diversification_drag_term * 10000:.2f} bp/yr" if pd.notna(diversification_drag_term) else "N/A", 
                                        help=f"Annualized alpha (in basis points) required to offset the geometric return drag from *under-diversification* relative to a benchmark of {num_bm_stocks_for_hurdle} stocks. Current IVar_n=1: {avg_idio_var_n1:.4f}.")
                    else:
                        col_alpha_2.metric("Diversification Advantage", f"{abs(diversification_drag_term) * 10000:.2f} bp/yr" if pd.notna(diversification_drag_term) else "N/A",
                                            help="Your portfolio is more diversified than the benchmark in this metric, providing a geometric return bonus.")
                    
                    # --- Enhanced Caption for Diversification Hurdle ---
                    if pd.notna(avg_idio_var_n1) and pd.notna(num_port_stocks_current) and pd.notna(num_bm_stocks_for_hurdle):
                        nP_current = num_port_stocks_current
                        nBM = num_bm_stocks_for_hurdle
                        idio_var = avg_idio_var_n1
                        lev_f_val = locals().get('effective_leverage_f', 1.0) # From TWR parameters or default 1.0

                        # Calculate hypothetical drag changes for +5 and -5 stocks
                        drag_plus_5_bps = np.nan
                        nP_plus_5 = nP_current + 5
                        if nP_plus_5 > 0:
                            idio_diff_plus_5 = (1/nP_plus_5 - 1/nBM) * idio_var
                            drag_plus_5_bps = 0.5 * idio_diff_plus_5 * (lev_f_val**2) * 10000

                        drag_minus_5_bps = np.nan
                        nP_minus_5 = max(1, nP_current - 5) # Ensure at least 1 stock
                        if nP_minus_5 > 0:
                            idio_diff_minus_5 = (1/nP_minus_5 - 1/nBM) * idio_var
                            drag_minus_5_bps = 0.5 * idio_diff_minus_5 * (lev_f_val**2) * 10000
                        
                        caption_text = (
                            f"**Impact of Portfolio Size:** With your current {nP_current} stocks, the diversification drag is {diversification_drag_term * 10000:.2f} bp/yr. "
                            f"If you had {nP_plus_5} stocks, this drag would be {drag_plus_5_bps:.2f} bp/yr. "
                            f"If you had {nP_minus_5} stocks, it would be {drag_minus_5_bps:.2f} bp/yr. "
                            f"This illustrates how adding stocks can reduce the geometric drag, leaving more room for your active alpha to shine."
                        )
                        st.caption(caption_text)
                else:
                    st.info("Alpha vs. Diversification Hurdle cannot be calculated (missing Idiosyncratic Variance, Information Ratio, or portfolio/benchmark size).")

                # --- "Average Beta to SPY" Display ---
                if not top_15_df.empty and 'Beta_to_SPY' in top_15_df.columns and pd.notna(top_15_df['Beta_to_SPY']).any():
                    avg_beta_val = top_15_df['Beta_to_SPY'].mean()
                    st.metric(
                        "Average Beta to SPY", 
                        f"{avg_beta_val:.2f}" if pd.notna(avg_beta_val) else "N/A", 
                        delta=f"{(avg_beta_val - 1.0):.2f}" if pd.notna(avg_beta_val) else None, 
                        help="Portfolio sensitivity to the S&P 500. SPY is 1.0 by definition."
                    )
                    # --- Explanation for Beta's Impact ---
                    st.caption(
                        "Your portfolio's Beta to SPY shows how sensitive it is to overall market movements. "
                        "A beta of 1.0 moves with the market. Above 1.0 means more movement, below 1.0 means less. "
                        "Your active alpha is the extra return you generate *beyond* what this market sensitivity would predict."
                    )
                else:
                    st.info("Beta to SPY not available for current portfolio.")

            else: # This 'else' block belongs to the `if pd.notna(total_var) and total_var > 0:` condition
                st.info("Total portfolio variance is not available or is zero. Cannot perform detailed risk decomposition.") 


    with col2:
        st.subheader("Systematic Risk Exposure Breakdown")
        if not aligned_returns_for_metrics.empty and not p_weights.empty and not factor_returns_df.empty and not final_returns.empty and final_returns.std() > 1e-6: 
            portfolio_betas = calculate_portfolio_factor_betas(final_returns, factor_returns_df) 
            meaningful_factors = ['SPY', 'QQQ', 'IWM', 'MTUM', 'QUAL', 'IVE', 'IVW', 'USMV']
            display_betas = portfolio_betas.reindex(meaningful_factors).dropna()
            if not display_betas.empty:
                fig = plot_factor_exposure_breakdown(display_betas)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No meaningful factor betas to display after filtering.")
        else:
            st.info("Portfolio returns, weights, or factor returns are not available for factor exposure analysis.")
    # --- Detailed Reports Section ---
    st.header("📑 Detailed Reports")
    st.sidebar.divider()
    st.sidebar.header("Individual Stock Analysis")
    
    # Default ticker selection logic (made more robust)
    default_ticker = None
    if not top_15_df.empty and 'Ticker' in top_15_df.columns and not top_15_df['Ticker'].empty:
        first_top_ticker = top_15_df['Ticker'].iloc[0]
        if first_top_ticker in results_df['Ticker'].unique().tolist():
            default_ticker = first_top_ticker
    elif not results_df.empty and 'Ticker' in results_df.columns and not results_df['Ticker'].empty:
        default_ticker = results_df['Ticker'].iloc[0]

    options = sorted(results_df['Ticker'].unique().tolist()) # Ensure options are always sorted for consistency

    # --- DEFINE TABS FIRST - This makes them independent ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔬 Stock Dashboard & Financials", 
        "🛄 Factor Analysis", 
        "🔥 Financial Turbulence", 
        "📜 Theoretical Background", 
        "📄 Full Data Table"
    ])

    # --- Populate Tab 1: Stock Dashboard (This one is stock-specific) ---
    with tab1:
        if default_ticker and options: # Ensure there's a default ticker and options exist
            selected_ticker = st.sidebar.selectbox(
                "Select a Ticker for Deep Dive", 
                options=options, 
                index=options.index(default_ticker) if default_ticker in options else 0 
            )
            
            display_stock_dashboard(selected_ticker, results_df, winsorized_returns_dict, etf_histories)
            display_deep_dive_data(selected_ticker)
        else:
            st.warning("No stocks available in the portfolio to display a dashboard.")

    # --- Populate Tab 2: Factor Analysis (Context is portfolio-level) ---
    with tab2:
        st.subheader("Pure Factor Returns (Aggregated & Individual Horizons)")
        st.write("These are the underlying factor returns calculated across the entire stock universe.")
        if not active_rationale.empty:
            st.dataframe(active_rationale)
        else:
            st.info("Factor rationale not available (Factor Stability Analysis failed).")
        
        if stability_results: 
            time_horizons_display = {
                "1W": "Return_5d",   
                "2W": "Return_10d",  
                "1M": "Return_21d",  
                "3M": "Return_63d",  
                "6M": "Return_126d", 
                "12M": "Return_252d", 
            }
            for horizon_label, target_col_short_name in time_horizons_display.items():
                stability_df = stability_results.get(horizon_label) 
                if stability_df is not None and not stability_df.empty:
                    display_name = METRIC_NAME_MAP.get(target_col_short_name, target_col_short_name)
                    with st.expander(f"Details for {horizon_label} Horizon (Target: {display_name})"):
                        st.dataframe(stability_df)
                else:
                    st.info(f"No significant factors found for {horizon_label} horizon or data missing.")
        else:
            st.info("Factor stability analysis results are not available.")

    # --- Populate Tab 3: Financial Turbulence (This is a macro analysis) ---
    with tab3:
        if 'display_turbulence_and_regime_analysis' in globals():
            display_turbulence_and_regime_analysis()
        else:
            st.error("Error: The 'display_turbulence_and_regime_analysis' function is missing or not fully defined.") 

    # --- Populate Tab 4: Theoretical Background (This is general info) ---
    with tab4:
        if 'display_theoretical_background' in globals():
            display_theoretical_background()
        else:
            st.error("Error: The 'display_theoretical_background' function is missing or not fully defined.") 

    # --- Populate Tab 5: Full Data Table (This is universe-level data) ---
    with tab5:
        st.subheader("Full Processed Data Table")
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("No processed data available.")
if __name__ == "__main__":
    main()
