# streamlit_market_trend_app.py
# Streamlit UI for Market Trend Analyzer

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Market Trend Analyzer", layout="wide")

# --------------------
# MarketTrendAnalyzer
# --------------------
class MarketTrendAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
        self.analyze_trends()
    
    def prepare_data(self):
        """Clean and prepare the options data"""
        self.df.columns = [col.replace('\n', '').replace('/', '') for col in self.df.columns]
        # numeric columns
        numeric_cols = ['OI', 'OIChange', 'Volume', 'Chg in LTP', 'LTP', 'StrikePrice', 
                       'LTP.1', 'Chg in LTP.1', 'Volume.1', 'OIChange.1', 'OI.1']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', ''), errors='coerce')
        self.df = self.df.fillna(0)
        # create cleaner names
        if 'StrikePrice' in self.df.columns:
            self.df['Strike'] = self.df['StrikePrice']
        else:
            # try to find a numeric column that looks like strike
            for c in self.df.columns:
                if 'strike' in c.lower():
                    self.df['Strike'] = self.df[c]
                    break
        self.df['CALL_OI'] = self.df.get('OI', 0)
        self.df['PUT_OI'] = self.df.get('OI.1', 0)
        self.df['CALL_OI_Change'] = self.df.get('OIChange', 0)
        self.df['PUT_OI_Change'] = self.df.get('OIChange.1', 0)
        self.df['CALL_Volume'] = self.df.get('Volume', 0)
        self.df['PUT_Volume'] = self.df.get('Volume.1', 0)
        self.df['CALL_LTP'] = self.df.get('LTP', 0)
        self.df['PUT_LTP'] = self.df.get('LTP.1', 0)
        self.df['CALL_Price_Change'] = self.df.get('Chg in LTP', 0)
        self.df['PUT_Price_Change'] = self.df.get('Chg in LTP.1', 0)
    
    def analyze_trends(self):
        self.df = self.df[self.df['Strike'] > 0].copy() if 'Strike' in self.df.columns else self.df.copy()
        self.calculate_pcr_ratios()
        self.identify_support_resistance()
        self.analyze_skew()
        self.calculate_money_flow()
        self.determine_sentiment()
    
    def calculate_pcr_ratios(self):
        total_call_oi = self.df['CALL_OI'].sum()
        total_put_oi = self.df['PUT_OI'].sum()
        self.pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        total_call_vol = self.df['CALL_Volume'].sum()
        total_put_vol = self.df['PUT_Volume'].sum()
        self.pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        total_call_oi_change = self.df['CALL_OI_Change'].sum()
        total_put_oi_change = self.df['PUT_OI_Change'].sum()
        self.pcr_oi_change = total_put_oi_change / total_call_oi_change if total_call_oi_change != 0 else 0
    
    def identify_support_resistance(self):
        self.df['Total_OI'] = self.df['CALL_OI'] + self.df['PUT_OI']
        self.df['Total_Volume'] = self.df['CALL_Volume'] + self.df['PUT_Volume']
        if len(self.df) > 0:
            oi_threshold = self.df['Total_OI'].quantile(0.75)
            self.major_levels = self.df[self.df['Total_OI'] >= oi_threshold]['Strike'].tolist()
            self.max_pain = self.df.loc[self.df['Total_OI'].idxmax(), 'Strike'] if self.df['Total_OI'].max() > 0 else self.df['Strike'].median()
        else:
            self.major_levels = []
            self.max_pain = 0
        if len(self.df) > 0:
            self.df['Call_Put_Diff'] = abs(self.df['CALL_LTP'] - self.df['PUT_LTP'])
            atm_idx = self.df['Call_Put_Diff'].idxmin()
            self.atm_strike = self.df.loc[atm_idx, 'Strike']
        else:
            self.atm_strike = 0
    
    def analyze_skew(self):
        epsilon = 1e-6
        otm_calls = self.df[self.df['Strike'] > self.atm_strike]
        itm_calls = self.df[self.df['Strike'] < self.atm_strike]
        otm_call_activity = otm_calls['CALL_Volume'].sum() + otm_calls['CALL_OI_Change'].sum()
        itm_call_activity = itm_calls['CALL_Volume'].sum() + itm_calls['CALL_OI_Change'].sum()
        otm_puts = self.df[self.df['Strike'] < self.atm_strike]
        itm_puts = self.df[self.df['Strike'] > self.atm_strike]
        otm_put_activity = otm_puts['PUT_Volume'].sum() + otm_puts['PUT_OI_Change'].sum()
        itm_put_activity = itm_puts['PUT_Volume'].sum() + itm_puts['PUT_OI_Change'].sum()
        self.call_skew = otm_call_activity / (itm_call_activity + epsilon)
        self.put_skew = otm_put_activity / (itm_put_activity + epsilon)
    
    def calculate_money_flow(self):
        self.df['CALL_Money_Flow'] = self.df['CALL_Volume'] * self.df['CALL_Price_Change']
        self.df['PUT_Money_Flow'] = self.df['PUT_Volume'] * (-self.df['PUT_Price_Change'])
        self.total_call_flow = self.df['CALL_Money_Flow'].sum()
        self.total_put_flow = self.df['PUT_Money_Flow'].sum()
        self.net_money_flow = self.total_call_flow + self.total_put_flow
    
    def determine_sentiment(self):
        sentiment_score = 0
        signals = []
        if self.pcr_oi < 0.8:
            sentiment_score += 2
            signals.append("Bullish: Low Put-Call Ratio (OI)")
        elif self.pcr_oi > 1.2:
            sentiment_score -= 2
            signals.append("Bearish: High Put-Call Ratio (OI)")
        else:
            signals.append("Neutral: Balanced Put-Call Ratio (OI)")
        if self.pcr_volume < 0.8:
            sentiment_score += 1
            signals.append("Bullish: Low Put-Call Ratio (Volume)")
        elif self.pcr_volume > 1.2:
            sentiment_score -= 1
            signals.append("Bearish: High Put-Call Ratio (Volume)")
        if abs(self.pcr_oi_change) < 0.5:
            signals.append("Neutral: Balanced OI changes")
        elif self.pcr_oi_change < 0:
            sentiment_score += 1
            signals.append("Bullish: Calls being added faster than Puts")
        elif self.pcr_oi_change > 2:
            sentiment_score -= 1
            signals.append("Bearish: Puts being added faster than Calls")
        if self.net_money_flow > 0:
            sentiment_score += 1
            signals.append("Bullish: Positive net money flow")
        elif self.net_money_flow < 0:
            sentiment_score -= 1
            signals.append("Bearish: Negative net money flow")
        if self.call_skew > self.put_skew:
            sentiment_score += 1
            signals.append("Bullish: Higher call skew (upside positioning)")
        elif self.put_skew > self.call_skew:
            sentiment_score -= 1
            signals.append("Bearish: Higher put skew (downside protection)")
        sentiment_score = max(min(sentiment_score, 5), -5)
        if sentiment_score >= 3:
            self.sentiment = "STRONG BULLISH"
        elif sentiment_score >= 1:
            self.sentiment = "BULLISH"
        elif sentiment_score <= -3:
            self.sentiment = "STRONG BEARISH"
        elif sentiment_score <= -1:
            self.sentiment = "BEARISH"
        else:
            self.sentiment = "NEUTRAL"
        self.sentiment_score = sentiment_score
        self.signals = signals
    
    def create_trend_dashboard_figure(self, figsize=(14, 10)):
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(f'Market Trend Analysis - Overall Sentiment: {self.sentiment}', fontsize=14)
        strikes = self.df['Strike']
        # Open Interest Profile
        ax1 = axes[0, 0]
        ax1.bar(strikes - 2, self.df['CALL_OI'], width=4, alpha=0.7, label='CALL OI')
        ax1.bar(strikes + 2, self.df['PUT_OI'], width=4, alpha=0.7, label='PUT OI')
        ax1.set_title('Open Interest Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Volume Profile
        ax2 = axes[0, 1]
        ax2.bar(strikes - 2, self.df['CALL_Volume'], width=4, alpha=0.7, label='CALL Vol')
        ax2.bar(strikes + 2, self.df['PUT_Volume'], width=4, alpha=0.7, label='PUT Vol')
        ax2.set_title('Volume Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # OI Changes
        ax3 = axes[0, 2]
        ax3.bar(strikes - 2, self.df['CALL_OI_Change'], width=4, alpha=0.7, label='CALL Change')
        ax3.bar(strikes + 2, self.df['PUT_OI_Change'], width=4, alpha=0.7, label='PUT Change')
        ax3.set_title('OI Changes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        # PCR
        ax4 = axes[1, 0]
        pcr_data = [self.pcr_oi, self.pcr_volume, abs(self.pcr_oi_change)]
        pcr_labels = ['PCR (OI)', 'PCR (Volume)', 'PCR (Change)']
        bars = ax4.bar(pcr_labels, pcr_data, alpha=0.7)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.7)
        ax4.set_title('Put-Call Ratios')
        for bar, value in zip(bars, pcr_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{value:.2f}', ha='center')
        # Money Flow
        ax5 = axes[1, 1]
        flow_data = [self.total_call_flow, self.total_put_flow, self.net_money_flow]
        flow_labels = ['CALL Flow', 'PUT Flow', 'Net Flow']
        ax5.bar(flow_labels, flow_data, alpha=0.7)
        ax5.set_title('Money Flow Analysis')
        ax5.grid(True, alpha=0.3)
        # Support/Resistance
        ax6 = axes[1, 2]
        ax6.plot(strikes, self.df['Total_OI'], marker='o', linewidth=2)
        for level in getattr(self, 'major_levels', []):
            ax6.axvline(level, color='red', alpha=0.5, linestyle=':')
        ax6.set_title('Support/Resistance Levels')
        ax6.grid(True, alpha=0.3)
        # Price Change Distribution
        ax7 = axes[2, 0]
        call_changes = self.df['CALL_Price_Change'].replace(0, np.nan).dropna()
        put_changes = self.df['PUT_Price_Change'].replace(0, np.nan).dropna()
        if len(call_changes) > 0:
            ax7.hist(call_changes, bins=10, alpha=0.6, label='CALL Changes')
        if len(put_changes) > 0:
            ax7.hist(put_changes, bins=10, alpha=0.6, label='PUT Changes')
        ax7.set_title('Price Change Distribution')
        ax7.legend()
        # Sentiment & Summary
        ax8 = axes[2, 1]
        ax8.axis('off')
        summary_text = f"""
PCR (OI): {self.pcr_oi:.2f}\nPCR (Vol): {self.pcr_volume:.2f}\nATM Strike: {getattr(self,'atm_strike','N/A')}\nMax Pain: {getattr(self,'max_pain','N/A')}\nNet Money Flow: {self.net_money_flow:.0f}\nSentiment: {self.sentiment} ({self.sentiment_score})\n"""
        ax8.text(0.01, 0.5, summary_text, fontsize=10, fontfamily='monospace')
        ax9 = axes[2, 2]
        ax9.axis('off')
        plt.tight_layout()
        return fig

    def generate_trend_report(self):
        return {
            'sentiment': self.sentiment,
            'sentiment_score': self.sentiment_score,
            'pcr_oi': self.pcr_oi,
            'pcr_volume': self.pcr_volume,
            'atm_strike': getattr(self, 'atm_strike', None),
            'max_pain': getattr(self, 'max_pain', None),
            'net_money_flow': self.net_money_flow,
            'signals': self.signals
        }

# --------------------
# Helper functions
# --------------------
@st.cache_data(ttl=60)
def load_gsheet(gsheet_id: str, gid: str = '0'):
    url = f"https://docs.google.com/spreadsheets/d/{gsheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(url)
    return df

# --------------------
# Streamlit UI
# --------------------
st.sidebar.header('Data Source')
gsheet_id_input = st.sidebar.text_input('Google Sheet ID', value='1vjBxHJzoH5qPIFtBNbuVt6kaTSXv6yrzywVl80RufhQ')
gid_input = st.sidebar.text_input('GID (sheet tab id)', value='0')
refresh = st.sidebar.button('Refresh Data')

st.title('📈 Market Trend Analyzer (Streamlit)')
st.markdown('A simple UI to analyze options market data and visualize key metrics.')

# Load data
try:
    if refresh:
        st.cache_data.clear()
    df = load_gsheet(gsheet_id_input, gid_input)
    st.success(f'Data loaded: {df.shape[0]} rows, {df.shape[1]} cols')
except Exception as e:
    st.error(f'Error loading sheet: {e}')
    st.stop()

# Show raw data toggle
if st.checkbox('Show raw data'):
    st.dataframe(df)

# Small data preview
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.metric('Rows', df.shape[0])
with col2:
    st.metric('Columns', df.shape[1])
with col3:
    st.metric('GSheet ID', gsheet_id_input)

# Run analysis
if st.button('Run Analysis'):
    with st.spinner('Analyzing...'):
        analyzer = MarketTrendAnalyzer(df)
        report = analyzer.generate_trend_report()
        st.subheader('Summary')
        st.write(report)
        # show signals
        st.subheader('Signals')
        for s in analyzer.signals:
            st.write('- ' + s)
        # show figure
        fig = analyzer.create_trend_dashboard_figure(figsize=(12, 9))
        st.pyplot(fig)
        # Download report
        st.download_button('Download JSON report', data=pd.Series(report).to_json(), file_name='trend_report.json')

st.markdown('---')
st.caption('Built with ❤️ — modify the script to customize visuals or add more metrics.')
