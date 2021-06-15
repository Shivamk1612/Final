import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
END = date.today().strftime("%Y-%m-%d")

st.title("Crude Oil Price Prediction App")

st.subheader("Crude Oil(CL=F)")

st.subheader("Brent Crude Oil(BZ=F)")

st.subheader("E-mini Crude Oil(QM=F)")

st.subheader("ProShares Ultra Bloomberg Crude Oil(UCO)")

st.subheader("WTI Houston Crude Oil(HCL=F)")

st.subheader("iPath Pure Beta Crude Oil(OIL)") 

st.subheader("Select dataset for prediction")
crude_oil = ("CL=F", "BZ=F", "QM=F", "UCO", "HCL=F")
selected_oil = st.selectbox("DATASET",crude_oil)

st.subheader("Select days of prediction")
n_days = st.slider("Days", 10, 30)
period = n_days

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data.....")
data = load_data(selected_oil)
data_load_state.text("Loading data.....DONE!")

st.subheader('Raw data')
st.write(data)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Raw data representation", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting

df_train1 = data[['Date', 'Close']]
df_train1 = df_train1.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train1)


future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast)

st.write("Forecast data representation")
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write("Forecast component")
fig2 = m.plot_components(forecast)
st.write(fig2)

