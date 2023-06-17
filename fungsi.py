import streamlit as st
from PIL import Image
import plotly.graph_objs as go
import statistics
from bdes import formatrupiah
def convert_df(df):
    return df.to_csv().encode('utf-8')

def foto(path, ukuran):
    foto=Image.open(path)
    foto=foto.resize((ukuran, ukuran))
    st.image(foto)

def kosumsi(path,tahun,jumlah,pertumbuhan,warna):
    img1=Image.open(path)
    img1=img1.resize((50,50))
    st.image(img1)
    st.markdown(f"<h1 style='text-align: right; padding-bottom:10px; font-size:20px; padding-top:0px'>TAHUN {tahun}</h1>", unsafe_allow_html=True) 
    st.markdown(f"<h1 style='text-align: center; padding-bottom:5px; font-size:27px; padding-top:0px; color:#6A51BC'>{jumlah}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:15px; padding-top:0px; color:{warna}'>{pertumbuhan}</h1>", unsafe_allow_html=True)

def visualisasitest(data, data2, data3, ukuran):
    data_1b = go.Scatter(x=data['Tanggal'], y=data['Harga'], name="Data Aktual", mode="lines")
    data_2b = go.Scatter(x=data2['Tanggal'], y=data2['prediksi'], name="Data Training", mode="lines")
    data_3b= go.Scatter(x=data3['Tanggal'], y=data3['prediksi'], name="Data Testing", mode="lines")

    figtesting = go.Figure([data_1b, data_2b, data_3b]) 
    figtesting.update_layout(xaxis_title = 'Tanggal', yaxis_title = 'Harga', width = ukuran)
    figtesting.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
    st.write(figtesting, unsafe_allow_html=True)

def visualisasi_prediksi(data, data1, data2, data3):
    data_1 = go.Scatter(x=data['Tanggal'], y=data['Harga'], name="Data Aktual", mode="lines")
    data_2 = go.Scatter(x=data1['Tanggal'], y=data1['prediksi'], name="Data Training", mode="lines")
    data_3= go.Scatter(x=data2['Tanggal'], y=data2['prediksi'], name="Data Testing", mode="lines")
    data_4= go.Scatter(x=data3['Tanggal'], y=data3['hasilprediksi'], name="Periode Depan", mode="lines")

    figtesting = go.Figure([data_1, data_2, data_3, data_4])
    figtesting.update_layout(xaxis_title = 'Tanggal',yaxis_title = 'Harga',width = 620)
    figtesting.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
    st.write(figtesting, unsafe_allow_html=True)

def statistika(data):
    st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px'>Statistika Deskriptif</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:25px;font-size: 33px'>{formatrupiah(max(data))}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; padding-bottom:0px; font-size:18px; padding-top:0px'>Harga Tertinggi</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:10px;font-size: 33px'>{formatrupiah(min(data))}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; padding-bottom:0px; font-size:18px; padding-top:0px'>Harga Terendah</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:10px;font-size: 33px'>{formatrupiah(round(statistics.mean(data)))}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: right; padding-bottom:30px; font-size:18px; padding-top:0px'>Rata - rata Harga</h1>", unsafe_allow_html=True)