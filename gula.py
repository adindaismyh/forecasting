import streamlit as st
from streamlit_option_menu import option_menu
import time
import pandas as pd
import numpy as np
import plotly.express as px
import statistics
from bdes import bdesmodel, pred_periodedepan, tanggal_kedepan, mape,golden_section, formatrupiah
from bwema import golden_section2,bwema, nilaiBt
from fungsi import convert_df, foto,kosumsi, visualisasitest, visualisasi_prediksi
import plotly.graph_objs as go

def gula():
    with open("style.css") as gaya:
        st.markdown(f'<style>{gaya.read()}</style>', unsafe_allow_html=True)

    col1,col2 = st.columns([4,2])
    with col1:
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:40px; padding-top:0px'>Hello, Everyone</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:18px; padding-top:10px'>Halaman dashboard komoditi gula pasir</h1>", unsafe_allow_html=True)
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:45px'>Dataset</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:0px'>Berikut merupakan data set yang digunakan untuk membuat dan melatih model forecasting. Data yang digunakan merupakan data harga gula pasir mulai tanggal 1 Januari 2020 hingga 31 Mei 2023</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:13px; padding_top: 0px;color: #DE4D86'>Sumber : Web resmi PHPS Nasional bi.go.id</h1>", unsafe_allow_html=True)
        st.divider()
        gula = pd.read_csv("DATA/DATA GULA.csv")
        gula = gula.assign(index = np.arange(len(gula)))
        gula['Tanggal'] = pd.to_datetime(gula['Tanggal'], infer_datetime_format=True)
        d1,d2 = st.columns([2,3])
        with d1:
            mindate = gula['Tanggal'][0]
            maxdate= gula['Tanggal'][845]
            st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)
            start_date = st.date_input(label="Mulai Tanggal",min_value=mindate,value=mindate)
            st.write(start_date)
            end_date = st.date_input(label="Hingga Tanggal",max_value=maxdate,value=maxdate)
            st.write(end_date)
            start_date = pd.to_datetime(start_date, infer_datetime_format=True)
            end_date = pd.to_datetime(end_date, infer_datetime_format=True)
                
            hasil=gula[(gula['Tanggal']>=start_date) &(gula['Tanggal']<=end_date) ]
            
        with d2:
            st.dataframe(hasil, use_container_width=True)
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:45px'>Visualisasi Data</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:0px'>Plot data harga gula pasir mulai bulan januari 2020 hingga maret 2023</h1>", unsafe_allow_html=True)
        
        v1,v2= st.columns([4,2])
        with v1:
            st.markdown(f"<h1 style='padding-top:10px'></h1>", unsafe_allow_html=True)
            tab1, tab2= st.tabs(["Diagram Line","Analisa"])
            with tab1:
                fig = px.line(hasil, x="Tanggal", y="Harga") 
                fig.update_layout(width=500,height=300,margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
                st.write(fig)
            with tab2:
                st.markdown(f"""<h1 style='text-align: justify; padding-right:25px; font-size:20px; padding-top:10px;padding-bottom:0px'>pada bulan Juni 2022 hingga Januari 2021 harga gula pasir mempunyai kecenderungan trend turun setelah mengalami kenaikan pada bulan sebelumnya.
                 Menuju ke periode akhir harga gula pasir mengalami perubahan harga naik turun yang tidak cepat. </h1>""", unsafe_allow_html=True)

        with v2:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px'>Statistika Deskriptif</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:25px;font-size: 33px'>{formatrupiah(max(hasil['Harga']))}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; padding-bottom:0px; font-size:18px; padding-top:0px'>Harga Tertinggi</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:10px;font-size: 33px'>{formatrupiah(min(hasil['Harga']))}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; padding-bottom:0px; font-size:18px; padding-top:0px'>Harga Terendah</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; color: #DE4D86;padding-bottom:0px;padding-top:10px;font-size: 33px'>{formatrupiah(round(statistics.mean(hasil['Harga'])))}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; padding-bottom:30px; font-size:18px; padding-top:0px'>Rata - rata Harga</h1>", unsafe_allow_html=True)
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px'>Metode Forecasting Komoditi Sembako</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:13px; padding-top:8px; color: #674FBB'>Pilih metode di bawah ini untuk melihat hasil prediksi</h1>", unsafe_allow_html=True)
        st.divider()
        genre = st.radio( "OPTION", ('Brown’s Double Exponential Smoothing', 'Brown’s Weighted Exponential Moving Average', 'Perbandingan Metode'))
        st.markdown(f"<h1 style='text-align: left; padding-bottom:30px'></h1>", unsafe_allow_html=True)

    with col2:
        waktu = time.ctime()
        st.markdown(f"<h1 style='text-align: right; padding-bottom:30px; font-size:18px; padding-top:0px'>⏲️ {waktu}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:40px; padding-top:0px'>KOMODITI</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px'>Gula Pasir</h1>", unsafe_allow_html=True)
        d1,d2,d3 = st.columns(3)
        with d1:
            foto('gambar/prediksi.png',60)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:15px; font-size:12px; padding-top:0px'>Prediksi</h1>", unsafe_allow_html=True)
        with d2:
            foto('gambar/analisis.png',60)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:15px; font-size:12px; padding-top:0px'>Statistika</h1>", unsafe_allow_html=True)
        with d3:
            foto('gambar/plot.png',60)
            st.markdown(f"<h1 style='text-align: center; padding-bottom:15px; font-size:12px; padding-top:0px'>Visualisasi</h1>", unsafe_allow_html=True)

        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:20px; padding-top:0px'>Rata-Rata Konsumsi per Kapita Gula Pasir (kg/kapita/tahun)</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:13px; padding-top:8px; color: #674FBB'>Sumber : Statistik Konsumsi Pangan Tahun 2022</h1>", unsafe_allow_html=True)
  
        st.divider()
        col3,col4 = st.columns(2)
        with col3:
            kosumsi('gambar/gambar1.png',2019,'6.634 kg','⬇️193 kg', "red")
        with col4:
            kosumsi('gambar/gambar2.png',2020,'6.539 kg','⬇️95 kg', "red" )
        col5,col6 = st.columns(2)
        with col5:
            kosumsi('gambar/gambar3.png',2021,'6.677 kg','⬆️138 kg', "green" )
        with col6:
            kosumsi('gambar/gambar3.png',2022,'6.319 kg','⬇️358 kg', "red" )
        
        data = pd.DataFrame({
            'Tahun' :["2020","2021","2022"],
            'Produksi':[7289,7384,6747]
        })
        fig_produksi = px.bar(data, x='Tahun', y='Produksi', color="Tahun")
        
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:20px; padding-top:40px'>Produksi Gula Pasir di Indonesia (ton)</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:13px; padding-top:8px; color: #674FBB'>Sumber : Buku Statistik Konsumsi 2022</h1>", unsafe_allow_html=True)
        st.divider()
        fig_produksi.update_layout(width=360,height=300,margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="#D8D9F9")
        st.write(fig_produksi)
    
    Datates = 118
    traingula = gula.iloc[:-Datates]
    testgula = gula.iloc[-Datates:]
    test_gula = np.zeros(len(testgula)) 
    kons= np.zeros(len(traingula))
    slope =np.zeros(len(traingula))

    # membuat model
    opt = golden_section(traingula, testgula, test_gula,0, 1, slope, kons,len(testgula))
    prediksi = pd.DataFrame(bdesmodel(traingula, opt , slope, kons, test_gula,len(testgula)))
    training ={'Tanggal': traingula['Tanggal'],'data_aktual': traingula['Harga'],'prediksi': prediksi[0]}
    training= pd.DataFrame(training)
    training['Tanggal'] = pd.to_datetime(training['Tanggal'], infer_datetime_format=True)

    # uji coba model
    testing ={'Tanggal': testgula['Tanggal'],'data_aktual': testgula['Harga'],'prediksi':test_gula}
    testing= pd.DataFrame(testing)
    testing['Tanggal'] = pd.to_datetime(testing['Tanggal'], infer_datetime_format=True)
    

    ## PREDIKSI BDES

    if genre == 'Brown’s Double Exponential Smoothing':
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; font-style: italic'>Brown's Double Exponential Smoothing</h1>", unsafe_allow_html=True)
        st.markdown("_Brown’s Double Exponential Smoothing_  (B-DES) adalah metode peramalan yang dapat digunakan untuk data yang mempunyai pola _trend_, dimana keakuratannya bergantung dengan nilai parameter pemulusan ∝ yang bisa memperbaiki trend")
        tab5, tab6, tab7= st.tabs(["Prediksi Data Training","Prediksi Data Testing", "Prediksi Periode Depan"])
        with tab5:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Data Training</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Tabel Hasil prediksi data training mulai tanggal 1 Januari 2020 hingga 16 Desember 2022 </h1>", unsafe_allow_html=True)

            d1,d2 = st.columns([2,3])
            with d1:
                mindate1 = training['Tanggal'][0]
                maxdate1= training['Tanggal'][727]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date1 = st.date_input(label="Tanggal Awal",min_value=mindate1,value=mindate1)
                st.write(start_date1)
                end_date1 = st.date_input(label="Tanggal Akhir",max_value=maxdate1,value=maxdate1)
                st.write(end_date1, use_container_width=True)
                start_date1 = pd.to_datetime(start_date1, infer_datetime_format=True)
                end_date1 = pd.to_datetime(end_date1, infer_datetime_format=True)
                        
                hasiltraining=training[(training['Tanggal']>=start_date1) &(training['Tanggal']<=end_date1) ]
                    
            with d2:
                st.dataframe(hasiltraining, use_container_width=True)
            
            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>STATISTIKA DESKRIPTIF</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan statistika deskriptif dari {start_date1} hingga {end_date1}</h1>", unsafe_allow_html=True)
            s1,s2,s3 = st.columns(3)
            with s1:  
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar5.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Tertinggi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:38px ; padding-top:0px'>{formatrupiah(round(max(hasiltraining['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s2:
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar6.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Terendah</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(min(hasiltraining['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s3:
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar7.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Rata - Rata Harga Prediksi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(statistics.mean(hasiltraining['prediksi'])))}</h1>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Visualisasi Data</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Visualisasi hasil prediksi data training harga gula pasir dari model B- DES yang telah dibuat mulai tanggal {start_date1} hingga tanggal {end_date1}</h1>", unsafe_allow_html=True)

            v3,v4 = st.columns([5,3])
            with v3:
    
                fig = px.line(hasiltraining, x="Tanggal", y=["data_aktual", "prediksi"]) 
                fig.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
                st.write(fig, unsafe_allow_html=True)
            with v4:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:28px; padding-top:50px; color:#6A51BC'>ANALISA PLOT</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:18px; padding-top:0px'>Plot Hasil Data Training 1 Januari 2020 hingga 16 Desember 2022</h1>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"""<h3 style='text-align: justify; padding-right:25px; font-size:18px;padding-bottom:30px; font-family: helvetica'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga gula pasir. Grafik warna biru merupakan data aktual, dan grafik warna biru muda merupakan hasil prediksi
                Nilai fitted value dari data training yang ditampilkan pada plot tersebut dapat disimpulkan bahwa hasil prediksi mampu mengikuti pola data aktual secara keseluruhan dengan baik, karena pola data prediksi terlihat sama dengan pola data aktual.</h3>""", unsafe_allow_html=True)
        with tab6:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Data Testing</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Tabel Hasil prediksi data testing mulai tanggal 19 Desember 2022 hingga 31 Mei 2023 </h1>", unsafe_allow_html=True)

            d1,d2 = st.columns([2,3])
            with d1:
                mindate2 = testing['Tanggal'][728]
                maxdate2= testing['Tanggal'][845]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date2 = st.date_input(label="Pilih Tanggal Awal",min_value=mindate2,value=mindate2)
                st.write(start_date2)
                end_date2 = st.date_input(label="Pilih Tanggal Akhir",max_value=maxdate2,value=maxdate2)
                st.write(end_date2, use_container_width=True)
                start_date2 = pd.to_datetime(start_date2, infer_datetime_format=True)
                end_date2 = pd.to_datetime(end_date2, infer_datetime_format=True)   
                        
                hasiltesting=testing[(testing['Tanggal']>=start_date2) &(testing['Tanggal']<=end_date2) ]
                    
            with d2:
                st.dataframe(hasiltesting, use_container_width=True)
            
            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>STATISTIKA DESKRIPTIF</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan statistika deskriptif dari {start_date2} hingga {end_date2}</h1>", unsafe_allow_html=True)
        
            s1,s2,s3 = st.columns(3)
            with s1:  
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar5.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Tertinggi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:38px ; padding-top:0px'>{formatrupiah(round(max(hasiltesting['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s2:
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar6.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Terendah</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(min(hasiltesting['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s3:
                sd1,sd2 = st.columns([2,3])
                with sd1:
                    foto('gambar/gambar7.png', 100)
                with sd2:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Rata - Rata Harga Prediksi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(statistics.mean(hasiltesting['prediksi'])))}</h1>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Visualisasi Data</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Visualisasi hasil prediksi data testing harga gula pasir dari model B- DES yang telah dibuat mulai tanggal {start_date2} hingga tanggal {end_date2}</h1>", unsafe_allow_html=True)
            v3, v4, v5 =st.columns([4,2,1])
            with v3:
                visualisasitest(gula, training, testing, 750)
                
            with v4:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:28px; color:#6A51BC'>ANALISA PLOT</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:16px; padding-top:0px'>Analisa Hasil Data Testing 19 Desember 2022 hingga 31 Mei 2023</h1>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"""<h3 style='text-align: justify; font-size:15px; font-family: helvetica'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga gula pasir. Grafik warna biru tua merupakan data aktual, dan grafik warna biru muda merupakan hasil prediksi
                dan grafik warna merah merupakah hasil prediksi data testing. Berdasarkan gambar disamping dapat disimpulkan bahwa hasil prediksi dari data uji coba atau testing mulai
                tanggal 19 Desember 2022 hingga 31 Mei 2023 menunjukkan adanya kecenderungan trend naik yang sangat lambat</h3>""", unsafe_allow_html=True)
            
            with v5:
                st.info('MAPE MODEL',icon="📌")
                st.markdown(f"<h1 style='text-align: center; color: #DE4D86; padding-bottom:10px; font-size:25px'>{round(mape(training['data_aktual'], training['prediksi']),2)} %</h1>", unsafe_allow_html=True)
                st.info('MAPE TESTING',icon="📌")
                st.markdown(f"<h1 style='text-align: center; color: #DE4D86; padding-bottom:5px; font-size:25px'>{round(mape(testing['data_aktual'], testing['prediksi']),2)} %</h1>", unsafe_allow_html=True)

                st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:18px; padding-top: 13px;color:#6A51BC'>KETERANGAN :</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: justify; padding-bottom:0px; font-size: 15px; padding-top:0px'>Prediksi harga gula pasir di Jawa Timur menggunakan metode B-DES menghasilkan MAPE dibawah 10% sehingga dapat disimpulkan bahwa mempunyai peramalan yang sangat baik atau akurat yang tinggi</h1>", unsafe_allow_html=True)

        with tab7:
            
            st.markdown(f"<h1 style='text-align: left; padding-bottom:15px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Periode Ke Depan</h1>", unsafe_allow_html=True)
            d1,d2 = st.columns([2,3])
            with d1:
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Masukkan besar prediksi periode ke depan yang diinginkan dalam satuan hari</h1>", unsafe_allow_html=True)            
                title = st.text_input('Besar Prediksi', '365')
                title = int(title)
                indexing = np.arange((len(gula)+1),(len(gula)+(title +1)))

                periodedepan =np.zeros(title)
                periodedepan = pd.DataFrame(periodedepan)
                periodedepan = periodedepan.rename(columns = {0: "Tanggal"}) 


                tanggal_kedepan(testgula, periodedepan)
                prediksi = pred_periodedepan(kons, slope, title, testgula)
                periodedepan = periodedepan.assign(hasilprediksi = prediksi)
                periodedepan = periodedepan.assign(index= indexing)
                periodedepan = periodedepan.set_index('index')


                mindate3 = periodedepan['Tanggal'][len(gula)+1]
                maxdate3= periodedepan['Tanggal'][len(gula)+title]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date3 = st.date_input(label="Tanggal Awal",min_value=mindate3,value=mindate3)
                st.write(start_date3)
                end_date3 = st.date_input(label="Tanggal Akhir",max_value=maxdate3,value=maxdate3)
                st.write(end_date3, use_container_width=True)
                start_date3 = pd.to_datetime(start_date3, infer_datetime_format=True)
                end_date3 = pd.to_datetime(end_date3, infer_datetime_format=True)

                hasilprediksi=periodedepan[(periodedepan['Tanggal']>=start_date3) &(periodedepan['Tanggal']<=end_date3) ]
                st.markdown(f"<h1 style='padding-bottom:5px'></h1>", unsafe_allow_html=True)
                        
                    
            with d2:
                st.markdown(f"<h1 style='text-align: center; padding-right:25px; font-size:17px; padding-top:10px;padding-bottom:30px;color: #6C5CD7'>Tabel Hasil prediksi selama {title} hari ke depan yaitu dari tanggal {periodedepan['Tanggal'][len(gula)+1]} hingga {periodedepan['Tanggal'][len(gula)+title]} </h1>", unsafe_allow_html=True)
                
                st.dataframe(hasilprediksi, use_container_width=True)
                csv = convert_df(hasilprediksi)
                st.markdown(f"<h1 style='text-align: left; color: #6C5CD7 ; padding-bottom:10px; font-size:15px ; padding-top:0px'>Simpan data hasil prediksi</h1>", unsafe_allow_html=True)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='gulapasir_bdes.csv',
                    mime='text/csv',
                )
            
            sd8,sd9= st.columns(2)
            with sd8:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:25px; padding-top:10px;  color:#6A51BC'>Visualisasi Periode Depan</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: justify; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:0px'>Plot data Prediksi {title} hari ke depan bersama data aktual harga gula pasir di Jawa Timur, prediksi data training, dan testing secara keseluruhan</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:12px; padding-top:10px;  color:red'>Gambar 1</h1>", unsafe_allow_html=True)
                st.divider()
                visualisasi_prediksi(gula, training, testing, periodedepan)

            with sd9:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:25px; padding-top:10px; color:#6A51BC'>Visualisasi Periode Depan Interaktif</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: justify; padding-right:25px; font-size:15px; padding-top:10px'>Plot data prediksi periode depan sesuai dengan tanggal yang dipilih pada tabel di atas</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; font-size:12px; padding-top:10px;padding-bottom:5px;  color:red'>Gambar 2</h1>", unsafe_allow_html=True)
                st.divider()
                fig_depan = px.line(hasilprediksi, x="Tanggal", y="hasilprediksi") 
                fig_depan.update_layout(width=490,margin=dict(l=6,r=1,b=1,t=1),paper_bgcolor="white", )
                st.write(fig_depan)

            col7, col8 = st.columns([3,4])
            with col7:
                st.markdown(f"<h1 style='text-align: left;font-size:25px; color:#6A51BC; padding-top:0px'>Analisa Hasil Prediksi Selama {title} hari ke Depan</h1>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"""<h3 style='text-align: justify; font-size:15px; font-family: helvetica'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga gula pasir. Grafik warna biru tua merupakan data aktual, dan grafik warna biru muda merupakan hasil prediksi
                ,grafik warna merah merupakah hasil prediksi data testing, dan grafik warna orange merupakan hasil prediksi periode depan. Berdasarkan Gambar 1 dapat disimpulkan bahwa hasil prediksi selama {title} hari ke depan yaitu dari tanggal {periodedepan['Tanggal'][len(gula)+1]} hingga {periodedepan['Tanggal'][len(gula)+title]}
                menunjukkan bahwa harga gula pasir di Jawa Timur mempunyai kecenderungan trend naik yang sangat lambat</h3>""", unsafe_allow_html=True)
            with col8:
                
                st.markdown(f"<h1 style='text-align: center; padding-bottom:0px; font-size:33px; padding-top:10px; color:#6A51BC'>Statistika Deskriptif</h1>", unsafe_allow_html=True)
                st.divider()
                col9,col10,col11=st.columns(3)
                with col9:
                    st.info('HARGA TERENDAH',icon="📌")
                    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{formatrupiah(min(hasilprediksi['hasilprediksi']))}</h2>", unsafe_allow_html=True)
                with col10:
                    st.info('HARGA TERTINGGI',icon="📌")
                    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{formatrupiah(max(hasilprediksi['hasilprediksi']))}</h2>", unsafe_allow_html=True)
                with col11:
                    st.info('RATA-RATA HARGA',icon="📌")
                    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{formatrupiah(round(statistics.mean(hasilprediksi['hasilprediksi'])))}</h2>", unsafe_allow_html=True)


# PREDIKSI BWEMA 

    traingula2 = gula.iloc[:-Datates]
    testgula2 = gula.iloc[-Datates:]
    test_gula = np.zeros(len(testgula2)) 
    kons= np.zeros(len(traingula2))
    slope =np.zeros(len(traingula2))
    Bt=[]
    weights = [ 22,21,20,19,18,17,16,15,14,13,12,10,9,8,7,6,5,4,3,2,1]

    nilaiBt(weights, traingula, Bt)
    Bt = pd.DataFrame(Bt)
    Bt = Bt.rename(columns = {0: "Bt"})

    # membuat model

    opt2 = golden_section2(traingula2, testgula2, test_gula,0, 1, slope, kons, Bt)
    prediksi2 = pd.DataFrame(bwema(traingula2, opt2 , slope, kons, test_gula, Bt))
    training2 ={'Tanggal': traingula2['Tanggal'],'data_aktual': traingula2['Harga'],'prediksi': prediksi2[0]}
    training2= pd.DataFrame(training2)
    training2['Tanggal'] = pd.to_datetime(training2['Tanggal'], infer_datetime_format=True)

    # uji coba model
    testing2 ={'Tanggal': testgula2['Tanggal'],'data_aktual': testgula2['Harga'],'prediksi':test_gula}
    testing2= pd.DataFrame(testing2)
    testing2['Tanggal'] = pd.to_datetime(testing2['Tanggal'], infer_datetime_format=True)

    if genre == 'Brown’s Weighted Exponential Moving Average':
        st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:45px; font-style: italic'>Brown's Weighted Exponential Moving Average</h1>", unsafe_allow_html=True)
        st.markdown("_Brown’s Weighted Exponential Moving Average_ adalah gabungan antara metode WMA (_Weighted Moving Average_) dan B-DES dimana metode tersebut dapat digunakan untuk meramalkan data deret waktu masa depan yang mempunyai pola trend ")

        tab5_b, tab6_b, tab7_b= st.tabs(["Prediksi Data Training","Prediksi Data Testing", "Prediksi Periode Depan"])
        with tab5_b:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Data Training</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Tabel Hasil prediksi data training mulai tanggal 1 Januari 2020 hingga 16 Desember 2022 </h1>", unsafe_allow_html=True)

            d1_b,d2_b = st.columns([2,3])
            with d1_b:
                mindate1_b = training2['Tanggal'][0]
                maxdate1_b= training2['Tanggal'][727]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date1_b = st.date_input(label="Tanggal Awal :",min_value=mindate1_b,value=mindate1_b)
                st.write(start_date1_b)
                end_date1_b = st.date_input(label="Tanggal Akhir :",max_value=maxdate1_b,value=maxdate1_b)
                st.write(end_date1_b, use_container_width=True)
                start_date1_b = pd.to_datetime(start_date1_b, infer_datetime_format=True)
                end_date1_b = pd.to_datetime(end_date1_b, infer_datetime_format=True)
                        
                hasiltraining2=training2[(training2['Tanggal']>=start_date1_b) &(training2['Tanggal']<=end_date1_b) ]
                    
            with d2_b:
                st.dataframe(hasiltraining2, use_container_width=True)
            
            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>STATISTIKA DESKRIPTIF</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan statistika deskriptif dari {start_date1_b} hingga {end_date1_b}</h1>", unsafe_allow_html=True)
            s1_b,s2_b,s3_b = st.columns(3)
            with s1_b:  
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar5.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Tertinggi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:38px ; padding-top:0px'>{formatrupiah(round(max(hasiltraining2['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s2_b:
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar6.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Terendah</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(min(hasiltraining2['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s3_b:
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar7.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Rata - Rata Harga Prediksi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(statistics.mean(hasiltraining2['prediksi'])))}</h1>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Visualisasi Data</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Visualisasi hasil prediksi data training harga gula pasir dari model B- DES yang telah dibuat mulai tanggal {start_date1_b} hingga tanggal {end_date1_b}</h1>", unsafe_allow_html=True)

            v3_b,v4_b = st.columns([5,3])
            with v3_b:
    
                fig2 = px.line(hasiltraining2, x="Tanggal", y=["data_aktual", "prediksi"]) 
                fig2.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
                st.write(fig2, unsafe_allow_html=True)
            with v4_b:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:28px; padding-top:50px; color:#6A51BC'>ANALISA PLOT</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:18px; padding-top:0px'>Plot Hasil Data Training 1 Januari 2020 hingga 16 Desember 2022</h1>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"""<h3 style='text-align: justify; padding-right:25px; font-size:18px;padding-bottom:30px; font-family: helvetica'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga gula pasir. Grafik warna biru merupakan data aktual, dan grafik warna biru muda merupakan hasil prediksi
                Nilai fitted value dari data training yang ditampilkan pada plot tersebut dapat disimpulkan bahwa hasil prediksi mampu mengikuti pola data aktual secara keseluruhan dengan baik, karena pola data prediksi terlihat sama dengan pola data aktual.</h3>""", unsafe_allow_html=True)
        with tab6_b:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Data Testing</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Tabel Hasil prediksi data testing mulai tanggal 19 Desember 2022 hingga 31 Mei 2023 </h1>", unsafe_allow_html=True)

            d1_b,d2_b = st.columns([2,3])
            with d1_b:
                mindate2_b = testing2['Tanggal'][728]
                maxdate2_b= testing2['Tanggal'][845]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date2_b = st.date_input(label="Pilih Tanggal Awal :",min_value=mindate2_b,value=mindate2_b)
                st.write(start_date2_b)
                end_date2_b = st.date_input(label="Pilih Tanggal Akhir :",max_value=maxdate2_b,value=maxdate2_b)
                st.write(end_date2_b, use_container_width=True)
                start_date2_b = pd.to_datetime(start_date2_b, infer_datetime_format=True)
                end_date2_b = pd.to_datetime(end_date2_b, infer_datetime_format=True)   
                        
                hasiltesting2=testing2[(testing2['Tanggal']>=start_date2_b) &(testing2['Tanggal']<=end_date2_b) ]
                csv2 = convert_df(hasiltesting2)
                st.markdown(f"<h1 style='text-align: left; color: #6C5CD7 ; padding-bottom:10px; font-size:15px ; padding-top:0px'>Simpan data hasil prediksi</h1>", unsafe_allow_html=True)
                st.download_button(
                    label="Download data as CSV",
                    data= csv2,
                    file_name='gulapasir_bwema.csv',
                    mime='text/csv',
                )
                st.markdown(f"<h1 style='padding-bottom:15px'></h1>", unsafe_allow_html=True)
                    
            with d2_b:
                st.dataframe(hasiltesting2, use_container_width=True)
            
            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>STATISTIKA DESKRIPTIF</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan statistika deskriptif dari {start_date2_b} hingga {end_date2_b}</h1>", unsafe_allow_html=True)
        
            s1_b,s2_b,s3_b = st.columns(3)
            with s1_b:  
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar5.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Tertinggi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:38px ; padding-top:0px'>{formatrupiah(round(max(hasiltesting2['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s2_b:
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar6.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Harga Prediksi Terendah</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(min(hasiltesting2['prediksi'])))}</h1>", unsafe_allow_html=True)
            with s3_b:
                sd1_b,sd2_b = st.columns([2,3])
                with sd1_b:
                    foto('gambar/gambar7.png', 100)
                with sd2_b:
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:20px ; padding-top:0px'>Rata - Rata Harga Prediksi</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:38px ; padding-top:0px'>{formatrupiah(round(statistics.mean(hasiltesting2['prediksi'])))}</h1>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:30px; padding-top:0px; color:#6A51BC'>Visualisasi Data</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:30px'>Visualisasi hasil prediksi data testing harga gula pasir dari model B- DES yang telah dibuat mulai tanggal {start_date2_b} hingga tanggal {end_date2_b}</h1>", unsafe_allow_html=True)
            v3_b, v4_b, v5_b =st.columns([4,2,1])
            with v3_b:
                visualisasitest(gula, training2, testing2, 750)
                
            with v4_b:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:28px; color:#6A51BC'>ANALISA PLOT</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:16px; padding-top:0px'>Analisa Hasil Data Testing 19 Desember 2022 hingga 31 Mei 2023</h1>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"""<h3 style='text-align: justify; font-size:15px; font-family: helvetica'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga gula pasir. Grafik warna biru tua merupakan data aktual, dan grafik warna biru muda merupakan hasil prediksi
                dan grafik warna merah merupakah hasil prediksi data testing. Berdasarkan gambar disamping dapat disimpulkan bahwa hasil prediksi dari data uji coba atau testing mulai
                tanggal 19 Desember 2022 hingga 31 Mei 2023 menunjukkan adanya kecenderungan trend naik yang lambat</h3>""", unsafe_allow_html=True)
            
            with v5_b:
                st.info('MAPE MODEL',icon="📌")
                st.markdown(f"<h1 style='text-align: center; color: #DE4D86; padding-bottom:10px; font-size:25px'>{round(mape(training2['data_aktual'], training2['prediksi']),2)} %</h1>", unsafe_allow_html=True)
                st.info('MAPE TESTING',icon="📌")
                st.markdown(f"<h1 style='text-align: center; color: #DE4D86; padding-bottom:5px; font-size:25px'>{round(mape(testing2['data_aktual'], testing2['prediksi']),2)} %</h1>", unsafe_allow_html=True)

                st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:18px; padding-top: 13px;color:#6A51BC'>KETERANGAN :</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: justify; padding-bottom:0px; font-size: 15px; padding-top:0px'>Prediksi harga gula pasir di Jawa Timur menggunakan metode B-DES menghasilkan MAPE dibawah 10% sehingga dapat disimpulkan bahwa mempunyai peramalan yang sangat baik atau akurat yang tinggi</h1>", unsafe_allow_html=True)

        with tab7_b:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:15px; font-size:30px; padding-top:0px; color:#6A51BC'>Tabel Pediksi Periode Ke Depan</h1>", unsafe_allow_html=True)
            
            d1_b,d2_b = st.columns([2,3])
            with d1_b:
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Masukkan besar prediksi periode ke depan yang diinginkan dalam satuan hari</h1>", unsafe_allow_html=True)            
                title2 = st.text_input('Besar Prediksi', '365')
                title2 = int(title2)
                indexing2= np.arange((len(gula)+1),(len(gula)+(title2 +1)))

                periodedepan2 =np.zeros(title2)
                periodedepan2 = pd.DataFrame(periodedepan2)
                periodedepan2 = periodedepan2.rename(columns = {0: "Tanggal"}) 


                tanggal_kedepan(testgula2, periodedepan2)
                prediksi = pred_periodedepan(kons, slope, title2, testgula2)
                periodedepan2 = periodedepan2.assign(hasilprediksi = prediksi)
                periodedepan2 = periodedepan2.assign(index= indexing2)
                periodedepan2 = periodedepan2.set_index('index')

                mindate3_b = periodedepan2['Tanggal'][len(gula)+1]
                maxdate3_b= periodedepan2['Tanggal'][len(gula)+title2]
                st.markdown(f"<h1 style='text-align: left; color: #DE4D86; padding-bottom:10px; font-size:15px ; padding-top:0px'>Pilih tanggal untuk melihat data peramalan sesuai tanggal yang diinginkan</h1>", unsafe_allow_html=True)            
                start_date3_b = st.date_input(label="Tanggal Awal :",min_value=mindate3_b,value=mindate3_b)
                st.write(start_date3_b)
                end_date3_b = st.date_input(label="Tanggal Akhir :",max_value=maxdate3_b,value=maxdate3_b)
                st.write(end_date3_b, use_container_width=True)
                start_date3_b = pd.to_datetime(start_date3_b, infer_datetime_format=True)
                end_date3_b = pd.to_datetime(end_date3_b, infer_datetime_format=True)
                        
                hasilprediksi2=periodedepan2[(periodedepan2['Tanggal']>=start_date3_b) &(periodedepan2['Tanggal']<=end_date3_b) ]
                    
            with d2_b:
                st.markdown(f"<h1 style='text-align: center; padding-right:25px; font-size:17px; padding-top:10px;padding-bottom:30px;color: #6C5CD7'>Tabel Hasil prediksi selama {title2} hari ke depan yaitu dari tanggal {periodedepan2['Tanggal'][len(gula)+1]} hingga {periodedepan2['Tanggal'][len(gula)+title2]}  </h1>", unsafe_allow_html=True)
                st.dataframe(hasilprediksi2, use_container_width=True)
                csv2 = convert_df(hasilprediksi2)
                st.markdown(f"<h1 style='text-align: left; color: #6C5CD7 ; padding-bottom:10px; font-size:15px ; padding-top:0px'>Simpan data hasil prediksi</h1>", unsafe_allow_html=True)
                st.download_button(
                    label="Download data as CSV",
                    data=csv2,
                    file_name='gulapasir_bwema.csv',
                    mime='text/csv',
                )
            
            sd8_b,sd9_b= st.columns(2)
            with sd8_b:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:25px; padding-top:10px;  color:#6A51BC'>Visualisasi Periode Depan</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: justify; padding-right:25px; font-size:15px; padding-top:10px;padding-bottom:0px'>Plot data Prediksi {title2} hari ke depan bersama data aktual harga gula pasir di Jawa Timur, prediksi data training, dan testing secara keseluruhan</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:12px; padding-top:10px;  color:red'>Gambar 1</h1>", unsafe_allow_html=True)
                st.divider()
                visualisasi_prediksi(gula, training2, testing2, periodedepan2)
                
            with sd9_b:
                st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:25px; padding-top:10px; color:#6A51BC'>Visualisasi Periode Depan Interaktif</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: justify; padding-right:25px; font-size:15px; padding-top:10px'>Plot data prediksi periode depan sesuai dengan tanggal yang dipilih pada tabel di atas</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; font-size:12px; padding-top:10px;padding-bottom:15px;  color:red'>Gambar 2</h1>", unsafe_allow_html=True)
                st.divider()
                fig_depan2 = px.line(hasilprediksi2, x="Tanggal", y="hasilprediksi") 
                fig_depan2.update_layout(width=600,margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
                st.write(fig_depan2)

            col7_b, col8_b = st.columns([3,4])
            with col7_b:
                st.markdown(f"<h1 style='text-align: left;font-size:25px; color:#6A51BC; padding-top:0px'>Analisa Hasil Prediksi Selama {title2} hari ke Depan</h1>", unsafe_allow_html=True)
                st.divider()
                st.markdown(f"""<h3 style='text-align: justify; font-size:15px; font-family: helvetica'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga gula pasir. Grafik warna biru tua merupakan data aktual, dan grafik warna biru muda merupakan hasil prediksi
                ,grafik warna merah merupakah hasil prediksi data testing, dan grafik warna orange merupakan hasil prediksi periode depan. Berdasarkan Gambar 1 dapat disimpulkan bahwa hasil prediksi selama {title2} hari ke depan yaitu dari tanggal {periodedepan2['Tanggal'][len(gula)+1]} hingga {periodedepan2['Tanggal'][len(gula)+title2]}
                menunjukkan bahwa harga gula pasir di Jawa Timur mempunyai kecenderungan trend naik yang lambat</h3>""", unsafe_allow_html=True)
            with col8_b:
                
                st.markdown(f"<h1 style='text-align: center; padding-bottom:0px; font-size:33px; padding-top:10px; color:#6A51BC'>Statistika Deskriptif</h1>", unsafe_allow_html=True)
                st.divider()
                col9_b,col10_b,col11_b=st.columns(3)
                with col9_b:
                    st.info('HARGA TERENDAH',icon="📌")
                    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{formatrupiah(min(hasilprediksi2['hasilprediksi']))}</h2>", unsafe_allow_html=True)
                with col10_b:
                    st.info('HARGA TERTINGGI',icon="📌")
                    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{formatrupiah(max(hasilprediksi2['hasilprediksi']))}</h2>", unsafe_allow_html=True)
                with col11_b:
                    st.info('RATA-RATA HARGA',icon="📌")
                    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{formatrupiah(round(statistics.mean(hasilprediksi2['hasilprediksi'])))}</h2>", unsafe_allow_html=True)

    
    if genre == 'Perbandingan Metode' :
        st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px'>PERBANDINGAN ANTARA METODE B - DES DAN B - WEMA</h1>", unsafe_allow_html=True)
        st.divider()

        st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>PERBANDINGAN AKURASI</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Berikut merupakan perbandigan akurasi MAPE TESTING dari metode Brown's Double Exponential Smoothing dan Brown's Weighted Exponential Moving Average</h1>", unsafe_allow_html=True)
        pm1, pm2, pm3 = st.columns(3)
        with pm1 :
            sd1_b,sd2_b = st.columns([2,3])
            with sd1_b:
                foto('gambar/gambar5.png', 80)
            with sd2_b:
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:23px ; padding-top:0px'>MAPE METODE B - DES</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:40px ; padding-top:0px'>{round(mape(testing['data_aktual'], testing['prediksi']),2)}</h1>", unsafe_allow_html=True)
        with pm2 :
            sd1_b,sd2_b = st.columns([2,3])
            with sd1_b:
                foto('gambar/gambar6.png', 80)
            with sd2_b:
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:21px ; padding-top:0px'>MAPE METODE B - WEMA</h1>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:40px ; padding-top:0px'>{round(mape(testing2['data_aktual'], testing2['prediksi']),2)}</h1>", unsafe_allow_html=True)
        with pm3:
            st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:10px; font-size:23px ; padding-top:0px'>KESIMPULAN</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: right; color: #DE4D86; padding-bottom:0px; font-size:15px ; padding-top:0px'>Metode terbaik untuk memprediksi komoditi harga gula pasir adalah Brown's Weighted Exponential Moving Average karena menghasilkan MAPE terkecil</h1>", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"<h1 style='text-align: left; padding-bottom:10px; font-size: 30px; color:#6A51BC'>VISUALISASI</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: left ; padding-bottom:20px; font-size:18px ; padding-top:0px; font-style:normal'>Analisa Perbandingan visualisasi data testing pada metode B - DES dan B - WEMA</h1>", unsafe_allow_html=True)
        vt1, vt2 = st.columns([5,2])
        with vt1:
            data_1b = go.Scatter(x=gula['Tanggal'], y=gula['Harga'], name="Data Aktual", mode="lines")
            data_2b = go.Scatter(x=testing['Tanggal'], y=testing['prediksi'], name="Data Testing B- DES", mode="lines")
            data_3b= go.Scatter(x=testing2['Tanggal'], y=testing2['prediksi'], name="Data Testing B - WEMA", mode="lines")

            figtesting = go.Figure([data_1b, data_2b, data_3b]) 
            figtesting.update_layout(xaxis_title = 'Tanggal', yaxis_title = 'Harga', width = 1000)
            figtesting.update_layout(margin=dict(l=1,r=1,b=1,t=1),paper_bgcolor="white")
            st.write(figtesting, unsafe_allow_html=True)
        with vt2:
            st.markdown(f"<h1 style='text-align: left; padding-bottom:5px; font-size:28px'>ANALISA PLOT</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: left; padding-bottom:0px; font-size:16px; padding-top:0px'>Analisa Perbandingan visualisasi data testing pada metode B - DES dan B - WEMA</h1>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"""<h3 style='text-align: justify; font-size:14px; font-family: helvetica; padding-bottom:40px'>Sumbu x merupakan tanggal, dan sumbu y merupakan harga gula pasir . Grafik warna biru tua merupakan data aktual, grafik warna biru muda merupakan hasil prediksi data testing metode B - DES
            dan grafik warna merah merupakah hasil prediksi data testing metode B - WEMA. Berdasarkan gambar disamping dapat disimpulkan bahwa hasil prediksi dari data testing antara kedua metode tersebut sama - sama mempunyai kecenderungan trend naik. Namun, 
            hasil prediksi B-DES bergerak naik sangat lambat jika dibandingkan dengan metode B - WEMA. Jika dibandingkan maka metode B-WEMA dapat memprediksi lebih baik karena grafiknya berada lebih dekat dengan data aktual</h3>""", unsafe_allow_html=True)
        
        st.divider()

            
            