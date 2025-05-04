# -*- coding: utf-8 -*-
"""
Created on Tue Mon 10 10:57:37 2025

@author: TAJ2HIG
"""
import streamlit as st
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import glob
import plotly.graph_objects as go 
import plotly.io as pio
import plotly.express as px
import datetime
import cv2
import numpy as np
import lightgbm as lgb
import pickle
from PIL import Image
import sklearn.multioutput

import scipy.sparse.linalg

# ==================================================================================
### seabornのスタイル設定 ###
#sns.set_style(style="darkgrid")
#sns.set_context("paper")
#sns.set_color_codes("pastel")
#sns.set(font='IPAexGothic')

### List ###
Power_List = ["原子力", "火力", "水力", "地熱", "バイオマス", "太陽光発電実績", "風力発電実績"]
Power_List_Short = ["原子", "火力", "水力", "地熱", "バイオ", "太陽", "風力"]
Weekday_List = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
Temp_Trend = ["年の値", "1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
Temp_Trend_Name = ["all", "jan", "feb", "mar", "apr", "may", "jun", "jly", "aug", "sep", "oct", "nov", "dec"]

### Functions ###
def data_preproc(_df):
    """ Data PreProcessing """

    # Convert the date column to datetime
    _df['date'] = pd.to_datetime(_df['年月日時'])

    # Get each data info
    _df['year'] = _df['date'].dt.year
    _df['month'] = _df['date'].dt.month
    _df['day'] = _df['date'].dt.day
    _df['hour'] = _df['date'].dt.hour
    #_df['day_str'] = _df['date'].dt.strftime('%y/%m/%d')
    _df['day_str'] = _df['date'].dt.strftime('%Y-%m-%d')
    _df['month_str'] = _df['date'].dt.strftime('%y/%m')

    df_out = _df

    return df_out



# ==================================================================================

def main():
    ### Page Configuration ###
    title = "天気と電力を確認してみよう！"
    st.set_page_config(
        page_title=title,
        page_icon="🌈",
        layout="wide",
    )
    st.title(title)

    ### Page Content ###
    ## sidebar
    st.sidebar.title("電力データ分析ツール")
    st.sidebar.subheader("2-1.確認したい年を選ぶ")
    select_year = st.sidebar.radio("確認したい年を選んでみよう",
                                   (2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023))

    st.sidebar.subheader("2-2.データを比較してみよう")
    min_date = datetime.date(2016, 4, 1)
    max_date = datetime.date(2023, 12, 31)
    date_1 = st.sidebar.date_input('１つ目のデータ', 
                                   datetime.date(2023, 1, 1), 
                                   min_value=min_date, 
                                   max_value=max_date)
    date_2 = st.sidebar.date_input('２つ目のデータ', 
                                   datetime.date(2023, 1, 2), 
                                   min_value=min_date, 
                                   max_value=max_date)


    ########################
    # 1. File load
    ########################
    st.header("１．ファイルを読み込もう")
    with st.container(border=True):
        uploaded_file = st.file_uploader("ファイルをアップロードしてください", type="csv")

        ## Data pre-processsing
        if uploaded_file is not None:
            st.subheader("Data")
            # CSVファイルの読み込み、Headerを指定して読み込む
            df_input = pd.read_csv(uploaded_file, encoding='shift_jis')

            # Dataの前処理
            df_input_tmp = data_preproc(df_input)

            # Diaplay dataframe 
            st.text("▼ １時間ごとのデータ")
            st.dataframe(df_input_tmp.iloc[:,:-7], height=300) # 追加した列は非表示とする
            # Diaplay dataframe 
            st.text("▼ 日平均データ")
            # Daliy Data作成
            df_grouped = df_input_tmp.groupby('day_str').mean(numeric_only=True)
            st.dataframe(df_grouped.iloc[:,:-4], height=300) # 追加した列は非表示とする
            # Diaplay dataframe 
            st.text("▼ 月平均データ")
            df_grouped_month = df_input_tmp.groupby('month_str').mean(numeric_only=True)
            st.dataframe(df_grouped_month.iloc[:,:-4], height=300) # 追加した列は非表示とする

    st.markdown("---")

    ########################
    # 2. Graph
    ########################  
    st.header("２．グラフで確認してみよう") 
    with st.container(border=True):
        ## Data pre-processsing
        if uploaded_file is not None:
            st.subheader("2-1.一年間のデータの変化")
            # 指定された年のデータ作成
            df_cut_year = df_input_tmp[df_input_tmp['year']==select_year]

            st.write(f'{select_year}年が選択されています')
            st.dataframe(df_cut_year, height=300) # 追加した列は非表示とする

            fig = go.Figure()

            fig.add_trace(go.Scatter(x = df_cut_year['date'],
                                     y = df_cut_year['気温(℃)'],
                                     line=dict(color='orange'),
                                     name = '気温(℃)',
                                     yaxis = 'y1'))
            fig.add_trace(go.Scatter(x = df_cut_year['date'],
                                     y = df_cut_year['降水量(mm)'],
                                     line=dict(color='blue'),
                                     name = '降水量(mm)',
                                     yaxis = 'y1'))

            fig.update_layout(yaxis1=dict(side='left',
                                          showgrid=True,
                                          title='気温/降水量'),
                              plot_bgcolor='white',
                              title ='温度と降水量')

            st.plotly_chart(fig)

            # 1日のデータを比較する
            st.subheader("2-2. 1日のデータを比較しよう")
            # Data準備
            df_cut_1day_1 = df_input_tmp[df_input_tmp['day_str']==str(date_1)]
            df_cut_1day_2 = df_input_tmp[df_input_tmp['day_str']==str(date_2)]

            if date_1 == date_2:
                st.write(f'{date_1}と{date_2}を別日に設定してください。比較できません')
            else:
                st.write(f'{date_1}と{date_2}を比較しています')
                st.dataframe(df_cut_1day_1.iloc[:,:-7], height=300) # 追加した列は非表示とする
                st.dataframe(df_cut_1day_2.iloc[:,:-7], height=300) # 追加した列は非表示とする

                # グラフを表示
                col1,col2 = st.columns(2)
                with col1:
                    #st.line_chart(df_cut_1day_1[['気温(℃)','降水量(mm)']])
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x = df_cut_1day_1['hour'],
                                            y = df_cut_1day_1['気温(℃)'],
                                            line=dict(color='orange'),
                                            name = '気温(℃)',
                                            yaxis = 'y1'))
                    fig1.add_trace(go.Scatter(x = df_cut_1day_1['hour'],
                                            y = df_cut_1day_1['降水量(mm)'],
                                            line=dict(color='blue'),
                                            name = '降水量(mm)',
                                            yaxis = 'y1'))
                    fig1.update_layout(yaxis1=dict(side='left',
                                                showgrid=True,
                                                title='気温/降水量'),
                                    legend=dict(xanchor='left',
                                                yanchor='bottom',
                                                x=0.02,
                                                y=0.9,
                                                orientation='h',
                                                ),
                                    plot_bgcolor='white',
                                    title ='温度と降水量')
                    fig1.update_yaxes(range=(0, 40))
                    st.plotly_chart(fig1)

                with col2:
                    #st.line_chart(df_cut_1day_2[['気温(℃)','降水量(mm)']])
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x = df_cut_1day_2['hour'],
                                            y = df_cut_1day_2['気温(℃)'],
                                            line=dict(color='orange'),
                                            name = '気温(℃)',
                                            yaxis = 'y1'))
                    fig2.add_trace(go.Scatter(x = df_cut_1day_2['hour'],
                                            y = df_cut_1day_2['降水量(mm)'],
                                            line=dict(color='blue'),
                                            name = '降水量(mm)',
                                            yaxis = 'y1'))
                    fig2.update_layout(yaxis1=dict(side='left',
                                                showgrid=True,
                                                title='気温/降水量'),
                                    legend=dict(xanchor='left',
                                                yanchor='bottom',
                                                x=0.02,
                                                y=0.9,
                                                orientation='h',
                                                ),
                                    plot_bgcolor='white',
                                    title ='温度と降水量')
                    fig2.update_yaxes(range=(0, 40))
                    st.plotly_chart(fig2)

                # 棒グラフ
                col1,col2 = st.columns(2)
                with col1:
                    #st.bar_chart(df_cut_1day_1, x="hour", y=Power_List, y_label="発電量")
                    fig3 = go.Figure()
                    for idx in range(len(Power_List)):
                        fig3.add_trace(go.Bar(x=df_cut_1day_1["hour"], y=df_cut_1day_1[Power_List[idx]], name=Power_List_Short[idx]))
                    fig3.update_yaxes(range=(0, 5000)) # y軸を固定
                    fig3.update_layout(yaxis1=dict(side='left',
                                                showgrid=True,
                                                title='[万kW]'),
                                    legend=dict(xanchor='left',
                                                yanchor='bottom',
                                                x=0.02,
                                                y=0.98,
                                                orientation='h',
                                                ),
                                    barmode='stack',
                                    title ='発電量') # 積み上げBarGraphにする
                    st.plotly_chart(fig3)
                with col2:
                    #st.bar_chart(df_cut_1day_2, x="hour", y=Power_List, y_label="発電量")
                    fig4 = go.Figure()
                    for idx in range(len(Power_List)):
                        fig4.add_trace(go.Bar(x=df_cut_1day_2["hour"], y=df_cut_1day_2[Power_List[idx]], name=Power_List_Short[idx]))
                    fig4.update_yaxes(range=(0, 5000))
                    fig4.update_layout(yaxis1=dict(side='left',
                                                showgrid=True,
                                                title='[万kW]'),
                                    legend=dict(xanchor='left',
                                                yanchor='bottom',
                                                x=0.02,
                                                y=0.98,
                                                orientation='h',
                                                ),
                                    barmode='stack',
                                    title ='発電量') # 積み上げBarGraphにする
                    st.plotly_chart(fig4)
    
    st.markdown("---")

    ########################
    # AI Solusion
    ######################## 
    st.header("３．AIでデータを予測してみよう") 
    with st.container(border=True):  
        ## Data pre-processsing
        if uploaded_file is not None: 
            col1,col2,col3 = st.columns(3)
            with col1:
                img = Image.open('./Image/ai_chara.png')
                st.image(img, use_container_width=True)
            with col2:
                st.write("""##### 3-1.予測したい条件を入力""")
                select_model = st.selectbox ('使用したいモデルは何年分？',[2,8])
                min_date2 = datetime.date(2025, 1, 1)
                max_date2 = datetime.date(2040, 12, 31)
                date_3 = st.date_input('予測したい日は', 
                                        datetime.date(2025, 6, 1), 
                                        min_value=min_date2, 
                                        max_value=max_date2)
                
                select_month = date_3.month
                select_weekday = date_3.weekday() # 曜日情報を取得する
                st.write(f'{Weekday_List[select_weekday]}です')

                select_hour = st.selectbox ('時間を設定して下さい',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
                
                col21,col22,col23 = st.columns(3)
                with col21:
                    select_temp = st.selectbox ('気温(℃)',[0,5,10,15,20,25,30,35,40])
                with col22:
                    select_rain = st.selectbox ('降水量(mm)',[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,10.0,20.0])
                with col23:
                    select_humi = st.selectbox ('湿度(%)',[0,10,20,30,40,50,60,70,80,90,100])

            with col3:
                st.write("""##### 3-2.予測結果""")

                list_ai_input = [] # データを格納するList
                list_ai_input.append(select_month)
                list_ai_input.append(select_hour)
                list_ai_input.append(select_weekday)
                list_ai_input.append(select_temp)
                list_ai_input.append(select_rain)
                list_ai_input.append(select_humi)
                ai_input = np.array(list_ai_input)
                #
                #  保存したモデルをロードする
                if select_model == 2:
                    loaded_model = pickle.load(open("./Model/model_wo_power.sav", 'rb')) # 2年
                else:
                    loaded_model = pickle.load(open("./Model/model_wo_power_long.sav", 'rb')) # 8年
                # テストデータの予測
                y_result = loaded_model.predict(ai_input.reshape(1, -1))

                result_fire = round(y_result[0][0], 1)
                result_sun = round(y_result[0][1], 1)

                # 表示部分
                if st.button("AI予測スタート", key=1):
                    st.write("")
                    st.write("")
                    st.write("火力発電量は、")
                    col32,col33 = st.columns(2)
                    with col32:
                        st.header(result_fire)
                    with col33:
                        st.write("[万kw]です")
                    st.write("")
                    st.write("太陽光発電量は、")
                    col35,col36 = st.columns(2)
                    with col35:
                        st.header(result_sun)
                    with col36:
                        st.write("[万kw]です")

    st.markdown("---")

    ########################
    # 日時から、温度などを予測しよう
    ########################
    st.header("4．温度や湿度を予測しよう")
    with st.container(border=True):
        if uploaded_file is not None: 
            col1,col2,col3 = st.columns(3)
            with col1:
                st.write("""##### 4-1.予測したい日は？""")
                min_date4 = datetime.date(2025, 1, 1)
                max_date4 = datetime.date(2030, 12, 31)
                date_4 = st.date_input('予測したい日は', 
                                        datetime.date(2025, 1, 1), 
                                        min_value=min_date4, 
                                        max_value=max_date4)
                select_month_4 = date_4.month
                select_day_4 = date_4.day
                select_weekday_4 = date_4.weekday() # 曜日情報を取得する
                st.write(f'{Weekday_List[select_weekday_4]}です')

                select_hour_4 = st.selectbox ('時間は？',[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])

            with col2:
                st.write("""##### 4-2.温度と降水量、湿度は""")
                st.write('過去8年のデータから予測すると...')

                #  保存したモデルをロードする
                load_model_temp = pickle.load(open("./Model/model_temp_predict_long.sav", 'rb'))
                load_model_rain = pickle.load(open("./Model/model_rain_predict_long.sav", 'rb'))
                load_model_humi = pickle.load(open("./Model/model_humi_predict_long.sav", 'rb'))
                # 予測用データ作成
                list_ai_input2 = [] # データを格納するList
                list_ai_input2.append(select_month_4)
                list_ai_input2.append(select_day_4)
                list_ai_input2.append(select_hour_4)
                list_ai_input2.append(select_weekday_4)
                ai_input2 = np.array(list_ai_input2)
                # テストデータの予測
                y_result_temp = load_model_temp.predict(ai_input2.reshape(1, -1))
                y_result_rain = load_model_rain.predict(ai_input2.reshape(1, -1))
                y_result_humi = load_model_humi.predict(ai_input2.reshape(1, -1))

                col211,col212,col213 = st.columns(3)
                with col211: st.write('気温：')
                with col212: st.write(f'##### {round(y_result_temp[0],3)}')
                with col213: st.write('(℃)')
                col221,col222,col223 = st.columns(3)
                with col221: st.write('降水量：')
                with col222: st.write(f'##### {round(y_result_rain[0],3)}')
                with col223: st.write('(mm)')
                col231,col232,col233 = st.columns(3)
                with col231: st.write('湿度：')
                with col232: st.write(f'##### {round(y_result_humi[0],3)}')
                with col233: st.write('(%)')

                st.write('です。')
            with col3:
                img = Image.open('./Image/ai_chara_2.png')
                st.image(img, use_container_width=True)

    st.markdown("---") 


if __name__ == '__main__':
    main()
