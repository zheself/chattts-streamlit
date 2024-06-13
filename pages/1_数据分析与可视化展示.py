import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px

with st.container():
    st.write("# 数据分析与可视化展示")
    st.markdown("---")  # 分隔线



st.write("### 1.地震分布热点图：")
image1 = Image.open('./img/1.png')
st.image(image1, caption='地震分布热点图')

with st.container():
    st.write("可以很清楚地看到地震带的分布，重大地震也主要发生在地震带上。")
    st.markdown("---")  # 分隔线



st.write("### 2.重大地震分布：")
data = pd.read_csv('./csv/earthquake1.csv')
# 设置页面配置
fig2 = px.scatter_geo(data,
                      color = data.Magnitude,
                      color_continuous_scale = px.colors.sequential.Inferno,
                      lon = data.Longitude,
                      lat = data.Latitude,
                      animation_frame = data.Year,
                      hover_name = data.Type,
                      hover_data = ["Longitude",
                                    "Latitude",
                                    "Date",
                                    "Time",
                                    "Magnitude",
                                    "Depth"
                                   ],
                      size = np.exp(data.Magnitude)/100,
                      projection = "equirectangular",
                      title = '1965-2016年全球重大地震'
                      )

# 在Streamlit应用程序中显示图形
st.plotly_chart(fig2)

with st.container():
    st.write("每年份、月份、天份发生重大地震的次数。")
    st.markdown("---")  # 分隔线



st.write("### 3.按年份统计柱状图：")
image2 = Image.open('./img/2.png')
st.image(image2, caption='按年份统计柱状图')

with st.container():
    st.write("前十多年的地震次数明显少于之后的，可能是早期地震监测技术比较落后的原因。")
    st.markdown("---")  # 分隔线



st.write("### 4.按月份统计柱状图：")
image2 = Image.open('./img/3.png')
st.image(image2, caption='按月份统计柱状图')

with st.container():
    st.write("每月发生的地震次数基本上一致。")
    st.markdown("---")  # 分隔线



st.write("### 5.按天统计柱状图：")
image2 = Image.open('./img/4.png')
st.image(image2, caption='按天统计柱状图')

with st.container():
    st.write("由于有一半的月份没有31号，因此31号的次数明显较少。")
    st.markdown("---")  # 分隔线



st.write("### 6.国内重大地震：")
image2 = Image.open('./img/5.png')
st.image(image2, caption='国内重大地震图')

with st.container():
    st.write("1955-2016年中国境内不同省份的重大地震次数")
    st.markdown("---")  # 分隔线



st.write("### 7.各省份柱状图：")
image2 = Image.open('./img/6.png')
st.image(image2, caption='各省份柱状图')

with st.container():
    st.write("每个省份（海域）所生成的柱状图。")
    st.markdown("---")  # 分隔线



st.write("### 8.词云图：")
image2 = Image.open('./img/7.png')
st.image(image2, caption='词云图')

with st.container():
    st.write("每个省份（海域）所生成的词云。")
    st.markdown("---")  # 分隔线



st.write("### 9.震级TOP500：")
image2 = Image.open('./img/8.png')
st.image(image2, caption='震级TOP500')

with st.container():
    st.write("震级前500的重大地震。")
    st.markdown("---")  # 分隔线



st.write("### 10.震深TOP500：")
image2 = Image.open('./img/9.png')
st.image(image2, caption='震深TOP500')

with st.container():
    st.write("震源深度前500的重大地震。")
    st.markdown("---")  # 分隔线



st.write("### 11.地震来源（Type）：")
image2 = Image.open('./img/10.png')
st.image(image2, caption='地震来源（Type）')

with st.container():
    st.write("可见99.2%是天然形成的地震，不到1%是核爆或者爆炸形成的。同样地也可以画出中国境内不同类型的地震占比。")
    st.markdown("---")  # 分隔线



st.write("### 12.地震相关性分析：")
image2 = Image.open('./img/11.png')
st.image(image2, caption='地震相关性分析')

with st.container():
    st.write("可见震级8.0以上的地震大部分震源深度较浅，但从整体看地震的震级与震源深度的相关性不大。")
    st.markdown("---")  # 分隔线



st.write("### 13.地震数据聚类分析分析：")
image2 = Image.open('./img/12.png')
st.image(image2, caption='地震数据聚类分析分析图')

with st.container():
    st.write("可以清晰的看出地震带分布，环太平洋地震带与地中海喜马拉雅地震带地震明显频繁。")
    st.markdown("---")  # 分隔线