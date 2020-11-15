import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# https://github.com/streamlit/demo-self-driving/blob/master/streamlit_app.py

def plot_plotly(df,x, y,title):
    n = df[x].values.tolist()
    fig = go.Figure()
    for name in y:
        m = df[name]
        fig.add_trace(go.Scatter(x=n, y=m,
                      mode='lines',#mode='lines+markers',
                      name=name))
    fig.update_layout(
        showlegend=False,
        hovermode = "x",
        #paper_bgcolor = "rgb(0,0,0)" ,
        #plot_bgcolor = "rgb(10,10,10)" , 
        dragmode="pan",
        title=dict(
            x = 0.5,
            text = title,
            font=dict(
                size = 20,
                color = "rgb(0,0,0)"
            )
        )
    )
    return fig


def page_covid():

    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: 95%;
                padding-top: 5px;
                padding-left: 5px;
                padding-right: 5px;
                padding-bottom: 5px;                
            }}
        </style>
        """,
                unsafe_allow_html=True,
        )


    st.title("Covid Analisi")
    df = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
    df["data"] = [el[0:10] for el in df["data"].values.tolist()]
    st.dataframe(df) #, height=300, width=700)

    select = ["deceduti","totale_casi","dimessi_guariti"]
    select_options = st.multiselect('Seleziona cosa vuoi plottare', list(df.keys()), default=select)

    fig = plot_plotly(df,x ="data", y=select_options,title="Andamento Nazionale")    
    st.plotly_chart(fig, use_container_width=True)


    col1, col2, col3 = st.beta_columns(3)
    with col1:
       st.header("A cat")
       st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

    with col2:
       st.header("A dog")
       st.image("https://static.streamlit.io/examples/dog.jpg", use_column_width=True)

    with col3:
       st.header("An owl")
       st.image