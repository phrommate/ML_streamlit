import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE
import pandas as pd

st.title('Iris Data Visualization!')
st.subheader('PCA')
df = px.data.iris()
#df
# PCA
d = st.select_slider("Select PCA dimension",
                     options=[1, 2, 3])
if d == 3:
    X = PCA(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    X['species'] = df['species']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z',
                  color='species')
    st.plotly_chart(fig)
if d == 2:
    X = PCA(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['species'] = df['species']
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='species')
    st.plotly_chart(fig)
if d == 1:
    X = PCA(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['species'] = df['species']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='species')
    st.plotly_chart(fig)
# MDS
st.subheader('MDS')
d = st.select_slider("Select MDS dimension",
                     options=[1, 2, 3])
if d == 3:
    X = MDS(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    X['species'] = df['species']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z',
                  color='species')
    st.plotly_chart(fig)
if d == 2:
    X = MDS(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['species'] = df['species']
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='species')
    st.plotly_chart(fig)
if d == 1:
    X = MDS(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['species'] = df['species']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='species')
    st.plotly_chart(fig)
# Isomap
st.subheader('Isomap')
d = st.select_slider("Select Isomap dimension",
                     options=[1, 2, 3])
k = st.select_slider("Select Isomap neighbors",
                     options=list(range(3, 20)))
if d == 3:
    X = Isomap(n_components=3, n_neighbors=k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    X['species'] = df['species']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z',
                  color='species')
    st.plotly_chart(fig)
if d == 2:
    X = Isomap(n_components=2, n_neighbors=k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['species'] = df['species']
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='species')
    st.plotly_chart(fig)
if d == 1:
    X = Isomap(n_components=1, n_neighbors=k).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['species'] = df['species']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='species')
    st.plotly_chart(fig)
# t-SNE
st.subheader('t-SNE')
d = st.select_slider("Select t-SNE dimension",
                     options=[1, 2, 3])
if d == 3:
    X = TSNE(n_components=3).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    X['species'] = df['species']
    fig = px.scatter_3d(X,
                  x='x',
                  y='y',
                  z='z',
                  color='species')
    st.plotly_chart(fig)
if d == 2:
    X = TSNE(n_components=2).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['species'] = df['species']
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='species')
    st.plotly_chart(fig)
if d == 1:
    X = TSNE(n_components=1).fit_transform(df.iloc[:, :4])
    X = pd.DataFrame(X, columns=['x'])
    X['species'] = df['species']
    X['y'] = 0
    fig = px.scatter(X,
                  x='x',
                  y='y',
                  color='species')
    st.plotly_chart(fig)