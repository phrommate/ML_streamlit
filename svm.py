import streamlit as st
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

tmp = load_iris(as_frame=True)
X = tmp['data']
Y = tmp['targer']
itrain = np.r_[0:25, 50:75, 100:125]
itest = np.r_[25:50, 75:100, 125:150]

cls = SVC()