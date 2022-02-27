import streamlit as st
import streamlit.components.v1 as components
import io
import PIL
import numpy as np
import breed
from reader import Reader
from segmentation import Segmenter
import matplotlib.pyplot as plt

def matrixToMpl(dct):
    res = dct['result']
    lung = dct['lung']
    ct = dct['ct']
    fig = plt.figure(figsize=(24, 10))
    plt.subplot(131)
    print(type(ct), ct.shape)
    plt.imshow(ct, cmap = 'bone')
    plt.title('original ct slice')

    plt.subplot(132)
    plt.imshow(lung, cmap = 'bone')
    plt.title('lung')

    plt.subplot(133)
    plt.imshow(ct, cmap = 'bone')
    plt.imshow(res, alpha=0.5, cmap = 'bone')
    plt.title('predicted infection mask')

    return fig
    
st.title("Моё Data science портфолио")
st.header("Проекты:")
st.subheader("Нейросеть, определяющая породу собаки по фото:")
st.write('[Github проекта](https://github.com/Kostia2004/BreedsNN)')
uploaded_photo = st.file_uploader("Choose a photo", type=['jpg', 'png'])
if uploaded_photo is not None:
     # To convert to a string based IO:
     img = PIL.Image.open(uploaded_photo)
     st.text(breed.resolve(img))
     st.image(img)

st.subheader("")
st.subheader("")

st.subheader("Нейросеть, сегментирующая области, пораженные COVID-19 на снимках КТ легких")
st.write('[Github проекта](https://github.com/Kostia2004/CovidSegmentation)')
uploaded_scan = st.file_uploader("Choose nii file", type=['nii'])
testScanPath = 'testfiles/radiopaedia_org_covid-19-pneumonia-29_86491_1-dcm.nii'
if uploaded_scan is None:
    reader = Reader()
    arr = reader.read(testScanPath)
    segmenter = Segmenter()
    result, lung, ct = segmenter.segmentation(arr)
    alllist = [{'result': list(result)[i][...,0], 'lung': list(lung)[i][...,0], 'ct': list(ct)[i][...,0]} for i in range(arr.shape[2])]
    print(len(alllist))
    figlist = list(map(matrixToMpl, alllist))
    st.pyplot(figlist[0])
