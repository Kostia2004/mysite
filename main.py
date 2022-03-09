import streamlit as st
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
    plt.imshow(ct, cmap = 'bone')
    plt.title('original ct slice')

    plt.subplot(132)
    plt.imshow(lung, cmap = 'bone')
    plt.title('lung')

    plt.subplot(133)
    plt.imshow(ct, cmap = 'bone')
    plt.imshow(res, alpha=0.5, cmap = 'nipy_spectral')
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
scan_choice = st.radio("Select scan for predict", ("Upload", "First test", "Second test", "Third test"))

arr = None

if 'scan_choice' not in list(st.session_state.keys()):
    st.session_state.scan_choice = None
if scan_choice=="Upload":
    uploaded_scan = st.file_uploader("Choose nii file", type=['nii'])
    if uploaded_scan is not None:
        bytestr = uploaded_scan.getvalue()
        reader = Reader()
        arr = reader.read_bytes(bytestr)

test1 = 'testfiles/radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii'
test2 = 'testfiles/radiopaedia_org_covid-19-pneumonia-29_86491_1-dcm.nii'
test3 = 'testfiles/radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii'

if scan_choice!="Upload":
    print(scan_choice, st.session_state.scan_choice)
    reader = Reader()
    testScanPath = ""
    match scan_choice:
        case "First test":
            testScanPath = test1
        case "Second test":
            testScanPath = test2
        case "Third test":
            testScanPath = test3

    arr = reader.read(testScanPath)

if arr is not None:
    segmenter = Segmenter()
    if (('result' not in st.session_state) and ('lung' not in st.session_state) and ('ct' not in st.session_state)) or scan_choice!=st.session_state.scan_choice:
        print(scan_choice, st.session_state.scan_choice)
        st.session_state.result, st.session_state.lung, st.session_state.ct = segmenter.segmentation(arr)
    if 'alllist' not in st.session_state or scan_choice!=st.session_state.scan_choice:
        st.session_state.alllist = [{'result': list(st.session_state.result)[i][...,0], 'lung': list(st.session_state.lung)[i][...,0], 'ct': list(st.session_state.ct)[i][...,0]} for i in range(arr.shape[2])]
    if 'figlist' not in st.session_state or scan_choice!=st.session_state.scan_choice:
        st.session_state.figlist = list(map(matrixToMpl, st.session_state.alllist))
    slicenum = st.slider("Number of slice", 1, len(st.session_state.alllist))
    st.pyplot(st.session_state.figlist[slicenum-1])
    st.session_state.scan_choice = scan_choice
