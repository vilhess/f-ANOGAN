import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json

st.set_page_config(layout="wide")

st.title("Anomaly Detections Using f-ANOGAN")
st.divider()

st.sidebar.title('Parameters: ')
anormal = st.sidebar.slider("Anomaly digit: ", min_value=0, max_value=9, value=0, step=1)
threshold = st.sidebar.slider("Threshold: ", min_value=0., max_value=0.2, step=0.01, value=0.05)

mean_error = Image.open(f"figures/Mean_error_{anormal}.jpg")
all_generation = Image.open(f"figures/Generated_Anomaly_{anormal}.jpg")
encoder_gen = Image.open(f"figures/Encoded_reconstructed_{anormal}.jpg")


col1, col2, col3 = st.columns(3)

with col1:
    st.header('Generator creation:')
    st.image(all_generation)

with col2:
    st.header('Encoder guiding generation:')
    st.write('')
    st.write('')
    st.write('')
    st.image(encoder_gen)

with col3:
    st.header('Mean score per digit:')
    st.image(mean_error)

st.divider()

st.header("Rejection using p-values")

with open(f"p_values/anogan_{anormal}.json", "r") as file:
    p_values_test = json.load(file)

with open(f"../VAE_AD/p_values/{anormal}.json", "r") as file:
    p_values_test_linear = json.load(file)

with open(f"../VAE_AD/p_values/conv_{anormal}.json", "r") as file:
    p_values_test_conv = json.load(file)

results = []
for digit in range(10):

    test_p_values, len_test = p_values_test[str(digit)]
    test_p_values = np.asarray(test_p_values)

    n_rejets = (test_p_values < threshold).sum().item()
    percentage_rejected = n_rejets / len_test

    test_p_values_linear, len_test_linear = p_values_test_linear[str(digit)]
    test_p_values_linear = np.asarray(test_p_values_linear)

    test_p_values_conv, len_test_conv = p_values_test_conv[str(digit)]
    test_p_values_conv = np.asarray(test_p_values_conv)

    n_rejets_linear = (test_p_values_linear < threshold).sum().item()
    percentage_rejected_linear = n_rejets_linear / len_test_linear

    n_rejets_conv = (test_p_values_conv < threshold).sum().item()
    percentage_rejected_conv = n_rejets_conv / len_test_conv


    dic = {
        "Digit": digit,
        "Anormal": "Yes" if digit == anormal else "No",
        "Threshold": threshold,

        "Rejections f-anogan": f"{n_rejets}/{len_test}",
        "Rejections vae (linear)": f"{n_rejets_linear}/{len_test_linear}",
        "Rejections vae (conv)": f"{n_rejets_conv}/{len_test_conv}",

        
        "Rejection Rate f-anogan": f"{percentage_rejected:.3%}",
        "Rejection Rate vae (linear)": f"{percentage_rejected_linear:.3%}",
        "Rejection Rate (conv)": f"{percentage_rejected_conv:.3%}"
    }


    results.append(dic)

df_results = pd.DataFrame(results)
st.table(df_results)