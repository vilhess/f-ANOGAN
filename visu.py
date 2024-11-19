import streamlit as st
import numpy as np
import pandas as pd
import json

st.set_page_config(layout="wide")

st.title("Anomaly Detections Using f-ANOGAN")
st.divider()

st.sidebar.title('Parameters: ')
anormal = st.sidebar.slider("Anomaly digit: ", min_value=0, max_value=9, value=0, step=1)
threshold = st.sidebar.slider("Threshold: ", min_value=0., max_value=0.2, step=0.01, value=0.05)

with open(f"p_values/anogan_{anormal}.json", "r") as file:
    p_values_test = json.load(file)

results = []
for digit in range(10):
    test_p_values, len_test = p_values_test[str(digit)]
    test_p_values = np.asarray(test_p_values)

    n_rejets = (test_p_values < threshold).sum().item()
    percentage_rejected = n_rejets / len_test

    # Ajouter les données au tableau
    results.append({
        "Digit": digit,
        "Anormal": "Yes" if digit == anormal else "No",
        "Threshold": threshold,
        "Rejections f-anogan": f"{n_rejets}/{len_test}",
        "Rejection Rate f-anogan": f"{percentage_rejected:.3%}",
    })

# Convertir les résultats en DataFrame pour l'affichage
df_results = pd.DataFrame(results)

# Afficher le tableau dans Streamlit
st.table(df_results)