from pycaret import clustering
import streamlit as st
import pandas as pd
from PIL import Image

clustering_model = clustering.load_model('FinalModel')


def predict(model, input_df):
    output_df = clustering.predict_model(model, data=input_df)
    predictions = output_df['Cluster'][0]
    return predictions


def run():

    image = Image.open('customer_segmentation.png')

    st.image(image, use_column_width=True)

    add_selectbox = st.sidebar.selectbox(
        "Choose your prediction mode.",
        ("Entry by entry", "As a batch")
    )

    st.sidebar.info(
        "This app segments customers based on their behavior."
    )

    st.title("Customer Segmentation App")

    if add_selectbox == 'Entry by entry':

        age = st.number_input('Age', min_value=25, max_value=100, value=25)
        income = st.number_input('Income', min_value=9000, max_value=200000, value=200000)
        spending_score = st.number_input('SpendingScore', min_value=0.0, max_value=1.0, format="%.2f")
        savings = st.number_input('Savings', min_value=0.0, max_value=25000.0, format="%.2f")

        output = ""

        input_df = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'SpendingScore': [spending_score],
            'Savings': [savings]
        })

        if st.button("Predict"):
            output = predict(model=clustering_model, input_df=input_df)
            output = str(output)

        st.success(f"Your customer belongs to {output}.")

    if add_selectbox == "As a batch":

        file_upload = st.file_uploader("Please upload your file.", type=['csv'])

        if file_upload is not None:
            input_df = pd.read_csv(file_upload)
            predictions = clustering.predict_model(model=clustering_model, data=input_df)
            st.write(predictions)


if __name__ == '__main__':
    run()




