import streamlit as st
from helpers import *

def main():
    st.title("Auto feature selector tool")

    uploaded_file = st.file_uploader("Choose a CSV file", type = "csv")

    if uploaded_file is not None:
        analyze_csv_file(uploaded_file)

if __name__ == "__main__":
    main()