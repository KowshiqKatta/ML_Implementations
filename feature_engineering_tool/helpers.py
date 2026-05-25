import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from fancyimpute import IterativeImputer

def analyze_csv_file(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = st.session_state.get('df', df) # to make it permenant even if the session is refreshed

        if 'columns_removed' not in st.session_state:
            st.write(f"**Columns:** {list(df.columns)}")
            selected_columns = st.multiselect("Select columns to remove", options = df.columns)

            if st.button("Remove selected columns"):
                if selected_columns:
                    df = df.drop(columns = selected_columns)
                    st.session_state['df'] = df
                    st.session_state['columns_removed'] = True
                    st.success(f"Columns removed: {', '.join(selected_columns)}")
                    st.write("### Dataframe after removing selected columns")
                    st.write(df)

        missing_values = df.isnull().sum()
        has_missing_values = missing_values.sum() > 0

        duplicate_count = df.duplicated().sum()
        has_duplicates = duplicate_count > 0

        if has_duplicates or has_missing_values:
            st.warning("There are missing/duplicated values with the uploaded csv file")

            if has_missing_values:
                st.write("### Missing values")
                st.write(missing_values[missing_values > 0])

                for column in missing_values[missing_values > 0].index:
                    st.write(f"#### column: {column}")

                    if st.button(f"Remove rows with missing values in '{column}'"):
                        df = df.dropna(subset = [column])
                        st.session_state['df'] = df
                        st.success(f"Rows with missing values in '{column}' removed successfully.")

                    if st.button(f"Fill missing values in '{column}' with mean"):

                        imputer = SimpleImputer(strategy = 'mean')
                        df[column] = imputer.fit_transform(df[column])

                        st.session_state['df'] = df
                        st.success(f"Missing values in '{column}' filled with mean.")
                    
                    if st.button(f"Fill missing values in '{column}' with median"):

                        imputer = SimpleImputer(strategy = 'median')
                        df[column] = imputer.fit_transform(df[column])

                        st.session_state['df'] = df
                        st.success(f"Missing values in '{column}' filled with median.")
                    
                    if st.button(f"Fill missing values in '{column}' with mode"):

                        imputer = SimpleImputer(strategy = 'mode')
                        df[column] = imputer.fit_transform(df[column])

                        st.session_state['df'] = df
                        st.success(f"Missing values in '{column}' filled with mode.")

                    custom_value = st.text_input(f"custom value to fill missing values in '{column}'")
                    if st.button(f"fill missing values in '{column}' with a custom value"):
                        if custom_value:
                            df[column].fillna(custom_value, inplace = True)
                            st.session_state['df'] = df
                            st.success(f"Missing values in '{column}' filled with custom value")
                        else:
                            st.warning("Please provide a custom value.")

                    if st.button(f"Apply iterative imputation for '{column}'"):
                        imputer = IterativeImputer()
                        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
                        st.session_state['df'] = df 
                        st.success(f"iterative imputation applied")