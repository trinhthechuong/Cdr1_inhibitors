import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import time
import os
from Utility.Featurizer_streamlit import featurizer
from Utility.cdr1_model_predict import ensemble_predict


            
st.set_page_config(layout="wide")

# HTML and CSS styles
html_temp = """
    <div style="background-color:#2C6D78;padding:1px">
    <h2 style="color:white;text-align:center;">EMCIP: CDR1 INHIBITORS PREDICTION MODEL </h2>
    </div>
    """
css = """
[data-testid="stSidebar"]{
        max-width: 200px;
    }
"""

st.markdown(html_temp, unsafe_allow_html=True)
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
 # Main menu
choose = option_menu("Main Menu", ["Predict a batch", "Predict one Molecule","About"],
                         menu_icon="app-indicator", default_index=0,
                         orientation = 'horizontal',
                         styles={
        "container": {"padding": "5!important", "background-color": "#03090a"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#9e9999"},
        "nav-link-selected": {"background-color": "#2C6D78"},
                             
    }
    )

st.session_state['choose'] = choose


if st.session_state['choose'] == "Predict a batch":
    # Upload a CSV file
    st.session_state["uploaded_file"] = None
    st.subheader('1. Upload CSV File')
    
    uploaded_file = st.file_uploader("Upload your file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Your file has {df.shape[0]} molecules")
        st.write(df)
        st.session_state["df"] = df
        st.session_state["uploaded_file"] = uploaded_file
    
    st.subheader('2. Data Featurization')
    ID_col = st.text_input("Input ID column", key="ID_col")
    SMILES_col = st.text_input("Input SMILES column", key="SMILES_col")
    n_jobs = st.slider("Number of processors", 1, 8, 2, 1, key="n_jobs")

    if st.button("Featurize", key="Featurize"):
        if st.session_state["uploaded_file"] is None:
            st.write("Please upload a file")
        else:
            df["Activity"] = 0
            featurizer_st = featurizer(df, ID=ID_col, smiles_col=SMILES_col, active_col="Activity", save_dir="Cdr1_classification/", n_jobs=n_jobs)
            rdk5_df, rdk6_df, rdk7_df, avalon_df, ph4_processed, mordred_df, graph_dataset = featurizer_st.run()
            st.session_state["rdk5_df"] = rdk5_df
            st.session_state["rdk6_df"] = rdk6_df
            st.session_state["rdk7_df"] = rdk7_df
            st.session_state["avalon_df"] = avalon_df
            st.session_state["ph4_processed"] = ph4_processed
            st.session_state["mordred_df"] = mordred_df
            st.session_state["graph_dataset"] = graph_dataset
            st.session_state["Calculated"] = "Yes"
            st.session_state["featurization_message"] = "Data featurization process is completed."

    if st.session_state.get("featurization_message"):
        st.success(st.session_state["featurization_message"])
    st.subheader('3. Cdr1 Inhibitors Classification')
    if st.button("Predict", key="Predict"):
        if st.session_state.get("Calculated") != "Yes":
            if st.session_state.get("uploaded_file") is None:
                st.write("Please upload your data")
            else:
                st.write("Please featurize your data")
        else:
            with st.spinner('Wait for it...'):
                time.sleep(5)
                fpt_datasets = [st.session_state["rdk5_df"], st.session_state["rdk6_df"], st.session_state["rdk7_df"], st.session_state["mordred_df"], st.session_state["avalon_df"], st.session_state["ph4_processed"]]
                predict = ensemble_predict(fpt_datasets, st.session_state["graph_dataset"], ID_col, SMILES_col)
                df_predictions = predict.predict()
                df_predictions = df_predictions.sort_values(by='Probability', ascending=False)
            st.success("Successfully predicted your data")
            st.write(df_predictions)
            df_predictions.to_csv("Cdr1_classification/predictions.csv", index=False)
            save_dir = os.path.join(os.getcwd(), "Cdr1_classification")
            st.markdown(f"All molecular representation dataset and prediction result are saved at **{save_dir}**")
            # csv = df_predictions.to_csv(index=False).encode('utf-8')
            # # st.download_button(
            # #     label="Download predictions as CSV",
            # #     data=csv,
            # #     file_name='predictions.csv',
            # #     mime='text/csv',
            # # )
            
            

