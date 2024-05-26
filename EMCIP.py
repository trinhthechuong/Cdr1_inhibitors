import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import time
import os
import shutil
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from Utility.Featurizer_streamlit import featurizer
from Utility.cdr1_model_predict import ensemble_predict
from stmol import showmol
import py3Dmol
from conformation_encode.generate_conformations import generate_conformations
import io, base64
from pathlib import Path
            
st.set_page_config(layout="wide")
st.markdown("<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'>",unsafe_allow_html=True)
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
# choose = option_menu("Main Menu", ["Predict a batch", "Predict a molecule","About"],
#                          menu_icon="app-indicator", default_index=0,
#                          orientation = 'horizontal',
#                          styles={
#         "container": {"padding": "5!important", "background-color": "#03090a"},
#         "icon": {"color": "orange", "font-size": "25px"}, 
#         "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#9e9999"},
#         "nav-link-selected": {"background-color": "#2C6D78"},
                             
#     }
#     )
choose = option_menu("Main Menu", ["Predict a batch", "Predict a molecule","About"],
                         menu_icon="app-indicator", default_index=0,
                         orientation = 'horizontal',
                         styles={
        "container": {"padding": "5!important", "background-color": "#F2F3F4"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#2C6D78"},                        
    }
    )
def restart():
    if os.path.exists("Cdr1_classification/"):
        shutil.rmtree("Cdr1_classification/")
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

def render_mol(mol_confs, cids):
    mol = Chem.RemoveHs(mol_confs)
    p = py3Dmol.view(width=1000, height=500)
    colors=('cyanCarbon','redCarbon','blueCarbon','magentaCarbon','whiteCarbon','purpleCarbon','greenCarbon','yellowCarbon','orangeCarbon','greyCarbon')
    for i,cid in enumerate(cids):
        IPythonConsole.addMolToView(mol,p,confId=cid)
    for i,cid in enumerate(cids):
        p.setStyle({'model':i,},
                        {'stick':{'colorscheme':colors[i%len(colors)]}})
    p.zoomTo()
    showmol(p,height=500,width=1500)



st.session_state['choose'] = choose


if st.session_state['choose'] == "Predict a batch":
    # Upload a CSV file
    st.session_state["uploaded_file"] = None
    st.subheader('1. Upload CSV File')
    
    uploaded_file = st.file_uploader("Upload your file", type=['csv'])
    st.session_state["uploaded"] = False
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        # st.write("Your file has ", st.session_state["df"].shape[0], " molecules")
        # st.write(df)
        # st.session_state["df"] = df
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["uploaded"] = True
    # st.write(st.session_state)
    if st.session_state.get("uploaded"):
        st.write("Your file has ", str(st.session_state["df"].shape[0]), " molecules")
        st.write(st.session_state["df"])
    
    st.subheader('2. Data Featurization')
    ID_col = st.text_input("Input ID column", key="ID_col")
    SMILES_col = st.text_input("Input SMILES column", key="SMILES_col")
    n_jobs = st.slider("Number of processors", 1, 8, 2, 1, key="n_jobs")

    if st.button("Featurize", key="Featurize"):
        if st.session_state["uploaded_file"] is None:
            st.write("Please upload your data")
        elif st.session_state["SMILES_col"] == "" or st.session_state["ID_col"] == "":
            st.write("Please input ID and SMILES column")
        
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
        elif st.session_state["uploaded"] == False:
            st.write("Please upload your data")
        else:
            with st.spinner('Wait for it...'):
                time.sleep(5)
                fpt_datasets = [st.session_state["rdk5_df"], st.session_state["rdk6_df"], st.session_state["rdk7_df"], st.session_state["mordred_df"], st.session_state["avalon_df"], st.session_state["ph4_processed"]]
                predict = ensemble_predict(fpt_datasets, st.session_state["graph_dataset"], ID_col, SMILES_col)
                df_predictions = predict.predict()
                df_predictions = df_predictions.sort_values(by='Probability', ascending=False)
                st.session_state["Predicted"]=True
                st.session_state["df_predictions"] = df_predictions
    if st.session_state.get("Predicted"):
        st.write(st.session_state["df_predictions"])
        st.success("Successfully predicted your data")
        
        st.session_state["df_predictions"].to_csv("Cdr1_classification/predictions.csv", index=False)
        save_dir = os.path.join(os.getcwd(), "Cdr1_classification")
        st.markdown(f"All molecular representation dataset and prediction result are saved at **{save_dir}**")
    if st.button("Restart"):
        restart()
            # csv = df_predictions.to_csv(index=False).encode('utf-8')
            # # st.download_button(
            # #     label="Download predictions as CSV",
            # #     data=csv,
            # #     file_name='predictions.csv',
            # #     mime='text/csv',
            # # )
if st.session_state["choose"] == "Predict a molecule":
    st.subheader("1. Input SMILES")
    cur_smiles = "COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O"
    smile_input = st.text_input("Please, input your SMILES",cur_smiles, key="smile_input")
    st.subheader("2. CDR1 Inhibitors Prediction")
    if st.button("Predict", key="Predict"):
        if smile_input == "":
            st.write("Please input your SMILES")
        with st.spinner('Wait for it...'):
            time.sleep(5)
            df_input = pd.DataFrame({"ID":[1],"Smiles":smile_input})
            df_input.columns = ["ID","SMILES"]
            st.session_state["df_input"] = df_input
            df_input["Activity"] = 0
            featurizer_st = featurizer(df_input, ID="ID", smiles_col="SMILES", active_col="Activity", save_dir="Cdr1_classification/", n_jobs=-1)
            rdk5_df, rdk6_df, rdk7_df, avalon_df, ph4_processed, mordred_df, graph_dataset = featurizer_st.run()
            fpt_datasets = [rdk5_df, rdk6_df, rdk7_df, mordred_df, avalon_df, ph4_processed]
            predict = ensemble_predict(fpt_datasets, graph_dataset, "ID", "SMILES")
            df_predictions = predict.predict()
            proba = df_predictions["Probability"].values[0]
            st.session_state["proba"] = proba
            st.session_state["The probability of your molecule to be a CDR1 inhibitor"] = f"The probability of your molecule to be a CDR1 inhibitor is: {proba}"
            mol_obj = Chem.MolFromSmiles(st.session_state["smile_input"])
            confs = generate_conformations(mol = mol_obj, num_confs = 50, rmsd = 0.5,seed= 42, energy = 10, max_attemp = 1000, num_threads = -1)
            mol, cids,_ = confs.gen_confs()
            st.session_state["mol"] = mol
            cid_extrated = []
            for conf in mol.GetConformers():
                cid_extrated.append(conf.GetId())
            st.session_state["cid_extrated"] = cid_extrated
            st.session_state["visulize"] = True
    if st.session_state.get("visulize"):
        choose_displayed_conf=st.radio("Number of displayed conformations",("1","2","All"), key="choose_displayed_conf", horizontal=True)
        if choose_displayed_conf == "1":
            conf_number = [st.session_state["cid_extrated"][0]]
            st.write("This is the first conformation of your molecule")
            render_mol(st.session_state["mol"], conf_number)
        elif choose_displayed_conf == "2":
            if len(st.session_state["cid_extrated"]) > 2:
                conf_number = [st.session_state["cid_extrated"][0], st.session_state["cid_extrated"][1]]
                st.write("These are the first two conformations of your molecule")
                render_mol(st.session_state["mol"], conf_number)
            else:
                conf_number = st.session_state["cid_extrated"]
                st.write("These are all the generated conformations of your molecule")
                render_mol(st.session_state["mol"], conf_number)
        else:
            st.write("These are all the generated conformations of your molecule")
            conf_number = st.session_state["cid_extrated"]
            render_mol(st.session_state["mol"], conf_number)
    if st.session_state.get("The probability of your molecule to be a CDR1 inhibitor"):
            st.success(st.session_state["The probability of your molecule to be a CDR1 inhibitor"])
    if st.button("Restart"):
        restart()


s_buff = io.StringIO()
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path,max_width=500):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='max-width: {}px;'>".format(
      img_to_bytes(img_path), max_width
    )
    return img_html
            
if choose == "About":
    st.markdown('')
    st.markdown("<h3 style='color:#2C6D78; text-align:center;'>Author</h3>",unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'> <a href='https://trinhthechuong.github.io/'>The-Chuong Trinh</a></p>",unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'> <a href='https://www.researchgate.net/profile/Viet-Khoa-Tran-Nguyen'>Viet-Khoa Tran-Nguyen</a></p>",unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'> <a href='https://www.researchgate.net/profile/Pierre-Falson'>Pierre Falson</a></p>",unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'> <a href='https://www.researchgate.net/profile/Ahcene-Boumendjel'>Ahcène Boumendjel*</a></p>",unsafe_allow_html=True)
    st.markdown("<h3 style='color:#2C6D78; text-align:center;'>Laboratory</h3>",unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>"+
                    img_to_html('Images/09_lrb_logo.png',256)+(' ')+ img_to_html('Images/UGA_logo.png',128)+
                    "</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'>Laboratoire Radiopharmaceutiques Biocliniques - Université Grenoble Alpes</h5>",unsafe_allow_html=True)
    ##Supervisor
    st.markdown("<h3 style='color:#2C6D78; text-align:center;'>Supervisor</h3>",unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'> <a href='https://www.researchgate.net/profile/Ahcene-Boumendjel'>Prof. Ahcène Boumendjel, Ph.D</a></p>",unsafe_allow_html=True)
    st.markdown("<h5 style='text-align:center;'>Laboratoire Radiopharmaceutiques Biocliniques - Université Grenoble Alpes</h5>",unsafe_allow_html=True)