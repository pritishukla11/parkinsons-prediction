import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
# from PIL import Image

# loading the trained model
pickle_in = open('PD.pkl', 'rb')
svm = pickle.load(pickle_in)
def predict_pd(MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter,  MDVP_Jitter1,  MDVP_RAP, MDVP_PPQ1,Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer1,
                             Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,D2,DFA,spread1,spread2,PPE):
    input_data=[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter1, MDVP_RAP, MDVP_PPQ1, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer1,
    Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, D2, DFA, spread1, spread2, PPE]
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray([input_data])

    # reshape the numpy array
    # input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    scaler = StandardScaler()
    std_data = scaler.fit_transform(input_data_as_numpy_array)

    prediction = svm.predict(std_data)
    print(prediction)


    return(int(prediction[0]))



def main():
    # front end elements of the web page

    html_temp = """ 
    <div style ="background-color: #2b3845   ;padding:13px"> 
    <h1 style ="font-family:monospace;font-size: 30px;color:#f2eef6;text-align:center;">Parkinson Diseases Prediction(PDP)</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.subheader('Visualization Settings')
    uploaded_file=st.sidebar.file_uploader(label="Upload your csv file or Excel file",type=['csv','xlsx'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)


    subheader= """ <p style="font-family:sans-serif; color: #40E0D0; font-size: 15px;">  Machine Learning Web App to predict Parkinson Disease,Built with Streamlit, Deployed using Heroku. </p>"""
    st.markdown(subheader, unsafe_allow_html=True)
    MDVP_Fo = st.number_input("Enter The Average vocal fundamental frequency in Hz")
    MDVP_Fhi = st.number_input("Enter The Maximum vocal fundamental frequency in Hz")
    MDVP_Flo = st.number_input(" Enter The Minimum vocal fundamental frequency in Hz")

    MDVP_Jitter = float(st.number_input(" Enter The measure of variation in fundamental frequency in  %"))
    MDVP_Jitter1 = st.number_input(" Enter The measure of variation in fundamental frequency in (Abs)")
    MDVP_RAP = st.number_input(" Enter The measure of variation in fundamental frequency RAP")
    MDVP_PPQ1 = st.number_input(" Enter The measure of variation in fundamental frequency PPQ")
    Jitter_DDP = st.number_input(" Enter The measure of variation in fundamental frequency DDP")

    MDVP_Shimmer = st.number_input(" Enter The measure of variation in Amplitude Shimmer")
    MDVP_Shimmer1 = st.number_input(" Enter The measure of variation in Amplitude in (dB)")
    Shimmer_APQ3 = st.number_input(" Enter The measure of variation in Amplitude APQ3")
    Shimmer_APQ5 = st.number_input(" Enter The measure of variation in Amplitude APQ5")
    MDVP_APQ = st.number_input(" Enter The measure of variation in Amplitude APQ")
    Shimmer_DDA = st.number_input(" Enter The measure of variation in Amplitude DDA")

    NHR = st.number_input(" Enter The  ratio of noise to tonal components in the voice NHR")
    HNR = st.number_input(" Enter The  ratio of noise to tonal components in the voice HNR")

    RPDE = st.number_input(" Enter The  nonlinear dynamical complexity measure RPDE")
    D2 = st.number_input(" Enter The  nonlinear dynamical complexity measure D2")

    DFA = st.number_input(" Enter The  Signal fractal scaling exponent measure DFA")

    spread1 = st.number_input(" Enter The  nonlinear measures of fundamental frequency variation spread1 ")
    spread2 = st.number_input(" Enter The  nonlinear measures of fundamental frequency variation spread2 ")

    PPE = st.number_input(" Enter The  nonlinear measures of fundamental frequency variation3 PPE ")







    if st.button("Predict Disease"):
        result = predict_pd(MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter,  MDVP_Jitter1,  MDVP_RAP, MDVP_PPQ1,Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer1,
                              Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,D2,DFA,spread1,spread2,PPE)


        if (result == 0):
            st.success("The Person does not have Parkinsons Disease")

        else:
            st.success("The Person has Parkinsons")




if __name__ == '__main__':
    main()


