import streamlit as st
st.set_page_config(page_title="Covid19 Detection", page_icon="🦠", layout="wide", initial_sidebar_state="collapsed")

import os
import time
from datetime import datetime

# Viz Pkgs
import cv2
from PIL import Image, ImageEnhance
import numpy as np

# AI Pkgs
import tensorflow as tf

def main():
    """Simple tool for covid19 detection from chest x-ray data"""
    
    html_template = """
        <div style="">
        <h1 style="">Covid19 detection tool</h1>
        </div>
    """
    
    st.markdown(html_template, unsafe_allow_html=True)
    
    st.sidebar.image("covid-19.webp", width=250)
    
    image_file = st.sidebar.file_uploader("Upload yout image file")
    if image_file is not None:
        our_image = Image.open(image_file)
        
        if st.sidebar.button("Image Preview"):
            st.sidebar.image(our_image, width=250)
        
        activities = ["Image Enhancement", "Diagnosis", "Disclaimer and Info"]
        choise = st.sidebar.selectbox("Select Activity", activities)
        
        match choise:
            case "Image Enhancement":
                st.subheader("Image Enhancement tool")
                
                col1, col2= st.columns(2)
               
                with col1:
                    c_rate = st.slider("Constrast", 0.5, 5.0, 1.0)
                    enhancer = ImageEnhance.Contrast(our_image)
                    img_constrast = enhancer.enhance(c_rate)
                
                with col2:
                    c_rate = st.slider("Brightness", 0.5, 5.0, 1.0)
                    enhancer = ImageEnhance.Brightness(img_constrast)
                    img_brightness = enhancer.enhance(c_rate)
                    
                if st.checkbox("Show Original Image"):
                    col3, col4 = st.columns(2)
                    with col3:
                        st.text("Original Image")
                        st.image(our_image, width=600, use_column_width=True)
                    with col4:
                        st.text("Enhanced Image")
                        img_output = img_brightness
                        st.image(img_output, width=600, use_column_width=True)
                else:
                    st.text("Enhanced Image")
                    img_output = img_brightness
                    st.image(img_output, width=600, use_column_width=True)
                
            case "Diagnosis":
                st.subheader("Diagnosis report")
                
                if st.sidebar.button("Diagnosis"):
                    new_img = np.array(our_image.convert('RGB')) #our image is converted into an array
                    new_img = cv2.cvtColor(new_img,1) #0 is original, 1 is grayscale
                    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                    st.text("Chest X-Ray")
                    st.image(gray, width=400, use_column_width=True)

                    #X-Ray Imge Pre-processing
                    IMG_SIZE = (200, 200)
                    img = cv2.equalizeHist(gray)
                    img = cv2.resize(img, IMG_SIZE)
                    img = img/255 #normalization

                    # Image reshaping according to tensorflow format
                    X_Ray = img.reshape((1, 200, 200, 1))

                    #Pre-trained CNN Model loading
                    model = tf.keras.models.load_model("./models/Covid19_CNN_Classifier.h5")
                    st.write(model.summary())
                    
                    #Diagnosis (Prediction== Binary Classification)
                    diagnosis_proba = model.predict(X_Ray)
                    diagnosis = np.argmax(diagnosis_proba,axis=1)

                    my_bar = st.sidebar.progress(0)

                    for percent_complete in range(100):
                        time.sleep(0.05)
                        my_bar.progress(percent_complete + 1)

                    #Diagnosis Cases: No-Covid=0, Covid=1
                    if diagnosis == 0:
                        st.sidebar.success("DIAGNOSIS: NO COVID-19")
                    else:
                        st.sidebar.error("DIAGNOSIS: COVID-19")

                    st.warning("This Web App is just a DEMO about Streamlit and Artificial Intelligence and there is no clinical value in its diagnosis!")




            case "Disclaimer and Info":
                st.subheader("Disclaimer and Info")
                st.write("**This Tool is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!**")
                st.write("**Please don't take the diagnosis outcome seriously and NEVER consider it valid!!!**")
                st.subheader("Info")
                st.write("This Tool gets inspiration from the following works:")
                st.write("- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)") 
                st.write("- [Fighting Corona Virus with Artificial Intelligence & Deep Learning](https://www.youtube.com/watch?v=_bDHOwASVS4)") 
                st.write("- [Deep Learning per la Diagnosi del COVID-19](https://www.youtube.com/watch?v=dpa8TFg1H_U&t=114s)")
                st.write("We used 206 Posterior-Anterior (PA) X-Ray [images](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/metadata.csv) of patients infected by Covid-19 and 206 Posterior-Anterior X-Ray [images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) of healthy people to train a Convolutional Neural Network (made by about 5 million trainable parameters) in order to make a classification of pictures referring to infected and not-infected people.")
                st.write("Since dataset was quite small, some data augmentation techniques have been applied (rotation and brightness range). The result was quite good since we got 94.5% accuracy on the training set and 89.3% accuracy on the test set. Afterwards the model was tested using a new dataset of patients infected by pneumonia and in this case the performance was very good, only 2 cases in 206 were wrongly recognized. Last test was performed with 8 SARS X-Ray PA files, all these images have been classified as Covid-19.")
                st.write("Unfortunately in our test we got 5 cases of 'False Negative', patients classified as healthy that actually are infected by Covid-19. It's very easy to understand that these cases can be a huge issue.")
                st.write("The model is suffering of some limitations:")
                st.write("- small dataset (a bigger dataset for sure will help in improving performance)")
                st.write("- images coming only from the PA position")
                st.write("- a fine tuning activity is strongly suggested")
                st.write("")
                st.write("Anybody has interest in this project can drop me an email and I'll be very happy to reply and help.")
                
            case _:
                pass
            
    if st.sidebar.button("About the author"):
        st.sidebar.subheader("Covid-19 detection tool")
        st.sidebar.markdown(f"""
                            by bakudas \n
                            [bakudas@vacaroxa.com](mailto:bakudas@vacaroxa.com)
                            """)
        st.sidebar.text(f"All rigths reserved @ {datetime.now().year}")
        

if __name__ == "__main__":
    main()
