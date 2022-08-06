import streamlit as st
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from mtcnn import MTCNN #Face detection
from PIL import Image # image Preprocess
import cv2 #image read 
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import cv2
            
#FUNCTION TO SAVE THE UPLOADED IMAGE BY USER
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join("uploads" , uploaded_image.name), "wb") as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

filename = pickle.load(open(r"filename.pkl" , "rb"))
model = VGGFace(model = "resnet50" , include_top = False , input_shape=(224,224,3) , pooling = "avg")
detector = MTCNN() 

#FUNCTION TO EXTRACT FEATURES
def extract_feaatures(path,model):

    sample_img = cv2.imread(path)     #reading image in np.array
    #print(sample_img)
    results = detector.detect_faces(sample_img)  #MTCNN used to detect the coordiantes of the face
    x,y,width,height =  results[0]["box"]         # gives the position of face 
    face = sample_img[y:y+height , x:x+width]    # crop the pixel to the onlt to have face
    image = Image.fromarray(face)                 # convert pixel to img
    image = image.resize((224,224))               # resize to enter into CNN
    face_array = np.array(image)                  # getting back to array 
    face_array  = face_array.astype("float32")    # converting values to float
    expanded_img = np.expand_dims(face_array,axis = 0)    # having desired format (1,224,224,3)
    #print(expanded_img.shape)                        
    preprocess_img = preprocess_input(expanded_img)       
    result = model.predict(preprocess_img).flatten()  #Extracting Features
    return result

#FUNCTION TO FIND SIMILARITY
features_list = pickle.load(open(r"features.pkl" , "rb"))
features_list = np.array(features_list)

def prediction(result):
    similarity = []
    for feature in features_list:
        similarity.append(list(cosine_similarity(result.reshape(1,-1) , feature.reshape(1,-1)))[0][0])
        similarity_score = sorted(list(enumerate(similarity)) , reverse=True , key = lambda x: x[1])[0][0]
    return similarity_score


   
#FUNCTION FOR WEBSITE
def load_image(image_file):
    display_img = Image.open(image_file)
    newsize = (200,200)
    display_img = display_img.resize(newsize)
    return display_img

st.title("Which celeb Are You?? ")

st.subheader("Image")
uploaded_image = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if uploaded_image is not None:      
        # To View Uploaded Image
        if save_uploaded_image(uploaded_image):
            display_image = load_image(uploaded_image)     
                       
            #Extract the features
            features = extract_feaatures(os.path.join("uploads" , uploaded_image.name),model )
            

            #Finding Prediciton
            similarity_score = prediction(features)

            #DISPLAY RESULTS
            col1,col2 = st.columns(2)

            with col1:
                st.header(" Uploaded Image")
                st.image(display_image)
            with col2:               
                path = filename[similarity_score]
                st.header(path.split("/")[3])                
                st.image(path[1:], width=200)             
                
            
                 
