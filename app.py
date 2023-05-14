import streamlit as st
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input

model = load_model(r"C:\Users\chink\best_model_incv3.h5")
model_class = load_model(r"C:\Users\chink\best_model1_incv3.h5")



# Set the title and description of the app
st.title("Fake or Real image")
st.write("Upload an image and we'll tell you if it's fake or real image.")

# Create a file uploader and accept only image files
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
img = load_img(image_file, target_size=(224,224))
i= img_to_array(img)
i = preprocess_input(i)

input_arr = np.array([i])
input_arr.shape




# Display the image and a message if it exists
if image_file is not None:
    # Open the image using Pillow
    image = Image.open(image_file)
    
    # Display the image using Streamlit
    st.image(image, caption="Uploaded Image")
    
    # Check if the image is a cat or a dog
pred = np.argmax(model.predict(input_arr))
pred_class = np.argmax(model_class.predict(input_arr))

if pred == 0:
    plt.imshow(input_arr[0])
    plt.title("input image: currency is fake")
    plt.axis = False
    plt.show()
    st.write("currency is fake")
    
else:
    if pred_class == 0:
        plt.imshow(input_arr[0])
        plt.title("input image: currency is real: 10 rupees")
        plt.axis = False
        plt.show()
        st.write("currency is real: 10 rupees")
        
    elif pred_class == 1:
        plt.imshow(input_arr[0])
        plt.title("input image: currency is real: 100 rupeesl")
        plt.axis = False
        plt.show()
        st.write("currency is real: 100 rupees")
        
    elif pred_class == 2:
        plt.imshow(input_arr[0])
        plt.title("input image: currency is real: 20 rupees")
        plt.axis = False
        plt.show()
        st.write("currency is real: 20 rupees")
        
    elif pred_class == 3:
        plt.imshow(input_arr[0])
        plt.title("input image: currency is real: 200 rupees")
        plt.axis = False
        plt.show()
        st.write("currency is real: 200 rupees")
        
    elif pred_class == 4:
        plt.imshow(input_arr[0])
        plt.title("input image: currency is real: 2000 rupees")
        plt.axis = False
        plt.show()
        st.write("currency is real: 2000 rupees")
        
    elif pred_class == 5:
        plt.imshow(input_arr[0])
        plt.title("input image: currency is real: 50 rupees")
        plt.axis = False
        plt.show()
        st.write("currency is real: 50 rupees")
        
    elif pred_class == 6:
        plt.imshow(input_arr[0])
        plt.title("input image: currency is real: 500 rupees")
        plt.axis = False
        plt.show()
        st.write("currency is real: 500 rupees")
    
    elif pred_class == 7:
        plt.imshow(input_arr[0])
        plt.title("input image: no image")
        plt.axis = False
        plt.show()
        st.write("no image")