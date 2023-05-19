# Currency-Classification-and-Recognition
This repository is for recognition of fake and real currency and classifying the real currency. 
Fake or Real Currency Detection using Inception V3 Training Model and VGG16 for classification purpose.
# Motivation and Goal
The purpose of this project is to develop a robust system for detecting fake or real currency using the Inception V3 and VGG16 training model and is to accurately distinguish between genuine and counterfeit banknotes, thereby aiding in the detection of counterfeit currency and enhancing financial security. Our goal is to build an efficient and accurate solution that can assist financial institutions, businesses, and law enforcement agencies in quickly identifying counterfeit currency , thus enhancing security measures and reducing financial losses.

# Model Description and Working
The Inception V3 model is a powerful convolutional neural network architecture that has been pre-trained on a large dataset. It is capable of learning intricate features and patterns from images, making it an ideal choice for our currency detection task. By fine-tuning the Inception V3 model with a dataset of genuine and counterfeit currency images, we can train it to recognize key differentiators between the two types, enabling accurate recognition. The currency classification model based on VGG16 utilizes the deep layers and convolutional filters of VGG16 architecture to extract high-level features from banknote images. By fine-tuning the pre-trained VGG16 model with a dataset of genuine and counterfeit currency, it can effectively classify and distinguish between real and fake banknotes with high accuracy.

# Experimental Setup and Dependencies
To replicate and run this project, you will need the following:

Python 3: The programming language used for implementation.

TensorFlow: An open-source deep learning framework used for building and training neural networks.

Keras: A high-level neural networks API that runs on top of TensorFlow, simplifying the model development process.

glob: Library used for file pattern matching and retrieval.

NumPy: A library for numerical computations in Python, used for handling arrays and matrices efficiently.

Streamlit: is an open-source Python library used for building interactive web applications , enabling quick and easy development with a focus on simplicity and intuitive user interfaces.

Dataset - Download dataset from Kaggle

For classification-
https://www.kaggle.com/datasets/vishalmane109/indian-currency-note-images-dataset-2020

For Fake or Real Recognition
https://www.kaggle.com/datasets/jayaprakashpondy/indian-currency-dataset

# How to Run
For training the model run model.ipynb file.

For interface run command -streamlit run app.py

# Performance Analysis
In our model for fake and real currency recognition we achieved accuracy of 98% for training and 88% for testing using Inception V3 pre-trained model.
For currency classification using VGG16 we achieved accuracy of 85% for training and 80% for validation .

First we created a scratch cnn model where we obtained accuracy of 70% for fake and real currency recognition and for classification we obtained accuracy of 68% .Since the accuracy was low the prediction was wrong and to avoid this and improve our model's performannce we used pre-trained models.

We choose this model because when we tried with MobileNet and Resnet 50 it gave us low accuracy compared to VGG16 and Inception V3 .
Therefore we built our final model using this pre trained models.


The performance of the currency detection and classification system heavily relies on the quality and diversity of the dataset used for training. A well-curated dataset with a sufficient number of genuine and counterfeit currency images is crucial for achieving high accuracy. Additionally, the detection accuracy may vary depending on factors such as image quality, lighting conditions, and the complexity of counterfeit techniques.

To evaluate the performance of the trained model, we suggest using metric as accuracy. These metrics can provide insights into the model's ability to correctly recognize real and fake currency instances and classify it. Regular evaluation and continuous improvement of the model can lead to enhanced detection accuracy and robustness.

Please note that the performance analysis may differ based on the specific dataset, training configuration, and evaluation methodology.
