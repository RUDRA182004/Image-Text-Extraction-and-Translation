import cv2 as cv
import easyocr
import matplotlib.pyplot as plt
import streamlit as st
from googletrans import Translator, LANGUAGES
import numpy as np
from PIL import Image

# Function to extract text from image
def extract_text(image):
    reader = easyocr.Reader(['en', 'es', 'fr', 'de', 'it'], gpu=False)
    extracted_text = reader.readtext(image)
    text = " ".join([words[1] for words in extracted_text])
    
    # Draw bounding boxes around text
    for words in extracted_text:
        bbox, text, score = words
        top_left = tuple([int(val) for val in bbox[0]])
        bottom_right = tuple([int(val) for val in bbox[2]])
        cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv.putText(image, text, top_left, cv.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)
    
    return image, text

# Function to translate text
def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Streamlit UI
st.title("Image Text Extraction and Translation")

# Upload an image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Select language for translation
language = st.selectbox("Select the language for translation", list(LANGUAGES.values()))

if uploaded_image is not None:
    # Read and process the uploaded image
    image = np.array(Image.open(uploaded_image))
    processed_image, extracted_text = extract_text(image.copy())
    
    # Display extracted text
    st.image(processed_image, caption='Processed Image', use_column_width=True)
    st.write("**Extracted Text:**")
    st.write(extracted_text)
    
    # Translate the extracted text
    translated_text = translate_text(extracted_text, language)
    
    # Display translated text
    st.write("**Translated Text:**")
    st.write(translated_text)
