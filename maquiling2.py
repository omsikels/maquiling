import streamlit as st
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import cv2
import re

# Initialize PaddleOCR (downloads necessary models on first run)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

st.title("PaddleOCR with Streamlit")
st.write("Upload an image to extract text using PaddleOCR (offline).")

# Predefined bounding boxes 
BBOXES = {
    "account_number": (24, 23, 109, 17),  # (x, y, width, height)
    "account_name": (181, 23, 197, 18),
    "check_number": (539, 9, 111, 29),
    "BRSTN": (662, 10, 97, 29),
    "DATE": (539, 44, 210, 30),
    "PESOS": (21, 107, 724, 24)
}

# File uploader widget for image upload
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image in RGB mode
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run OCR"):
        with st.spinner("Processing..."):
            # Convert PIL image to a NumPy array
            image_np = np.array(image)
            
            # Run OCR using PaddleOCR
            result = ocr_model.ocr(image_np, cls=True)
            
            # Extract text based on bounding boxes
            extracted_data = {"account_number": "", "account_name": "", "check_number": "", "BRSTN": "", "DATE": "", "PESOS": ""}

            for line in result:
                for word_info in line:
                    bbox, (text_candidate, _) = word_info
                    x, y = bbox[0]
                    width = bbox[2][0] - bbox[0][0]
                    height = bbox[2][1] - bbox[0][1]

                    for key, (bx, by, bwidth, bheight) in BBOXES.items():
                        if bx <= x <= bx + bwidth and by <= y <= by + bheight:
                            extracted_data[key] = text_candidate.strip()

            # Fix DATE formatting (MM-DD-YYYY)
            if "DATE" in extracted_data["DATE"]:
                extracted_data["DATE"] = extracted_data["DATE"].replace("DATE", "").strip()
            
            # Extract date using regex and format it properly
            date_match = re.search(r"(\d{2})(\d{2})(\d{4})", extracted_data["DATE"])
            if date_match:
                month, day, year = date_match.groups()
                extracted_data["DATE"] = f"{month}-{day}-{year}"

        st.success("OCR Complete!")

        # Display extracted values
        st.text_area("Extracted Account Number", extracted_data["account_number"], height=70)
        st.text_area("Extracted Account Name", extracted_data["account_name"], height=70)
        st.text_area("Extracted CHECK No.", extracted_data["check_number"], height=70)
        st.text_area("Extracted BRTSN", extracted_data["BRSTN"], height=70)
        st.text_area("Extracted DATE", extracted_data["DATE"], height=70)
        st.text_area("Extracted PESOS", extracted_data["PESOS"], height=70)

        # Optionally, display an annotated image with bounding boxes
        if st.checkbox("Show Annotated Image with Bounding Boxes"):
            for key, (bx, by, bwidth, bheight) in BBOXES.items():
                image_np = cv2.rectangle(image_np, (bx, by), (bx + bwidth, by + bheight), (0, 255, 0), 2)
            st.image(image_np, caption="Annotated Image", use_column_width=True)
