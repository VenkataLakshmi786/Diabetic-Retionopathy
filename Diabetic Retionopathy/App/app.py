# Import required libraries
import os
import html
import textwrap
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import keras
try:
    import keras_hub
except ImportError:
    keras_hub = None
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Page configuration
st.set_page_config(page_title="RetinaCare | Diabetic Retinopathy Detection", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

# Model configuration and constants
MODEL_PATH = "best_dr_model.keras"
IMAGE_SIZE = 224
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Helper function to render HTML content
def render_html(content):
    st.html(textwrap.dedent(content).strip())

# Custom styling for the application interface
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 80% 10%, rgba(91, 153, 225, 0.18), transparent 30%), linear-gradient(135deg, #eaf2ff 0%, #f8fbff 48%, #e6efff 100%);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102b55 0%, #071b3b 100%);
    border-right: 1px solid rgba(255,255,255,0.12);
}

[data-testid="stSidebar"] * {
    color: #ffffff;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

.sidebar-logo {
    text-align: center;
    padding: 10px 0 20px;
}

.sidebar-logo svg {
    width: 180px;
    height: 95px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    line-height: 1.4;
    margin-top: 8px;
    text-align: left;
}

.sidebar-title span {
    color: #ff5a5f;
}

.sidebar-divider {
    height: 1px;
    background: rgba(255,255,255,0.4);
    margin: 20px 0;
}

.sidebar-heading {
    font-size: 15px;
    font-weight: 800;
    margin: 18px 0 12px;
}

.sidebar-description {
    color: #e1eaff;
    font-size: 13px;
    line-height: 1.7;
}

.class-item {
    display: flex;
    align-items: center;
    gap: 11px;
    margin: 10px 0;
    font-size: 13px;
}

.class-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.sidebar-note {
    background: rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 14px;
    font-size: 12px;
    line-height: 1.6;
    margin-top: 22px;
}

.main-header {
    text-align: center;
    padding: 5px 5px 0;
}

.main-eye {
    width: 115px;
    height: 82px;
    margin: auto;
}

.main-title {
    color: #102c58;
    font-size: clamp(29px, 3.3vw, 45px);
    font-weight: 800;
    line-height: 1.2;
    margin: 5px 0 10px;
    letter-spacing: -1.2px;
}

.main-title span {
    color: #f04d52;
}

.main-subtitle {
    color: #102c58;
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 12px;
}

.main-description {
    color: #52698d;
    font-size: 15px;
    line-height: 1.6;
    max-width: 750px;
    margin: auto;
}

.heartbeat {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 20px auto;
    max-width: 500px;
    color: #ff5358;
}

.heartbeat:before,
.heartbeat:after {
    content: "";
    height: 1px;
    background: #ff777b;
    flex: 1;
}

.heartbeat span {
    font-size: 28px;
    padding: 0 12px;
}

.upload-heading {
    text-align: center;
    color: #102c58;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 6px;
}

.upload-description {
    text-align: center;
    color: #52698d;
    font-size: 13px;
    margin-bottom: 14px;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.85);
    border: 2px dashed #438cff;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(38,91,160,0.08);
}

[data-testid="stFileUploader"] section {
    border: none;
}

[data-testid="stFileUploader"] button {
    background: #102d59;
    color: white;
    border-radius: 25px;
    border: none;
    font-weight: 700;
}

[data-testid="stFileUploader"] button:hover {
    background: #1d4b87;
    color: white;
}

.feature-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #d4e3ff;
    border-radius: 19px;
    min-height: 130px;
    padding: 18px 8px;
    text-align: center;
    box-shadow: 0 7px 20px rgba(34,83,150,0.12);
}

.feature-icon {
    font-size: 34px;
    margin-bottom: 7px;
}

.feature-title {
    color: #102c58;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 6px;
}

.feature-text {
    color: #52698d;
    font-size: 12px;
    line-height: 1.5;
}

.section-title {
    color: #102c58;
    font-size: 22px;
    font-weight: 800;
    margin: 20px 0 14px;
}

.result-card {
    background: #ffffff;
    border: 1px solid #d8e5fa;
    border-radius: 18px;
    padding: 23px 18px;
    box-shadow: 0 7px 22px rgba(34,83,150,0.10);
    text-align: center;
    min-height: 160px;
    margin-bottom: 12px;
}

.result-label {
    color: #647795;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 9px;
}

.result-value {
    color: #102c58;
    font-size: 24px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.confidence-value {
    color: #168b55;
    font-size: 30px;
    font-weight: 800;
}

.image-info {
    background: #ffffff;
    border: 1px solid #d8e5fa;
    border-radius: 16px;
    padding: 15px;
    margin-top: 12px;
    box-shadow: 0 5px 18px rgba(34,83,150,0.08);
}

.image-info p {
    color: #647795;
    font-size: 13px;
    margin: 7px 0;
}

.waiting-card {
    background: #ffffff;
    border: 1px dashed #a9c5eb;
    border-radius: 18px;
    padding: 35px 18px;
    text-align: center;
    min-height: 250px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.waiting-icon {
    font-size: 45px;
    margin-bottom: 15px;
}

.waiting-title {
    color: #102c58;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 12px;
}

.waiting-text {
    color: #647795;
    font-size: 13px;
    line-height: 1.7;
}

.medical-notice {
    background: #e4edff;
    border-left: 4px solid #397ce0;
    color: #29466f;
    padding: 14px 17px;
    border-radius: 10px;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 22px;
}

.stButton > button {
    background: linear-gradient(90deg, #123568, #20569b);
    color: white;
    border: none;
    border-radius: 30px;
    height: 48px;
    font-weight: 750;
    font-size: 15px;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #20569b, #2874c5);
    color: white;
    border: none;
}

.footer {
    text-align: center;
    color: #38577f;
    font-size: 13px;
    font-weight: 600;
    font-style: italic;
    padding: 28px 0 12px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Load pre-trained diabetic retinopathy model
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

# Preprocess retinal image for model input
def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    image_array = np.array(image, dtype=np.uint8)
    processed_image = np.expand_dims(image_array, axis=0)
    return processed_image

# Predict diabetic retinopathy severity class and probabilities
def predict_severity(model, processed_image):
    predictions = model.predict(processed_image, verbose=0)
    probabilities = np.asarray(predictions[0], dtype=np.float32).flatten()
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} model outputs, but received {len(probabilities)}.")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(np.sum(probabilities), 1.0, atol=1e-3):
        probabilities = tf.nn.softmax(probabilities).numpy()
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)
    return predicted_class, confidence, probabilities

# Render sidebar with system information and class labels
with st.sidebar:
    render_html("""
    <div class="sidebar-logo">
        <svg viewBox="0 0 220 110" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 55 Q75 -5 135 55 Q75 115 15 55Z" fill="none" stroke="white" stroke-width="7"/>
            <circle cx="75" cy="55" r="25" fill="none" stroke="white" stroke-width="7"/>
            <circle cx="75" cy="55" r="10" fill="white"/>
            <path d="M140 55 L165 55 L175 25 L185 83 L195 43 L205 55 L218 55" fill="none" stroke="#ff555b" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div class="sidebar-title">
            Diabetic Retinopathy<br>
            <span>Detection</span>
        </div>
    </div>
    <div class="sidebar-divider"></div>
    <div class="sidebar-heading">ⓘ &nbsp; About This App</div>
    <div class="sidebar-description">
        This application analyses retinal fundus images and predicts diabetic retinopathy severity across five categories.
    </div>
    <div class="sidebar-divider"></div>
    <div class="sidebar-heading">▰ &nbsp; Classification Classes</div>
    <div class="class-item"><span class="class-dot" style="background:#27b45b"></span>No_DR</div>
    <div class="class-item"><span class="class-dot" style="background:#ffbd19"></span>Mild</div>
    <div class="class-item"><span class="class-dot" style="background:#ff922e"></span>Moderate</div>
    <div class="class-item"><span class="class-dot" style="background:#ff4e55"></span>Severe</div>
    <div class="class-item"><span class="class-dot" style="background:#9562f4"></span>Proliferate_DR</div>
    <div class="sidebar-divider"></div>
    <div class="sidebar-heading">▦ &nbsp; System Information</div>
    <div class="sidebar-description">
        Image Input: RGB<br>
        Image Processing: Enabled<br>
        Classification: 5 Categories
    </div>
    <div class="sidebar-note">
        🛡️ This tool is for research and educational purposes only. It is not a medical diagnosis.
    </div>
    """)

# Display main page header
render_html("""
<div class="main-header">
    <svg class="main-eye" viewBox="0 0 180 120" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 62 Q90 -15 168 62 Q90 139 12 62Z" fill="none" stroke="#102c58" stroke-width="9"/>
        <circle cx="90" cy="62" r="35" fill="#102c58"/>
        <circle cx="90" cy="62" r="13" fill="white"/>
        <path d="M12 42 Q90 -15 168 42" fill="none" stroke="#f04d52" stroke-width="8" stroke-linecap="round"/>
    </svg>
    <div class="main-title">Diabetic Retinopathy <span>Detection</span></div>
    <div class="main-subtitle">AI-Based Diabetic Retinopathy Severity Assessment</div>
    <div class="main-description">
        Upload a retinal fundus image to analyse and identify its diabetic retinopathy severity category.
    </div>
</div>
<div class="heartbeat"><span>⌁</span></div>
""")

# Upload section instructions
render_html("""
<div class="upload-heading">Upload Retinal Fundus Image</div>
<div class="upload-description">
    Drag and drop an image here or click to browse<br>
    Supported formats: JPG, JPEG, PNG
</div>
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload retinal fundus image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Feature highlights
st.markdown("<br>", unsafe_allow_html=True)
feature_columns = st.columns(4, gap="medium")
features = [
    ("🧠", "Intelligent Analysis", "Automated Image Assessment"),
    ("📈", "Confidence Scores", "Prediction Confidence"),
    ("⚙️", "Image Enhancement", "Image Quality Improvement"),
    ("🛡️", "5 Severity Classes", "Retinal Image Classification")
]
for column, feature in zip(feature_columns, features):
    icon, title, description = feature
    with column:
        render_html(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-text">{description}</div>
        </div>
        """)

# Process image and display analysis
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    current_file_id = (uploaded_file.name, uploaded_file.size)

    # Reset session state on new file upload
    if st.session_state.get("current_file_id") != current_file_id:
        st.session_state["current_file_id"] = current_file_id
        st.session_state.pop("prediction_class", None)
        st.session_state.pop("prediction_confidence", None)
        st.session_state.pop("prediction_probabilities", None)

    st.markdown("<br>", unsafe_allow_html=True)
    analyse_button = st.button("🔍  Analyse Retinal Image", use_container_width=True)

    if analyse_button:
        try:
            with st.spinner("Analysing retinal image..."):
                model = load_trained_model()
                processed_image = preprocess_image(image)
                predicted_class, confidence, probabilities = predict_severity(model, processed_image)
                st.session_state["prediction_class"] = predicted_class
                st.session_state["prediction_confidence"] = confidence
                st.session_state["prediction_probabilities"] = probabilities
            st.success("Image analysis completed successfully.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    st.markdown("<br>", unsafe_allow_html=True)
    image_col, result_col = st.columns([1, 1], gap="large")

    # Display uploaded retinal image
    with image_col:
        render_html("""
        <div class="section-title">
            Uploaded Retinal Image
        </div>
        """)
        st.image(image, caption="Uploaded retinal fundus image", use_container_width=True)
        safe_filename = html.escape(uploaded_file.name)
        render_html(f"""
        <div class="image-info">
            <div class="result-label">IMAGE INFORMATION</div>
            <p>File: {safe_filename}</p>
            <p>Dimensions: {image.width} × {image.height}</p>
            <p>Status: Image Ready</p>
        </div>
        """)

    # Display prediction results
    with result_col:
        render_html("""
        <div class="section-title">
            Prediction Result
        </div>
        """)

        if "prediction_class" in st.session_state:
            predicted_class = st.session_state["prediction_class"]
            confidence = st.session_state["prediction_confidence"]
            probabilities = st.session_state.get("prediction_probabilities")

            render_html(f"""
            <div class="result-card">
                <div class="result-label">PREDICTED SEVERITY CLASS</div>
                <div class="result-value">{html.escape(predicted_class)}</div>
                <div style="height:25px;"></div>
                <div class="result-label">PREDICTION CONFIDENCE</div>
                <div class="confidence-value">{confidence:.2f}%</div>
                <div style="height:15px;"></div>
                <div style="background:#e7f4ed; color:#168b55; padding:10px; border-radius:10px; font-size:13px; font-weight:700;">
                    Analysis Completed
                </div>
            </div>
            """)
        else:
            render_html("""
            <div class="waiting-card">
                <div class="waiting-icon">📊</div>
                <div class="waiting-title">Awaiting Prediction</div>
                <div class="waiting-text">Click the Analyse Retinal Image button to view the prediction result.</div>
            </div>
            """)

    # Display visualisation plots when analysis is available
    if "prediction_probabilities" in st.session_state:
        probabilities = st.session_state["prediction_probabilities"]
        st.markdown("<br>", unsafe_allow_html=True)
        render_html("""
        <div class="section-title">
            Visualisation Analysis
        </div>
        """)
        plot_col1, plot_col2 = st.columns([1, 1], gap="large")

        # First plot - Retinal fundus image
        with plot_col1:
            # Creating plot
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            ax1.imshow(image)
            ax1.axis('off')
            # Adding title
            plt.title('Retinal Fundus Image Visualisation')
            # Showing plot
            st.pyplot(fig1)

        # Second plot - Class prediction probabilities
        with plot_col2:
            # Creating plot
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            bar_colours = ['#27b45b', '#ffbd19', '#ff922e', '#ff4e55', '#9562f4']
            ax2.bar(CLASS_NAMES, probabilities * 100, color=bar_colours)
            ax2.set_ylabel('Probability (%)', fontsize=11)
            ax2.set_ylim(0, 100)
            # Adding title
            plt.title('Prediction Probabilities Across Classes')
            # Showing plot
            st.pyplot(fig2)

# Medical disclaimer notice
render_html("""
<div class="medical-notice">
    <b>Medical Disclaimer:</b>
    This application is intended for research and educational purposes only.
    Its predictions should not be used as a substitute for professional medical examination or diagnosis.
</div>
""")

# Application footer
render_html("""
<div class="footer">
    ━━━━━━━━━━━ &nbsp;
    Early Detection &nbsp; • &nbsp;
    Better Eye Health &nbsp; • &nbsp;
    A Healthier Tomorrow
    &nbsp; ━━━━━━━━━━━
</div>
""")
