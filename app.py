import streamlit as st
import joblib
import pandas as pd
import os
import requests
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sr
from gtts import gTTS
import uuid

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='te-IN')
        st.success(f"🗣 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("❌ Sorry, I didn't understand that.")
        return ""
    except sr.RequestError as e:
        st.error(f"⚠ Could not request results; {e}")
        return ""

# --- Voice Output Function ---
def speak_response(text, lang='te'):
    filename = f"response_{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    os.system(f"start {filename}")  # Use 'afplay' on Mac or 'xdg-open' on Linux

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- App Configuration ---
st.set_page_config(
    page_title="AgriBot - Farming Assistant",
    page_icon="🌾",
    layout="wide"
)
local_css("styles.css")

def add_hero():
    st.markdown("""
    <div class="hero">
        <h1 style="color:#2E8B57; margin-bottom:0.5rem;">🌾 AgriBot Pro</h1>
        <p style="font-size:1.1rem; color:#333;">Your intelligent farming assistant</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div class="footer">
        <p>© 2023 AgriBot Pro | Helping farmers make better decisions</p>
    </div>
    """, unsafe_allow_html=True)
# --- Constants ---
GROQ_API_KEY = "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"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


@st.cache_resource
def load_models():
    # Create model directory if needed
    os.makedirs('model', exist_ok=True)
    
    
    data = pd.read_csv('data/Crop_recommendation.csv')
    
    le_crop = LabelEncoder().fit(data['Crop'])
    le_state = LabelEncoder().fit(data['State_Name'])
    le_crop_type = LabelEncoder().fit(data['Crop_Type'])
    
    try:
        model = joblib.load('model/crop_recommendation_model.pkl')
    except:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        X = data[['State_Name', 'Crop_Type', 'N', 'P', 'K', 'pH', 'Rainfall', 'Temperature']]
        X['State_Name'] = le_state.transform(X['State_Name'])
        X['Crop_Type'] = le_crop_type.transform(X['Crop_Type'])
        y = le_crop.transform(data['Crop'])
        model.fit(X, y)
        joblib.dump(model, 'model/crop_recommendation_model.pkl')
    
    return le_crop, le_state, le_crop_type, model

label_encoder_crop, label_encoder_state, label_encoder_crop_type, model = load_models()

# Input value limits
limits = {
    "N": (10, 180),
    "P": (10, 125),
    "K": (10, 200),
    "pH": (3.00, 7.00),
    "rainfall": (3.00, 4000.00),
    "temperature": (1.00, 36.00),
}


#  Chatbot Function 
def get_agri_response(user_query):
    from langdetect import detect

    user_lang = detect(user_query)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    if user_lang == 'te':  # Telugu
        system_prompt = (
            "నీవు ఒక వ్యవసాయ నిపుణుడివి. వ్యవసాయ సంబంధిత ప్రశ్నలకు మాత్రమే సమాధానం ఇవ్వాలి. "
            "జీవాశ్రయ పద్ధతులు, ఎరువులు, పంటలు, నీటి వనరులు, నేల మరియు పంట సంరక్షణ గురించి తెలుగులో సమాధానం ఇవ్వండి. "
            "వ్యవసాయానికి సంబంధించినవే ప్రశ్నలు అడగమని వినయంగా చెప్పండి."
        )
    elif user_lang == 'hi':  # Hindi
        system_prompt = (
            "आप एक कृषि विशेषज्ञ हैं। कृपया केवल खेती, मिट्टी, फसलों, उर्वरकों, सिंचाई, कीट नियंत्रण और अन्य कृषि विषयों से जुड़े सवालों के ही उत्तर दें। "
            "उत्तर हिंदी में दें। यदि प्रश्न कृषि से संबंधित नहीं है तो विनम्रता से कहें: "
            "'मैं केवल कृषि संबंधी प्रश्नों के लिए यहाँ हूँ, कृपया खेती से जुड़ा प्रश्न पूछें।'"
        )
    else:
        system_prompt = (
            "You are AgriBot, an expert agricultural assistant. "
            "Only answer questions related to farming, soil, crops, fertilizers, irrigation, pest control, and other agriculture topics. "
            "If the question is outside of agriculture, politely respond: "
            "'I'm here to help with agriculture-related questions only. Please ask me something about farming.'"
        )

    body = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.5,
        "max_tokens": 600
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, json=body)
    result = response.json()
    reply = result["choices"][0]["message"]["content"].strip()
    return reply, user_lang

# --- App Layout --- 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Crop Prediction", "AgriBot Chat"])

# --- Page Routing ---
if page == "Home":
    st.title("🌾 AgriBot - Your Farming Assistant")

elif page == "Crop Prediction":
    st.title("🌱 Smart Crop Recommendation")
    st.write("Enter your farm details to get the best crop recommendation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        state = st.selectbox("State", sorted(label_encoder_state.classes_.tolist()))
        crop_type = st.selectbox("Crop Type", sorted(label_encoder_crop_type.classes_.tolist()))
        N = st.slider("Nitrogen (N) level", limits["N"][0], limits["N"][1], 50)
        P = st.slider("Phosphorus (P) level", limits["P"][0], limits["P"][1], 30)
        
    with col2:
        K = st.slider("Potassium (K) level", limits["K"][0], limits["K"][1], 40)
        ph = st.slider("Soil pH", float(limits["pH"][0]),float(limits["pH"][1]), 6.0,  step=0.1)
        rainfall = st.slider("Rainfall (mm)",float(limits["rainfall"][0]),float(limits["rainfall"][1]),1000.0, step=10.0 )
        temperature = st.slider("Temperature (°C)",float(limits["temperature"][0]),float(limits["temperature"][1]),25.0, step=0.5)
        
    if st.button("Recommend Crop"):
        try:
            encoded_state = label_encoder_state.transform([state])[0]
            encoded_crop_type = label_encoder_crop_type.transform([crop_type])[0]
            input_data = [[encoded_state, encoded_crop_type, N, P, K, ph, rainfall, temperature]]
            
            predicted_crop_index = model.predict(input_data)[0]
            predicted_crop = label_encoder_crop.inverse_transform([predicted_crop_index])[0]
            
            st.success(f"### Recommended Crop: {predicted_crop}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif page == "AgriBot Chat":
    st.title("🤖 AgriBot Chat Assistant")
    st.write("Ask me anything about farming, crops, soil, or agriculture!")

    user_input = st.text_area("Your question:", height=100)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("💬 Get Answer"):
            if user_input.strip():
                with st.spinner("Consulting with agricultural experts..."):
                    try:
                        reply, user_lang = get_agri_response(user_input)
                        st.success(reply)
                        lang_code = 'en'
                        if user_lang == 'hi':
                            lang_code = 'hi'
                        elif user_lang == 'te':
                            lang_code = 'te'
                        speak_response(reply, lang=lang_code)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter your question")

    with col2:
        if st.button("🎤 Ask via Voice"):
            user_query = get_voice_input()
            if user_query:
                with st.spinner("Consulting with agricultural experts..."):
                    try:
                        reply, user_lang = get_agri_response(user_query)
                        st.success(reply)
                        lang_code = 'en'
                        if user_lang == 'hi':
                            lang_code = 'hi'
                        elif user_lang == 'te':
                            lang_code = 'te'
                        speak_response(reply, lang=lang_code)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if page == "Home":
    add_hero()
    st.markdown("""
    <div style="padding:1.5rem; background-color:#f9f9f9; border-radius:10px; margin-top:1rem;">
        <h3 style="color:#2E8B57;">Welcome to AgriBot!</h3>
        <p>This application helps farmers with:</p>
        <ul style="margin-left:1.5rem;">
            <li>🌱 <strong>Crop recommendation</strong> based on soil and weather conditions</li>
            <li>🤖 <strong>AI-powered agricultural advice</strong> from our farming expert</li>
            <li>📊 <strong>Data-driven insights</strong> for better crop management</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="padding:1rem; background-color:#e8f5e9; border-radius:10px; margin:0.5rem 0; border-left:4px solid #2E8B57;">
            <h4 style="color:#2E8B57; margin-bottom:0.5rem;">🌦 Weather Analysis</h4>
            <p style="font-size:0.9rem;">Get crop recommendations based on current and forecasted weather patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding:1rem; background-color:#e8f5e9; border-radius:10px; margin:0.5rem 0; border-left:4px solid #2E8B57;">
            <h4 style="color:#2E8B57; margin-bottom:0.5rem;">🌱 Soil Health</h4>
            <p style="font-size:0.9rem;">Understand your soil's nutrient levels and get fertilization advice.</p>
        </div>
        """, unsafe_allow_html=True)
    
    add_footer()

elif page == "Crop Prediction":
    add_hero()
    st.markdown("""
    <div style="padding:1.5rem; background-color:#f9f9f9; border-radius:10px; margin-top:1rem; margin-bottom:1.5rem;">
        <h3 style="color:#2E8B57;">Smart Crop Recommendation</h3>
        <p>Enter your farm details to get the best crop recommendation based on machine learning analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    add_footer()

elif page == "AgriBot Chat":
    add_hero()
    st.markdown("""
    <div style="padding:1.5rem; background-color:#f9f9f9; border-radius:10px; margin-top:1rem; margin-bottom:1.5rem;">
        <h3 style="color:#2E8B57;">🤖 AgriBot Chat Assistant</h3>
        <p>Ask me anything about farming, crops, soil, or agriculture!</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    add_footer()           

# --- Run the App ---
if __name__ == '_main_':
    st.write("App is running!")
