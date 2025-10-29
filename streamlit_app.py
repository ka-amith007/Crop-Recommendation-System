import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="🌱 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
    return model, sc, ms

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Main app
def main():
    # Load models
    model, sc, ms = load_models()
    
    # Header
    st.title("🌱 Crop Recommendation System")
    st.markdown("### Get AI-powered crop recommendations based on soil and climate conditions")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Soil Nutrients")
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    
    with col2:
        st.subheader("🌤️ Climate Conditions")
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)
    
    # Predict button
    if st.button("🔮 Get Crop Recommendation", type="primary", use_container_width=True):
        # Prepare features
        feature_list = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        
        # Make prediction
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)
        
        # Display result
        if prediction[0] in crop_dict:
            recommended_crop = crop_dict[prediction[0]]
            
            # Success message with emoji
            st.success(f"🎯 **Recommended Crop: {recommended_crop}**")
            st.balloons()
            
            # Additional info
            st.info(f"✅ {recommended_crop} is the best crop to cultivate with the provided conditions!")
            
            # Display input summary
            with st.expander("📊 Input Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Soil Nutrients:**")
                    st.write(f"• Nitrogen: {nitrogen}")
                    st.write(f"• Phosphorus: {phosphorus}")
                    st.write(f"• Potassium: {potassium}")
                    st.write(f"• pH: {ph}")
                
                with col2:
                    st.write("**Climate Conditions:**")
                    st.write(f"• Temperature: {temperature}°C")
                    st.write(f"• Humidity: {humidity}%")
                    st.write(f"• Rainfall: {rainfall}mm")
        else:
            st.error("❌ Sorry, we could not determine the best crop with the provided data.")

    # Sidebar with info
    with st.sidebar:
        st.image("static/img.jpg", caption="Crop Recommendation", use_column_width=True)
        st.markdown("### 📋 How to use:")
        st.markdown("""
        1. **Enter soil nutrient values** (N, P, K, pH)
        2. **Provide climate data** (Temperature, Humidity, Rainfall)
        3. **Click 'Get Recommendation'**
        4. **Get your AI-powered crop suggestion!**
        """)
        
        st.markdown("### 🌾 Supported Crops:")
        crops_list = list(crop_dict.values())
        for i in range(0, len(crops_list), 2):
            if i+1 < len(crops_list):
                st.write(f"• {crops_list[i]} • {crops_list[i+1]}")
            else:
                st.write(f"• {crops_list[i]}")

if __name__ == "__main__":
    main()