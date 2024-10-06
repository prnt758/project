import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import google.generativeai as genai
import playsound

# Configure Google Generative AI
GOOGLE_API_KEY = "AIzaSyA5HtRnzGruiia-aKtMMLnBjJ0ovTh11nE"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Initialize recognizer
recognizer = sr.Recognizer()

# Supported languages for recognition and TTS
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Telugu': 'te'
}

# Function to listen for user's voice input in selected language
def listen(language_code='te'):
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=20, phrase_time_limit=10)
    
    try:
        # Recognize speech in the specified language
        text = recognizer.recognize_google(audio, language=language_code)
        st.write(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I didn't catch that.")
        return None
    except sr.RequestError as e:
        st.write(f"Error with speech recognition: {e}")
        return None
    except Exception as e:
        st.write(f"Unexpected error: {e}")
        return None

# Text-to-Speech function for multiple languages
def speak(text, lang='en', whisper=False):
    try:
        # Convert text to speech with gTTS
        tts = gTTS(text=text, lang=lang, slow=whisper)
        filename = "temp_audio.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
    except Exception as e:
        st.write(f"An error occurred while speaking: {e}")

# Function to process the command through the AI model
def process_command(command, whisper_mode=False):
    if command:
        try:
            # Pre-prompt for AstroBio assistant
            pre_prompt = (
                "You are Aira, a friendly assistant specialized in space bio-experiments. "
                "You explain things in short and simple terms, always keeping a friendly tone. "
                "You are assisting a user in space biology research."
            )
            full_command = f"{pre_prompt} User asked: {command}"
            
            # Generate the response from Google Generative AI
            response = model.generate_content(full_command)
            return response.text
        except Exception as e:
            return f"Sorry, there was an error: {e}"
    return "I couldn't understand that."

# Main function to run the assistant
def main():
    st.title("Aira Smart Voice Assistant")

    # Language selection
    lang_choice = st.selectbox("Choose your language", options=LANGUAGES.keys())

    # Whisper mode toggle
    whisper_mode = st.checkbox("Activate Whisper Mode")

    st.write(f"Selected Language: {lang_choice} | Whisper Mode: {'On' if whisper_mode else 'Off'}")

    use_voice = st.checkbox("Use Voice Input")

    if use_voice:
        if st.button("Start Listening"):
            language_code = LANGUAGES[lang_choice]
            command = listen(language_code)
            if command:
                response = process_command(command, whisper_mode)
                st.write("AstroBio:", response)
                speak(response, lang=language_code, whisper=whisper_mode)
    else:
        command = st.text_input("Enter your command:")
        if st.button("Submit"):
            response = process_command(command, whisper_mode)
            st.write("AstroBio:", response)
            language_code = LANGUAGES[lang_choice]
            speak(response, lang=language_code, whisper=whisper_mode)

    if st.button("Exit"):
        st.write("Goodbye! Feel free to ask for help anytime.")
        st.stop()

if __name__ == "__main__":
    main()