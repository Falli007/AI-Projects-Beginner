#importing all the necessary libraries
# pip install SpeechRecognition
# pip install pyttsx3
#pip install pyaudio

import speech_recognition as sr # SpeechRecognition library for recognizing speech
import pyttsx3      # Text-to-speech library for converting text to speech
import datetime     # For getting the current date and time
import webbrowser   # For opening web pages in the default browser
import os          # For interacting with the operating system

# Initialise the recognizer and the text-to-speech engine
engine = pyttsx3.init()  # Initialize the text-to-speech engine

#to make the assistant speak
def speak(text): # Function to make the assistant speak
    engine.say(text) # Pass the text to the engine
    engine.runAndWait()  # Run the engine to speak the text
    
#to listen to the user's command
def take_command():
    recogniser = sr.Recognizer()
    with sr.Microphone() as source: # Use the microphone as the source
        print("Listening...")
        recogniser.adjust_for_ambient_noise(source) # Adjust for ambient noise. An am
        audio = recogniser.listen(source)  # Listen for the command
        
        try:
            print("Recognizing...")
            command = recogniser.recognize_google(audio, language='en-in')  # Recognize the command using Google Speech Recognition
            print(f"User said: {command}\n") # Return the recognized command
        except sr.UnknownValueError:   # If the command is not recognized
            print("Sorry, I did not understand that.")
            return "None"   # Return None if the command is not recognized
        except sr.RequestError:  # If there is an error with the request
            print("Sorry, there was an error with the request.")
            return "None"
        
    return command.lower()

#the function to respond to different commands
def respond(command):
    if 'hello' in command or 'hi' in command:
        speak("Hello! How can I assist you today?")
        
    elif 'time' in command:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        speak(f"The current time is {current_time}.")
    
    elif 'date' in command:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        speak(f"Today's date is {current_date}.")
        
    elif 'open youtube' in command:
        speak("Opening YouTube.")
        webbrowser.open("https://www.youtube.com")
    
    elif 'open google' in command:
        speak("Opening Google.")
        webbrowser.open("https://www.google.com")
        
    elif 'open stack overflow' in command:
        speak("Opening Stack Overflow.")
        webbrowser.open("https://stackoverflow.com")
        
    elif 'search' in command:
        speak("What would you like to search for?")
        search_query = take_command()
        if search_query != "None":
            speak(f"Searching for {search_query} on Google.")
            webbrowser.open(f"https://www.google.com/search?q={search_query}")
            
    elif 'bye' in command or 'exit' in command or 'quit' in command:
        speak("Goodbye! Have a great day!")
        exit()
        
    else:
        speak("I'm sorry, I didn't understand that. Can you please repeat?")
    
# Main loop to keep the assistant running
def run_assistant():
    speak("Hello! I am your personal assistant. How can I help you today?")
    
    while True:
        command = take_command()  # Take the user's command
        if command != "None":  # If the command is recognised
            respond(command)  # Respond to the command
            
# Run the assistant
if __name__ == "__main__":
    run_assistant()  # Start the assistant
    
        
            