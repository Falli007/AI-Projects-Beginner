#Importing the regular expression library
import re

#A dictonary to store predefined responses
responses = {
    "hello": "Hello! How can I assist you today?",
    "how are you": "I'm just a program, but thanks for asking! How can I help you?",
    "what is your name": "I am a chatbot created to assist you. You can call me Chatbot!",
    "what can you do": "I can answer your questions, provide information, and assist with various tasks. Just ask!",
    "thank you": "You're welcome! If you have any more questions, feel free to ask.",
    "bye": "Goodbye! Have a great day!",
    "help": "Sure! What do you need help with?",
    "default": "I'm sorry, I don't understand that. Can you please rephrase?"
}

#Function to get a response based on user input
def chatbot_response(user_input):
    # Normalize the input to lowercase
    user_input = user_input.lower()
    
    # Check for a predefined response
    for key in responses:
        if re.search(key, user_input):
            return responses[key]
        
    # If no predefined response is found, return the default response
    return responses["default"]

#Main function to run the chatbot
def run_chatbot():
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        # Exit condition if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        # Get the chatbot's response
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
        
# Run the chatbot
if __name__ == "__main__":
    run_chatbot()