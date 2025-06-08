# Install if not already: pip install transformers torch
 
from transformers import GPT2LMHeadModel, GPT2Tokenizer # For text generation using GPT-2
import torch # For tensor operations and model inference
 
#To load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")   # Load the GPT-2 tokenizer 
model = GPT2LMHeadModel.from_pretrained("gpt2")     # Load the GPT-2 model
 
#To set model to evaluation mode
model.eval()   
 
#To provide a text prompt
prompt = "Once upon a time the people on earth moved to Mars,"   # Example prompt for text generation
input_ids = tokenizer.encode(prompt, return_tensors="pt")       # Encode the prompt into input IDs
attention_mask = torch.ones_like(input_ids)  # the attention mask to indicate which tokens should be attended to
 
#To generate continuation
with torch.no_grad():
    output = model.generate(
        input_ids,  # Input IDs for the model
        attention_mask = attention_mask,  # Attention mask to specify which tokens to attend to
        max_length=300,      # Maximum length of the generated text
        num_return_sequences=1, # Number of sequences to return
        no_repeat_ngram_size=2, # Prevent repetition of n-grams
        do_sample=True,        # Enable sampling for more diverse outputs
        top_k=50,       # Top-k sampling to limit the number of candidate tokens
        top_p=0.95,   # Nucleus sampling to limit the candidate tokens based on cumulative probability
        temperature=0.9,   # Temperature for controlling randomness in sampling
    )
 
#Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)   # Decode the output tensor to text
print("📝 Generated Text:\n")
print(generated_text)