#To install if not already: pip install transformers sentencepiece torch
 
from transformers import MarianTokenizer, MarianMTModel
 
# Define source and target language
src_lang = "en"   # English
tgt_lang = "fr"   # French
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
 
#To load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
 
#To input sentence in source language
text = "I love programming and machine learning."  # Example text to translate
 
#To tokenize and translate
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**inputs)
 
#To decode output
output = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"🌐 Original ({src_lang.upper()}): {text}")
print(f"🔁 Translated ({tgt_lang.upper()}): {output}")