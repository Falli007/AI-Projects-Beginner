# Install if not already: pip install torch torchvision transformers openai-clip matplotlib
#to install clip pip install git+https://github.com/openai/CLIP.git

 
import torch # For tensor operations and model inference
import clip # CLIP model for image and text processing
from PIL import Image # For image loading and preprocessing
import matplotlib.pyplot as plt # For displaying images
import seaborn as sns # For creating visualizations
import numpy as np
 
# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available, otherwise CPU
model, preprocess = clip.load("ViT-B/32", device=device)  # Load the CLIP model and preprocessing function
 
# Load and preprocess image
image_path = "DamiandElizabeth.jpeg"  # Replace with your own image
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)  # Preprocess the image and add batch dimension
 
# Candidate text descriptions
text_descriptions = [
    "Are they in love?",
    "They are both shy to kiss",
    "Two couples in love",
    "A selfie of a couple",
    "They love each other",
    "A romantic moment captured",
]
 
# Tokenize text inputs
text_tokens = clip.tokenize(text_descriptions).to(device)
 
# Encode image and text with CLIP
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
 
    # Compute cosine similarity
    logits_per_image, _ = model(image, text_tokens)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 

# Print matched results
print("🧠 Top Text Matches for Image:")
for i, (desc, prob) in enumerate(zip(text_descriptions, probs[0])):
    print(f"{i+1}. {desc} – {prob:.2%}")
    
# Print best match
top_index = probs[0].argmax()
top_match = text_descriptions[top_index]
confidence = probs[0][top_index]
print(f"\n💡 Best Match: '{top_match}' with confidence {confidence:.2%}")
    
# Sort results for better chart visualization
sorted_indices = np.argsort(probs[0])[::-1]
sorted_descs = [text_descriptions[i] for i in sorted_indices]
sorted_probs = [probs[0][i] for i in sorted_indices]

# Plot: image + sorted bar chart
fig, axs = plt.subplots(1, 2, figsize=(14, 6))


# Left: image
axs[0].imshow(Image.open(image_path))
axs[0].axis("off")
axs[0].set_title("Input Image")

# Right: bar chart
sns.barplot(x=sorted_probs, y=sorted_descs, ax=axs[1])
axs[1].set_title("CLIP Match Confidence (Sorted)")
axs[1].set_xlabel("Probability")
axs[1].set_xlim(0, 1)

plt.tight_layout()

# to Save output to file
fig.savefig("clip_match_results.png", dpi=300)

plt.show()