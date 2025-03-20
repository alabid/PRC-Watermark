import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def text_to_bigram_image(paragraph):
    # Tokenize the paragraph into words
    words = paragraph.lower().split()
    
    # Create bigrams (pairs of consecutive words)
    bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]
    
    # Count occurrences of each bigram
    bigram_counts = Counter(bigrams)
    
    # Get the unique words to define the axes
    unique_words = sorted(set(words))
    word_to_index = {word: i for i, word in enumerate(unique_words)}

    # Create a 2D tensor (bigram frequency matrix)
    vocab_size = len(unique_words)
    bigram_matrix = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)

    # Fill the matrix with bigram frequencies
    for (word1, word2), count in bigram_counts.items():
        i, j = word_to_index[word1], word_to_index[word2]
        bigram_matrix[i, j] = count

    # Normalize matrix to range [0, 255] for image representation
    if bigram_matrix.max() > 0:
        bigram_matrix = (bigram_matrix / bigram_matrix.max()) * 255

    bigram_matrix = bigram_matrix.to(torch.uint8)

    # Convert to PIL Image
    img = Image.fromarray(bigram_matrix.numpy(), mode='L')

    return img, unique_words  # Return the image and unique words for reference

# Example paragraph
paragraph = """A young traveler wandered through a dense forest, searching for adventure. 
The trees whispered secrets as the wind rustled their leaves. Birds chirped melodies, 
creating a peaceful harmony. Suddenly, the traveler stumbled upon an ancient bridge 
covered in vines. Cautiously, they stepped forward, feeling the wooden planks creak 
beneath them. A river flowed below, reflecting the golden sunset. Across the bridge, 
a hidden village emerged, glowing with lanterns. Curious, the traveler entered, 
greeted by kind villagers offering warm meals. Stories of forgotten legends filled 
the air. The journey had just begun, leading to mysteries, friendships, and endless discoveries ahead."""

# Generate the bigram frequency image
image, word_list = text_to_bigram_image(paragraph)

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray', aspect='auto')
plt.xticks(ticks=range(len(word_list)), labels=word_list, rotation=90, fontsize=8)
plt.yticks(ticks=range(len(word_list)), labels=word_list, fontsize=8)
plt.xlabel("Next Word in Bigram")
plt.ylabel("First Word in Bigram")
plt.title("Bigram Frequency Matrix")
plt.colorbar(label="Frequency")
plt.savefig('bigram_frequency_image_with_labels.png')

# Save the image
image.save("bigram_frequency_image.png")

