import urllib.request
import os

# Karpathy's official HuggingFace repo for the 42M parameter TinyLlama
url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.pt"
save_path = "../SanityCheck/stories42M.pt"

print("Downloading stories42M.pt (this is a ~160MB file, it might take a minute)...")
os.makedirs("../SanityCheck", exist_ok=True)
urllib.request.urlretrieve(url, save_path)
print("Model downloaded successfully! You can now hit CHECK.")