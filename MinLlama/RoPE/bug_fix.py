import urllib.request

# A working mirror of the original CMU repository's tensor data
url = "https://raw.githubusercontent.com/xiaobin-xs/minllama/master/rotary_embedding_actual.data"

print("Downloading correct tensor file...")
urllib.request.urlretrieve(url, "rotary_embedding_actual.data")
print("Valid file placed successfully! You can now hit CHECK.")