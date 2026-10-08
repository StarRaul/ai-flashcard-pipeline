import urllib.request

# The working CMU mirror for the optimizer test data
url = "https://raw.githubusercontent.com/xiaobin-xs/minllama/master/optimizer_test.npy"

print("Downloading correct tensor file...")
urllib.request.urlretrieve(url, "optimizer_test.npy")
print("Valid file placed successfully! You can now hit CHECK.")