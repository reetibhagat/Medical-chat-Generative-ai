import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    # Create model directory if it doesn't exist
    if not os.path.exists("model"):
        os.makedirs("model")
    
    # Download the smaller model (Mistral 7B quantized)
    model_url = "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf"
    model_path = "model/mistral-7b-v0.1.Q4_K_M.gguf"
    
    print("Downloading smaller model...")
    download_file(model_url, model_path)
    print("Download complete!")

if __name__ == "__main__":
    main() 