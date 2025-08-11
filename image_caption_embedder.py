import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import torch

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except ImportError:
    pass  # Not in Streamlit

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OpenAI API key not found. Set it in Streamlit secrets or as an environment variable.")

#os.environ["OPENAI_API_KEY"] = ""  

# Load captioning model 
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Embedder setup 
embedder = OpenAIEmbeddings()

# Generate a caption for a single image 
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    input_ids = processor(text=[""], return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, input_ids=input_ids, max_length=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption

# Process all JPEGs in folder
def process_images(folder="images"):
    docs = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg")):
            path = os.path.join(folder, fname)
            try:
                caption = generate_caption(path)
                print(f"{fname}: {caption}")
                doc = Document(page_content=caption, metadata={"filename": fname})
                docs.append(doc)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    return docs

# Main logic 
if __name__ == "__main__":
    output_dir = "image_vectorstore"
    os.makedirs(output_dir, exist_ok=True)

    docs = process_images("images")

    if not docs:
        print("No valid images found or captions failed. Vector store not created.")
    else:
        vectorstore = FAISS.from_documents(docs, embedder)
        vectorstore.save_local(output_dir)
        print(f"Saved FAISS vectorstore to `{output_dir}/` with {len(docs)} images.")
