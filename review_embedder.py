import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except ImportError:
    pass  # Not in Streamlit

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OpenAI API key not found. Set it in Streamlit secrets or as an environment variable.")

# Set your OpenAI API key 
#os.environ["OPENAI_API_KEY"] = ""  

# Embedder setup 
embedder = OpenAIEmbeddings()

# --- Load all text files from the reviews folder ---
def load_review_documents(folder="reviews"):
    docs = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".txt"):
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        doc = Document(page_content=content, metadata={"filename": fname})
                        docs.append(doc)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
    return docs

# Main logic 
if __name__ == "__main__":
    output_dir = "review_vectorstore"
    os.makedirs(output_dir, exist_ok=True)

    docs = load_review_documents("reviews")

    if not docs:
        print("⚠No valid text files found. Nothing embedded.")
    else:
        vectorstore = FAISS.from_documents(docs, embedder)
        vectorstore.save_local(output_dir)
        print(f"Saved FAISS vectorstore with {len(docs)} documents to `{output_dir}/`")




