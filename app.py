import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Placeholder for the app's state
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("BOOKS.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are here to help me find my next great read. ask me about the genres, authors, or types of stories I enjoy, and You'll recommend some books for me. Let's get started!"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response



demo = gr.Blocks()

with demo:
    gr.Markdown(
        "‼️Disclaimer: This document is utilized exclusively for implementing a Retrieval-Augmented Generation (RAG) chatbot.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
       examples=[
            ["I love mystery novels with a strong female lead."],
            ["Can you recommend some science fiction books?"],
            ["I'm looking for a good historical fiction novel."]

        ],
        title='📚 Personalized Book Recommendation Bot 📚',
    description='''<h2>Welcome to the Personalized Book Recommendation Bot!</h2>
                   <p>Tell me about your reading preferences, and I will suggest some books that you might enjoy.</p>
                   <p><strong>Examples:</strong></p>
                   <ul>
                       <li>I love mystery novels with a strong female lead.</li>
                       <li>Can you recommend some science fiction books?</li>
                       <li>I'm looking for a good historical fiction novel.</li>
                   </ul>''',
    )


if __name__ == "__main__":
    demo.launch()
