from fastapi import FastAPI
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = FastAPI()

df = pd.read_csv("hotel_bookings.csv")

df['text_data'] = df.apply(lambda row: f"Hotel: {row['hotel']}, Country: {row['country']}, "
                                       f"ADR: {row['adr']}, Cancellations: {row['is_canceled']}", axis=1)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array([embedding_model.encode(text) for text in df['text_data']])

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


qa_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

@app.post("/analytics")
def get_analytics():
    """
    Returns analytics summary including revenue trends and cancellation rates.
    """
    total_revenue = df['adr'].sum()
    cancellation_rate = (df['is_canceled'].sum() / len(df)) * 100
    return {
        "total_revenue": total_revenue,
        "cancellation_rate": f"{cancellation_rate:.2f}%"
    }

@app.post("/ask")
def ask_question(question: str):

    question_embedding = embedding_model.encode(question).reshape(1, -1)
    

    distances, indices = index.search(question_embedding, k=5)
    
    retrieved_data = "\n".join(df.iloc[indices[0]]['text_data'].tolist())

    prompt = f"Context: {retrieved_data}\n\nQuestion: {question}\nAnswer:"
    response = qa_pipeline(prompt, max_length=100, do_sample=True)
    
    return {"answer": response[0]['generated_text']}

@app.get("/health")
def health_check():
    """
    API health check endpoint.
    """
    return {"status": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)