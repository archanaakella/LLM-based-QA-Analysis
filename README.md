# LLM-based-QA-Analysis

# LLM-Powered Booking Analytics & QA System

# 📌 Overview

This project is an AI-powered booking analytics & question-answering system built using FastAPI, FAISS, and an open-source LLM. It processes hotel booking data, extracts insights, and enables Retrieval-Augmented Question Answering (RAG) for user queries.

# 🚀 Features

Hotel Booking Analytics: Revenue trends, cancellation rates, geographical distribution, lead time.

Natural Language Q&A: Answers queries using LLM (GPT-Neo, Llama 2, Falcon, Mistral, etc.).

Vector Database Integration: Stores booking data as embeddings using FAISS.

REST API: Built with FastAPI, supporting analytics & Q&A endpoints.

Health Check API: Verifies system status.

# 🛠 Tech Stack

Backend: FastAPI, Uvicorn

ML & NLP: FAISS, Sentence-Transformers, Transformers (GPT-Neo, etc.)

Data Processing: Pandas, NumPy

Database: CSV (can be extended to SQLite/PostgreSQL)

# 📂 Project Structure

├── app.py                # FastAPI application

├── hotel_bookings.csv     # Sample dataset

├── requirements.txt      # Dependencies

├── Dockerfile            # Containerization setup

├── README.md             # Project documentation

└── .gitignore            # Ignored files

# ⚡ Installation & Setup

1️⃣ Clone Repository

git clone https://github.com/yourusername/llm-booking-qa.git
cd llm-booking-qa

2️⃣ Create Virtual Environment (Recommended)

python -m venv fastapi_env
source fastapi_env/bin/activate  # (Linux/macOS)
fastapi_env\Scripts\activate  # (Windows)

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run FastAPI Server

uvicorn app:app --reload

API runs at: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs

# 📡 API Endpoints

Endpoint & Description

POST - Returns hotel booking analytics

POST - /ask - Answers booking-related questions using LLM

GET - /health - Returns API health status
