# Agentic RAG System with LangChain and Automated Evaluation

A production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, featuring automated quality assessment using LLM-as-a-Judge evaluation framework.

## 🎯 Project Overview

This project demonstrates an end-to-end agentic RAG pipeline that:
- Retrieves relevant context from a vector database
- Generates accurate, grounded responses using LLMs
- Evaluates system performance through automated synthetic testing

Built as part of NVIDIA's "Building RAG Agents with LLMs" course, this implementation showcases production-ready practices for deploying intelligent document chatbots.

## 🏗️ Architecture
```
User Query → Retriever Agent → Context Reordering → Generator Agent → Response
                ↓                                           ↓
          Vector Search (FAISS)                    LLM (Llama 3.1)
                                                            ↓
                                              LLM-as-a-Judge Evaluation
```

## ✨ Key Features

### 1. Agentic RAG Pipeline
- **Document Processing**: ArXiv paper ingestion with intelligent chunking strategies
- **Vector Search**: FAISS-based semantic retrieval with NVIDIA embeddings
- **Context Optimization**: Long-context reordering to prioritize most relevant documents
- **Dynamic Generation**: LLM-powered response generation grounded in retrieved context

### 2. LangChain Orchestration
- Modular runnable chains for retrieval and generation
- LangServe deployment for production API endpoints
- Streaming responses for real-time user experience

### 3. Automated Evaluation Framework
- **Synthetic Data Generation**: Agentic workflow creates test question-answer pairs from corpus
- **LLM-as-a-Judge**: Automated pairwise comparison between RAG outputs and baseline answers
- **Quantitative Metrics**: Preference scoring to measure system quality (achieved 60%+ score)

## 🛠️ Tech Stack

- **LLM Framework**: LangChain, LangServe
- **Models**: NVIDIA AI Endpoints (Llama 3.1, NV-Embed-v1)
- **Vector Database**: FAISS
- **API Framework**: FastAPI
- **Frontend**: Gradio

## 📁 Project Structure
```
├── 08_evaluation.ipynb          # RAG evaluation implementation
├── 09_langserve.ipynb           # LangServe deployment
├── server_app.py                # FastAPI server with endpoints
├── docstore_index/              # FAISS vector store
└── README.md
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install langchain langchain-nvidia-ai-endpoints gradio
pip install faiss-cpu langserve fastapi uvicorn
```

### Running the Server
```bash
python server_app.py
```

This starts the FastAPI server on `http://localhost:9012` with three endpoints:
- `/basic_chat` - Simple LLM interaction
- `/retriever` - Document retrieval agent
- `/generator` - Response generation agent

### Testing the RAG System
```python
from langserve import RemoteRunnable

# Use the retriever
retriever = RemoteRunnable("http://localhost:9012/retriever/")
docs = retriever.invoke("What is MRKL?")

# Use the generator
generator = RemoteRunnable("http://localhost:9012/generator/")
response = generator.invoke({
    "input": "What is MRKL?",
    "context": "MRKL systems combine language models with external tools..."
})
```

## 📊 Evaluation Results

The LLM-as-a-Judge evaluation framework compares the agentic RAG system against baseline answers:

- **Test Setup**: 8 synthetic question-answer pairs generated from document corpus
- **Evaluation Method**: Pairwise comparison using LLM judge
- **Results**: **60%+ preference score** - RAG agent responses preferred over limited-context baseline
- **Key Finding**: Retrieval-augmented approach significantly outperforms static context methods

## 🧪 How Evaluation Works

1. **Synthetic Test Generation**:
   - Sample random documents from vector store
   - Use LLM to generate realistic questions spanning multiple documents
   - Create baseline answers from limited context

2. **RAG Agent Testing**:
   - Feed same questions to full RAG pipeline
   - Retrieve relevant documents across entire corpus
   - Generate comprehensive answers

3. **Automated Judging**:
   - LLM judge compares RAG output vs. baseline
   - Scores based on accuracy, completeness, and consistency
   - Aggregate scores to assess system quality

## 💡 Key Learnings

- **Retrieval Quality Matters**: Increasing k from 5 to 8 documents improved answer comprehensiveness
- **Context Reordering**: LongContextReorder significantly improved model attention to relevant info
- **Prompt Engineering**: Explicit instructions for detailed, sourced answers boosted evaluation scores
- **Evaluation is Critical**: Systematic testing reveals gaps invisible during manual testing

## 🔧 Potential Improvements

- [ ] Implement query expansion for better retrieval recall
- [ ] Add re-ranking layer between retrieval and generation
- [ ] Experiment with different chunking strategies (semantic vs. fixed-size)
- [ ] Integrate conversation memory for multi-turn interactions
- [ ] Add guardrails for output validation and safety

## 📚 References

- [NVIDIA DLI: Building RAG Agents with LLMs](https://www.nvidia.com/en-us/training/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangServe](https://github.com/langchain-ai/langserve)
- [LLM-as-a-Judge Paper](https://arxiv.org/abs/2306.05685)

## 📝 License

This project was developed as part of NVIDIA Deep Learning Institute coursework.

## 🙏 Acknowledgments

Built as the capstone project for NVIDIA's "Building RAG Agents with LLMs" course, demonstrating production-ready RAG implementation and evaluation techniques.
```

---

# 2. GitHub Repository Description

**Short version (for the "About" section):**
```
Production RAG agent with LangChain, FAISS vector search, and automated LLM-as-a-Judge evaluation. Features agentic workflows, streaming responses via LangServe, and 60%+ quality score on synthetic benchmarks.
```

**Alternative (more concise):**
```
Agentic RAG system using LangChain + FAISS with automated evaluation framework. LangServe API deployment, LLM-as-a-Judge testing, NVIDIA AI endpoints.
```

**Topics/Tags to add:**
```
langchain
rag
llm
faiss
langserve
agents
nvidia
fastapi
vector-search
llm-evaluation
retrieval-augmented-generation
