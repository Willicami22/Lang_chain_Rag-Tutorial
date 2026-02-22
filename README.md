# LangChain Tutorial for RAGs

A hands-on tutorial notebook that walks through building LangChain-powered agents — from a simple tool-calling agent to a full **Retrieval-Augmented Generation (RAG)** pipeline using OpenAI embeddings and an in-memory vector store. The project also demonstrates how to integrate **LangSmith** for observability and tracing.

---

## Table of Contents

- [Architecture & Components](#architecture--components)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing](#installing)
- [Running the Notebook](#running-the-notebook)
- [Examples](#examples)
- [Project Structure](#project-structure)

---

## Architecture & Components

The tutorial is organized into three major sections:

### 1. Basic Agent
A minimal LangChain agent that binds a custom tool (`get_weather`) to a GPT-4 model and processes user messages through a LangGraph state machine.

```
User Message → Agent (GPT-4) → Tool Call (get_weather) → Final Response
```

### 2. Real-World Agent
A more complete agent featuring:
- **Custom tools** — `get_weather_for_location` and `get_user_location` decorated with `@tool`
- **Typed context** — a `Context` dataclass carrying a `user_id` for per-user tool behavior
- **Structured response format** — a `ResponseFormat` dataclass that enforces a punny reply and optional weather details
- **In-memory conversation history** — via LangGraph's `InMemorySaver` checkpointer, keyed by `thread_id`
- **LangSmith tracing** — enabled via `LANGSMITH_TRACING=true`

```
User Message
    └─► Agent (gpt-4o-mini)
            ├─► get_user_location(context.user_id)
            ├─► get_weather_for_location(city)
            └─► ResponseFormat { punny_response, weather_conditions }
```

### 3. RAG Agent
A full Retrieval-Augmented Generation pipeline:

| Stage | Component | Detail |
|---|---|---|
| **Load** | `WebBaseLoader` + `BeautifulSoup4` | Fetches and parses a blog post |
| **Split** | `RecursiveCharacterTextSplitter` | 1,000-char chunks, 200-char overlap |
| **Embed & Store** | `OpenAIEmbeddings` + `InMemoryVectorStore` | `text-embedding-3-large` model |
| **Retrieve** | `retrieve_context` tool | Top-2 similarity search |
| **Generate** | `create_agent` + `gpt-4o-mini` | Answers grounded in retrieved context |

```
User Query
    └─► RAG Agent (gpt-4o-mini)
            └─► retrieve_context(query)
                    └─► InMemoryVectorStore.similarity_search(k=2)
                            └─► Grounded Answer
```

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- **Python 3.12+**
- **pip** or a compatible package manager
- A virtual environment tool (e.g., `venv`)
- API keys for:
  - [OpenAI](https://platform.openai.com/account/api-keys) — required for LLM and embeddings
  - [LangSmith](https://smith.langchain.com/) — required for tracing (optional but recommended)
  - [Pinecone](https://www.pinecone.io/) — required only if swapping out the in-memory vector store

```bash
python --version   # should be 3.12+
pip --version
```

### Installing

**Step 1 — Clone the repository**

```bash
git clone https://github.com/your-username/langchain-rag-tutorial.git
cd langchain-rag-tutorial
```

**Step 2 — Create and activate a virtual environment**

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

**Step 3 — Install dependencies**

```bash
pip install langchain langchain-openai langchain-anthropic python-dotenv \
            langchain-pinecone langchain-text-splitters langchain-community \
            bs4 langchain-core
```

Or from within the notebook, run the first cell directly:

```python
%pip install langchain langchain-openai langchain-anthropic python-dotenv \
             langchain-pinecone langchain-text-splitters langchain-community \
             bs4 langchain-core
```

**Step 4 — Configure environment variables**

Create a `.env` file at the root of the project:

```env
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls-...
PINECONE_API_KEY=...        # only if using Pinecone
```

The notebook loads these automatically with `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))   # verify it loaded correctly
```

---

## Running the Notebook

**Step 5 — Launch Jupyter**

```bash
jupyter notebook Langchain_Tutorial.ipynb
# or
jupyter lab Langchain_Tutorial.ipynb
```

**Step 6 — Run cells in order**

Execute cells from top to bottom. The notebook is divided into clearly marked sections:

1. **SetUp** — installs packages and loads env vars
2. **Build a basic agent** — confirms tool-calling works end-to-end
3. **Build a real-world agent** — multi-turn conversation with typed context and structured output
4. **Build a RAG Agent** — loads a web document, indexes it, and answers grounded questions

---

## Examples

### Basic Agent — Weather Query

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like in New York?"}]}
)
# → AIMessage: "The current weather in New York is sunny."
```

### Real-World Agent — Punny Weather Forecaster

```python
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather outside?"}]},
    config=config,
    context=Context(user_id="1"),
)

print(response["structured_response"])
# ResponseFormat(
#   punny_response="Looks like Florida is bringing the heat! It's always sunny here,
#                   so don't forget your shades and sunscreen!",
#   weather_conditions="sunny"
# )
```

### RAG Agent — Grounded Q&A

```python
query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

**Sample output:**

```
================================ Human Message =================================
What is the standard method for Task Decomposition? ...

================================== Ai Message ==================================
Tool Calls:
  retrieve_context (query: "standard method for Task Decomposition")
  retrieve_context (query: "common extensions of Task Decomposition")

================================= Tool Message =================================
Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Task decomposition can be done (1) by LLM with simple prompting ...

================================== Ai Message ==================================
### Standard Method for Task Decomposition
The standard method involves using Chain of Thought (CoT) prompting ...

### Common Extensions
1. Tree of Thoughts — explores multiple reasoning branches at each step ...
```

---

## Project Structure

```
langchain-rag-tutorial/
├── Langchain_Tutorial.ipynb   # Main tutorial notebook
├── .env                       # API keys (not committed to version control)
├── .venv/                     # Virtual environment (not committed)
└── README.md                  # This file
```

