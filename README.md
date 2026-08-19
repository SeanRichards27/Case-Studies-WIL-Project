# Case-Studies-WIL-Project

WIL Project Git Repository for RAG model

**Group ID:** 68
**Team Name:** Census Enthusiasts

### Group Members

* Sean Richards - s3605481
* Maxwell Repin - s3822965
* Ryan Sozanski - s3840047
* Riyaz Basha Shaik - s4129750
* Sushmit Sharad Kadam – s4184118  

---

## Walert Reproduction

The `notebooks/Walert_BM25_Reproduction.ipynb` notebook contains our preliminary reproduction of the Walert RAG workflow.

The notebook uses the original Walert test collection to:

* Load and inspect the Walert knowledge base and test questions
* Build a BM25 retrieval model over the 120 Walert passages
* Retrieve relevant passages for test questions
* Evaluate retrieval using NDCG@1, NDCG@3 and NDCG@5
* Compare retrieval performance for Known and Inferred questions
* Pass retrieved passages to a local Ollama LLM to demonstrate a basic end-to-end RAG workflow

The required Walert data files are included in:

`data/walert/`

### Running the Notebook

Install the required Python packages:

```bash
pip install pandas numpy matplotlib rank-bm25 requests
```

The retrieval and evaluation sections can then be run directly from the notebook.

For the final RAG generation section, install and run [Ollama](https://ollama.com/) and download the model used in the notebook:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before executing the LLM generation cells.

### Current Status

The Walert retrieval and evaluation workflow has been successfully reproduced using the original Walert data. A lightweight local RAG workflow using BM25 retrieval and `llama3.2:3b` through Ollama has also been tested.

This reproduction is being used to understand the Test-Driven RAG workflow before applying the approach to the group's selected project domain.
