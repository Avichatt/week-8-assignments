# Week 08 Wednesday - CNNs + Embeddings

This repository contains the completion of the AI-ML Agentic AI Engineering assignment for Day 38 Wednesday focusing on CNN architectures and Semantic Embeddings.

## Setup Instructions

### Python Version
- Python 3.9 through 3.11 is strongly recommended to ensure `torch` compatibility.

### Packages Needed
Install the required dependencies using pip before running the notebook:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision sentence-transformers nbformat ipykernel
```

### How to Run
1. Ensure the dataset `social_media_posts.csv` exists in the local directory `week-08/wednesday/`. If not, run `python prepare_data.py` to synthesize it from the Kaggle archives.
2. Launch a Jupyter server:
   ```bash
   jupyter lab
   ```
3. Open `Week_08_Wednesday.ipynb`. The notebook strictly follows sub-steps 1 through 7, structured logically. Execute cells sequentially.
4. Note: AI Prompt usages and critiques are embedded directly within the designated markdown cells.

### Structure
- `prepare_data.py`: Prepares the dataset from requested input channels.
- `notebook_builder.py`: Synthesizes the assignment jupyter notebook.
- `social_media_posts.csv`: Processed output artifact mimicking the LMS requirements.
- `Week_08_Wednesday.ipynb`: Core execution module.
