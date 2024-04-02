
# LongChatGPT

Run conversations with the chatGPT API. Include summaries and a vector memory store of the conversation to improve long conversations.

# Installation

``` bash
conda create -n longchatgpt -c conda-forge faiss-cpu
conda activate longchatgpt
pip install -r requirements.txt
mkdir memory
```

# Running

``` bash 
conda activate longchatgpt
python app.py
```
