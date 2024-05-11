# wikipedia-2023-11-embed-multilingual-v3

This repo contains scripts compatible with 'Cohere/wikipedia-2023-11-embed-multilingual-v3'

Currently restricting to indonesian dataset, but will expand to optionally
download any or all of the 300+ languages as well.


# Description

Description from [huggingface
page](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3):

```
The wikipedia-2023-11-embed-multilingual-v3 dataset contains the
wikimedia/wikipedia dataset dump from 2023-11-01 from Wikipedia in all 300+
languages.

The individual articles have been chunked and embedded with the state-of-the-art
multilingual Cohere Embed V3 embedding model. This enables an easy way to
semantically search across all of Wikipedia or to use it as a knowledge source
for your RAG application. In total is it close to 250M paragraphs / embeddings.

You can also use the model to perform cross-lingual search: Enter your search
query in any language and get the most relevant results back.
```
