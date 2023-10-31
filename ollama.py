#!/usr/bin/env python
# coding: utf-8
import json
from langchain.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="mistral")
with open("lines.txt") as fi:
    for text in fi:
        query_result = embeddings.embed_query(text)
        print(json.dumps(dict(text=text,vector=query_result)))
