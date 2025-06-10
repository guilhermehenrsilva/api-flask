import google.generativeai as generativeai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
chave_secreta = os.getenv('GOOGLE_API_KEY')
print(chave_secreta)
generativeai.configure(api_key=chave_secreta)

csv_url = 'https://docs.google.com/spreadsheets/d/1pr9QbEENce-9NFrTtLdrfUd_swzQoOs0FDbShaqMECQ/export?format=csv'
df = pd.read_csv(csv_url)
print(df.head())

model = 'models/gemini-embedding-exp-03-07'
def gerarEmbeddings(title, text): 
  result = generativeai.embed_content(model=model,
                                content=text,
                                task_type="retrieval_document",
                                title=title)
  return result['embedding']


import time 
def gerarEmbeddingsComDelay(title, text, delay=10):
    time.sleep(delay)  # Atraso de 1 segundo
    return gerarEmbeddings(title, text)
df["Embeddings"] = df.apply(lambda row: gerarEmbeddingsComDelay(row["Titulo"],row["Conteúdo"]), axis=1)
print(df)

import pickle
pickle.dump(df, open('datasetEmbeddings2025.pkl','wb'))

modeloEmbeddings = pickle.load(open('datasetEmbeddings2025.pkl','rb'))
print(modeloEmbeddings)