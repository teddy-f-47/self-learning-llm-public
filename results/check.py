from unidecode import unidecode
from rake_nltk import Rake
import pandas as pd
import statistics
import pickle
import nltk
import os


nltk.download('stopwords')
nltk.download('punkt')
os.makedirs("dataset_analysis_dump", exist_ok=True)

class KWScorer():
  def __init__(self):
    self.rake = Rake()

  def get_keywords_from_a_text(self, text: str, limit: int = 0):
    self.rake.extract_keywords_from_text(unidecode(text.lower()))
    if limit > 0:
      return self.rake.get_ranked_phrases()[:limit]
    return self.rake.get_ranked_phrases()

  def evaluate_collected_document_relevance_to_prompt(self, prompt: str, document: str):
    prompt_kws = self.get_keywords_from_a_text(prompt)
    document_kws = self.get_keywords_from_a_text(document)
    matching_kws = list(set(prompt_kws).intersection(set(document_kws)))
    return len(matching_kws) / len(prompt_kws)

kw_scorer = KWScorer()

with open("res_Intel_neural-chat-7b-v3-3_OpenGen.pickle", "rb") as filehandle:
  res = pickle.load(filehandle)
qh = res["prompts_with_hallucination"]

with open("ds_Intel_neural-chat-7b-v3-3_open.pickle", "rb") as filehandle:
  ds = pickle.load(filehandle)

print("len(qh)", len(qh))
print("len(ds)", len(ds))
print(ds[0].keys())

all_prompt = [x["prompt"] for x in ds]
all_answer_text = [x["answer_text"] for x in ds]
all_answer_score = [x["answer_score"] for x in ds]
all_full_text = [x["full_text"] for x in ds]

all_kw_score = [kw_scorer.evaluate_collected_document_relevance_to_prompt(p, d) for p, d in zip(all_prompt, all_full_text)]

df = pd.DataFrame(ds)
df['kw_score'] = all_kw_score
sorted_df = df.sort_values(['prompt', 'answer_score'], ascending=False)

print("ALL")
print(len(sorted_df.groupby('prompt').mean(numeric_only=True)))
print(statistics.fmean(sorted_df.groupby('prompt').mean(numeric_only=True)['answer_score'].values.tolist()))
print(statistics.fmean(sorted_df.groupby('prompt').mean(numeric_only=True)['kw_score'].values.tolist()))

top_3 = sorted_df.groupby('prompt').head(3)
print("TOP-3")
print(len(top_3.groupby('prompt').mean(numeric_only=True)))
print(statistics.fmean(top_3.groupby('prompt').mean(numeric_only=True)['answer_score'].values.tolist()))
print(statistics.fmean(top_3.groupby('prompt').mean(numeric_only=True)['kw_score'].values.tolist()))

top_1 = sorted_df.groupby('prompt').head(1)
print("TOP-1")
print(len(top_1.groupby('prompt').mean(numeric_only=True)))
print("hard questions")
print(len(top_1.groupby('prompt').filter(lambda x: x['answer_score'] < 0.5)))
print(statistics.fmean(top_1.groupby('prompt').filter(lambda x: x['answer_score'] < 0.5)['answer_score'].values.tolist()))
print(statistics.fmean(top_1.groupby('prompt').filter(lambda x: x['answer_score'] < 0.5)['kw_score'].values.tolist()))
print("easy questions")
print(len(top_1.groupby('prompt').filter(lambda x: x['answer_score'] >= 0.5)))
print(statistics.fmean(top_1.groupby('prompt').filter(lambda x: x['answer_score'] >= 0.5)['answer_score'].values.tolist()))
print(statistics.fmean(top_1.groupby('prompt').filter(lambda x: x['answer_score'] >= 0.5)['kw_score'].values.tolist()))
print("both")
print(statistics.fmean(top_1.groupby('prompt').mean(numeric_only=True)['answer_score'].values.tolist()))
print(statistics.fmean(top_1.groupby('prompt').mean(numeric_only=True)['kw_score'].values.tolist()))
