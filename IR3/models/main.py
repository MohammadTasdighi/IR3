import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from preprocess import clean_text
from embeddings import get_embedding
from vector_db import setup_db, upload_to_qdrant, qdrant_client
from search_engines import rrf_fusion
from sentiment import SentimentAnalyzer


ES_INDEX = "digikala_comments"
Q_COLLECTION = "digikala_comments"
es = Elasticsearch("http://localhost:9200")

def main():

    print("--- مرحله ۱: بارگذاری داده‌ها ---")
    df = pd.read_csv("data/digikala-comments.csv")

    df = df.head(1000).copy()

    if 'id' not in df.columns:
        df['id'] = range(len(df))


    print("--- مرحله ۲: پیش‌پردازش و تولید بردارها (ParsBERT) ---")
    df['clean_body'] = df['body'].fillna("").apply(clean_text)

    all_embeddings = np.array([get_embedding(text) for text in df['clean_body']])


    print("--- مرحله ۳: ایندکس‌گذاری در Elasticsearch و Qdrant ---")
    setup_db(Q_COLLECTION)
    payloads = df[['id', 'body', 'rate']].to_dict('records')
    upload_to_qdrant(Q_COLLECTION, all_embeddings, payloads)

    for _, row in df.iterrows():
        es.index(index=ES_INDEX, id=str(row['id']), document={'body': row['clean_body']})


    print("--- مرحله ۴: آموزش مدل تحلیل احساسات (استراتژی اول: Feature Extraction) ---")

    train_df = df[df['rate'] != 3].copy()
    train_indices = train_df.index.tolist()
    X = all_embeddings[train_indices]
    y = (train_df['rate'] >= 4).astype(int) 

    sa = SentimentAnalyzer()
    sa.train(X, y)


    y_pred = sa.model.predict(X)
    print("\n[گزارش ارزیابی مدل احساسات]")
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


    def run_full_search(query):
        print(f"\n" + "="*60)
        print(f"جستجو برای پرس‌وجوی: '{query}'")
        print("="*60)


        cleaned_q = clean_text(query)
        res_es = es.search(index=ES_INDEX, query={"match": {"body": cleaned_q}}, size=10)
        lexical_ids = [int(hit['_id']) for hit in res_es['hits']['hits']]

        print("\n[۱] نتایج جستجوی واژگانی (Elasticsearch):")
        for i, idx in enumerate(lexical_ids[:5], 1):
            print(f"   {i}. {df[df['id']==idx]['body'].values[0][:70]}...")


        q_emb = get_embedding(cleaned_q)
        res_q = qdrant_client.search(collection_name=Q_COLLECTION, query_vector=q_emb.tolist(), limit=10)
        semantic_ids = [hit.id for hit in res_q]

        print("\n[۲] نتایج جستجوی معنایی (Qdrant):")
        for i, idx in enumerate(semantic_ids[:5], 1):
            print(f"   {i}. {df[df['id']==idx]['body'].values[0][:70]}...")


        final_results = rrf_fusion(lexical_ids, semantic_ids, k=60, top_n=5)

        print("\n[۳] نتایج نهایی جستجوی ترکیبی (RRF Fusion) + تحلیل احساسات:")
        for rank, (doc_id, score) in enumerate(final_results, 1):
            text = df[df['id'] == doc_id]['body'].values[0]

            doc_idx = df[df['id'] == doc_id].index[0]
            sentiment_label = sa.predict(all_embeddings[doc_idx])
            
            print(f"   {rank}. [{sentiment_label}] (RRF Score: {score:.4f})")
            print(f"      متن: {text[:100]}...")

    run_full_search("کیفیت پایین دوربین در شب")
    run_full_search("تاخیر در ارسال و برخورد بد مامور پست")

if __name__ == "__main__":
    main()