def rrf_fusion(lexical_ids, semantic_ids, k=60, top_n=5):
    scores = {}
    
    for rank, doc_id in enumerate(lexical_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        
    for rank, doc_id in enumerate(semantic_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)        

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]