from src.medrag import MedRAG
import ast
import re

document_checker = True
medrag = MedRAG(llm_name="Google/gemini-1.5-flash-exp-0827", retriever_name=None, document_checker=document_checker)

k = 100
cell_type = "CD8+ T Cells"
signal="IL-10"
snippet_path_100 = f'/mnt/scratch/MedRAG/abm-rules-MedCPT-k={k}/{cell_type}_{signal}_snippets.json'
import json
def load_json(fpath):
    with open(fpath, 'r') as j:
        return json.loads(j.read())
    
snippets = load_json(snippet_path_100)
#snippets = [snippets[1], snippets[2], snippets[5]]
def add_pmids(explanation, snippets):
    doc_refs = re.findall('Document \[[0-9][0-9]?\]', explanation)
    if len(doc_refs) > 0:
        for doc_ref in doc_refs:
            idx = int(doc_ref[doc_ref.index('[')+1:doc_ref.index(']')])
            try:
                explanation = explanation.replace(doc_ref, f"PMID={snippets[idx]['PMID']}")
            except IndexError:
                print(f"tried to retrieve document [{idx}] PMID, but got list out of range, num_snippets={len(snippets)}")
                print(explanation)
    return explanation
import time
for i in range(1):
    for n, snippet in enumerate(snippets):
        _, _, _ = medrag.answer(
            question="", 
            k=100, 
            document_checker=document_checker, 
            cell_type=cell_type, 
            signal=signal, 
            save_dir=f'document-check-MedCPT-doc={n}-flash/run-{i}',
            snippet=snippet)
        time.sleep(6)
        

