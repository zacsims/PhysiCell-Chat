from src_.medrag import MedRAG
import ast
import re
import time
import json
import numpy as np

def load_json(fpath):
    with open(fpath, 'r') as j:
        return json.loads(j.read())
    
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





medrag = MedRAG(llm_name="Google/gemini-exp-1206", retriever_name=None)

k = 100
cell_types = ["CD8+ T Cells"]
signals = ["IL-10"]
snippet_path_100 = f'/mnt/scratch/MedRAG/abm-rules-MedCPT-k={k}/{cell_types[0]}_{signals[0]}_snippets.json'
 
snippets = load_json(snippet_path_100)
#snippets = [snippets[1], snippets[2]]
#snippets = [snippets[1]]
#snippets = [snippets[1], snippets[2], snippets[5]]
full_text_i = [1, 2, 6, 8, 10, 12, 32, 35, 38, 42, 43, 46, 49, 53, 54, 55, 58, 61, 64, 65, 66, 69, 76, 77, 78, 81, 83, 87, 89, 90, 93, 97]
#full_text_i = full_text_i[1:10]
full_text_i.remove(2)
full_text_i.insert(9,2)
print(full_text_i)
for k in range(1,11):
    i = 1
    for cell_type in cell_types:
        for signal in signals:
            query = f"{signal} {cell_type} triple negative breast cancer"
            _, _, _ = medrag.answer(
                question=query, 
                k=k, 
                abm=True, 
                cell_type=cell_type, 
                signal=signal, 
                other_cell_types = ", ".join([c for c in cell_types if c != cell_type]),
                other_signals = ", ".join([s for s in signals if s != signal]),
                save_dir=f'abm-rules-MedCPT-k={k}-2-0-flash-thinking/run-{i}',
                snippets=np.array(snippets)[full_text_i[:k]])
            time.sleep(10)
        

