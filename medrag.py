from src.medrag import MedRAG
import ast
import re

medrag = MedRAG(llm_name="Google/gemini-1.5-flash-latest", rag=True, retriever_name="RRF-2", corpus_name="PubMed")

options_mc = {
    "A": "increase",
    "B": "decrease",
    "C": "no change",
    "D": "no relevant information"
}


options_yn = {
    "A": "Yes: the evidence supports the rule",
    "B": "No: the evidence does not support the rule",
    "C": "No relevant information"
}

cell_types = ['cancer associated fibroblasts (CAFs)', 'Macrophages', 'Monocytes', 'CD4+ T Cells', 'CD8+ T Cells', 'Neoplastic Epithelium']
signals = ['Ncam1',
 'Col4a2',
 'Thbs3',
 'Vegfb',
 'Ccl6',
 'ITGA4_ITGB1',
 'Tnxb',
 'Lamb1',
 'Lama2',
 'Col9a3',
 'Lgals9',
 'Osm',
 'Pdgfa',
 'Sema3d',
 'Tgfb2',
 'Icam1',
 'Ccl2',
 'Ccl3',
 'Tnn',
 'Postn',
 'Tnc',
 'ITGAV_ITGB1',
 'Cdh1',
 'Vegfa',
 'Ccl9',
 'App',
 'Lamb2',
 'Ccl7',
 'Pros1',
 'Ccl8',
 'Ptprc',
 'Hspg2',
 'Tnf',
 'Nectin3',
 'Wnt5a',
 'Sema5a',
 'Tgfb3',
 'Fgf7',
 'Col6a2',
 'Sema3c',
 'Cxcl12',
 'Ptn',
 'Cxcl16',
 'Nampt',
 'Cd55',
 'Igf1',
 'Lama4',
 'C3',
 'Mdk',
 'Ccl12',
 'Csf1',
 'Igfbp3',
 'Thbs2',
 'F11r',
 'Col6a1',
 'Thy1',
 'Angptl4',
 'Efnb2',
 'Jam3',
 'Sdc2',
 'Cldn3',
 'Apoe',
 'Col1a1',
 'Col4a5',
 'Agrn',
 'Cadm1',
 'Mpzl1',
 'Efna5',
 'Angptl2',
 'Lair1',
 'Hbegf',
 'Sema3b',
 'Cd6',
 'Fn1',
 'Lpar1',
 'Jag1',
 'Gja1',
 'Gas6',
 'Areg',
 'Tenm3',
 'Sdc1',
 'Efnb1',
 'Cdh2',
 'Col4a1',
 'Thbs1',
 'Lamb3',
 'Mif',
 'Flrt2',
 'Ccl4',
 'Tnfsf12',
 'Tgfb1',
 'Lamc1',
 'Kitl',
 'Spp1',
 'Col6a3',
 'Cd86',
 'Col1a2'] #pressure, volume
behaviors = ['proliferation', 'apotisis', 'necrosis','phagocytosis']
#behaviors = ['apoptosis','necrosis','migration bias', 'migration persistence time', 'motility speed', 'phagocytosis','proliferation']
#separate rules from explanation
#after checking output

with open("results.csv", "w") as f:
    f.write("cell type, signal, behavior, answer_mc_k=5, explanation_mc_k=5, answer_mc_k=50, explanation_mc_k=50, answer_yn_increase, explanation_yn_increase, answer_yn_decrease, explanation_yn_decrease, \n")

def get_json(answer):
    try:
        return ast.literal_eval(answer[answer.index("{"):answer.index("}")+1])
    except SyntaxError:
        return ast.literal_eval(answer[answer.index("{"):answer.index("}")+1].replace('"',"'"))
                            
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
    
                            
for cell_type in cell_types:
    for signal in signals:
        for behavior in behaviors:
            try:
                print(cell_type, signal, behavior)
                pmids = set()
                question_mc = f"Does uptake of {signal} increase, decrease, or cause no change in {behavior} in {cell_type}?"

                answer_mc_5, snippets_mc_5, scores = medrag.answer(question=question_mc, options=options_mc, k=5)
                answer_mc_5 = get_json(answer_mc_5)
                answer_mc_5['explanation'] = add_pmids(answer_mc_5['explanation'], snippets_mc_5)
                for snippet in snippets_mc_5:
                    pmids.add(snippet["PMID"])

                answer_mc_50, snippets_mc_50, scores = medrag.answer(question=question_mc, options=options_mc, k=50)
                answer_mc_50 = get_json(answer_mc_50)
                answer_mc_50['explanation'] = add_pmids(answer_mc_50['explanation'], snippets_mc_50)
                for snippet in snippets_mc_50:
                    pmids.add(snippet["PMID"])
                    
                #answer_mc_50 ={"answer":"na", "explanation":"na"}

                question_yn_inc = f"{signal} increases {behavior} in {cell_type}"
                answer_yn_inc, snippets_yn_inc, scores = medrag.answer(question=question_yn_inc, options=options_yn, k=5, mc=False)
                answer_yn_inc = get_json(answer_yn_inc)
                answer_yn_inc['explanation'] = add_pmids(answer_yn_inc['explanation'], snippets_yn_inc)
                for snippet in snippets_yn_inc:
                    pmids.add(snippet["PMID"])

                question_yn_dec = f"{signal} decreases {behavior} in {cell_type}"
                answer_yn_dec, snippets_yn_dec, scores = medrag.answer(question=question_yn_dec, options=options_yn, k=5, mc=False)
                answer_yn_dec = get_json(answer_yn_dec)
                answer_yn_dec['explanation'] = add_pmids(answer_yn_dec['explanation'], snippets_yn_dec)
                for snippet in snippets_yn_dec:
                    pmids.add(snippet["PMID"])

                with open("results.csv", "a") as f:
                    output = f'{cell_type},{signal},{behavior},{answer_mc_5["answer"]},"{answer_mc_5["explanation"].replace(",","")}",{answer_mc_50["answer"]},"{answer_mc_50["explanation"].replace(",","")}",{answer_yn_inc["answer"]},"{answer_yn_inc["explanation"].replace(",","")}",{answer_yn_dec["answer"]},"{answer_yn_dec["explanation"].replace(",","")}"\n'
                    f.write(output)
            except Exception as e:
                print(e)
                print('failed')
                print()

