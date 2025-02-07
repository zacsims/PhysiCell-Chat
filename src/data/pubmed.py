import os
import gzip
import tqdm
import json

import requests
import ast 

def return_full_text(pmid):
    url = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC{pmid}/unicode'
    try:
        x = requests.get(url)
    except:
        return ""
    if x.text.startswith('No record can be found'):
        url = f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode'
        x = requests.get(url)
        if x.text.startswith('No record can be found'):
            #print(f'PMC{pmid} retrieval failed')
            return ''
    #    else:
            #print(f'PMC{pmid} retrieval success')
    #else:
        #print(f'PMC{pmid} retrieval success')
    try:
        data = ast.literal_eval(x.text)[0]
    except ValueError:
        #print('error extracting text')
        return ''
    except SyntaxError:
        #print('error extracting text')
        return ''
        
    num_passages = len(data['documents'][0]['passages'])
    
    full_text = ''.join([data['documents'][0]['passages'][i]['text'] for i in range(num_passages)])
    return full_text


def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

def extract(gz_fpath):
    titles = []
    abstracts = []
    title = ""
    abs = ""
    ids = []

    for line in gzip.open(gz_fpath, 'rt').read().split('\n'):
        if line.strip() == "<Article>" or line.strip().startswith("<Article "):
            title = ""
            abs = ""
        elif line.strip() == "</Article>":
            if abs.strip() == "":
                continue
            titles.append(title)
            abstracts.append(abs)
            ids.append(id)
        if line.strip().startswith("<PMID"):
            id = line.strip().strip("</PMID>").split(">")[-1]        
        if line.strip().startswith("<ArticleTitle>"):
            title = line.strip()[14:-15]
        if line.strip().startswith("<AbstractText"):
            if len(abs) == 0: 
                abs += "".join(line.strip()[13:-15].split('>')[1:])
            else:
                abs += " "
                abs += "".join(line.strip()[13:-15].split('>')[1:])

    return titles, abstracts, ids

if __name__ == "__main__":
    fnames = sorted([fname for fname in os.listdir("/mnt/scratch/MedRAG/corpus/pubmed/baseline") if fname.endswith("xml.gz")])
    
    if not os.path.exists("/mnt/scratch/MedRAG/corpus/pubmed/chunk"):
        os.makedirs("/mnt/scratch/MedRAG/corpus/pubmed/chunk")

    for fname in tqdm.tqdm(fnames):
        if os.path.exists("/mnt/scratch/MedRAG/corpus/pubmed/chunk/{:s}".format(fname.replace(".xml.gz", ".jsonl"))):
            print('already exists')
            continue
        gz_fpath = os.path.join("/mnt/scratch/MedRAG/corpus/pubmed/baseline", fname)
        try:
            titles, abstracts, ids = extract(gz_fpath)
        except:
            print("error", fname)
            continue
        #full_texts = [return_full_text(id_) for id_ in tqdm.tqdm(ids)]
        #saved_text = [json.dumps({"id": "PMID:"+str(ids[i]), "title": titles[i], "content": abstracts[i], "contents": full_texts[i], "full_text":full_texts[i]}) for i in range(len(titles)) if full_texts[i] != ""]
        saved_text = [json.dumps({"id": "PMID:"+str(ids[i]), "title": titles[i], "content": abstracts[i], "contents": concat(titles[i], abstracts[i])}) for i in range(len(titles))]
        with open("/mnt/scratch/MedRAG/corpus/pubmed/chunk/{:s}".format(fname.replace(".xml.gz", ".jsonl")), 'w') as f:
            print(f'writing {"/mnt/scratch/corpus/pubmed/chunk/{:s}".format(fname.replace(".xml.gz", ".jsonl"))}')
            f.write('\n'.join(saved_text))