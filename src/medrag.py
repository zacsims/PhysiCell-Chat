import os
import re
import json
import tqdm
import time
import argparse
import tiktoken
import sys
sys.path.append("src")
from template import *
import google.generativeai as genai



class MedRAG:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", retriever_name="MedCPT", corpus_name="PubMed", db_dir="./corpus", cache_dir=None, document_checker=False):
        self.llm_name = llm_name
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        if retriever_name is not None:
            from utils import RetrievalSystem, DocExtracter
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
        self.templates = {"system": system_prompt, "user": user_prompt, "system-checker": system_checker_prompt, "user-checker": user_checker_prompt}
        
        if document_checker:
            system_prompt_ = self.templates["system-checker"]
        else:
            system_prompt_ = self.templates["system"]

        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model = genai.GenerativeModel(
            model_name=self.llm_name.split('/')[-1],
            generation_config={"temperature": 0},
            system_instruction=system_prompt_
        )
        self.max_length = 1048576
        self.context_length = 1040384
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def answer(self, question, options=None, k=32, rrf_k=100, save_dir=None, mc=False, abm=False, cell_type=None, signal=None, other_cell_types=None, other_signals=None, snippets=None, snippets_ids=None, snippet=None, document_checker=False):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        # retrieve relevant snippets
        if snippets is not None:
            snippets = [snippet for snippet in snippets if snippet['full_text'] != ""]
            retrieved_snippets = snippets[:k]
            print(len(retrieved_snippets))
            scores = []
        elif snippets_ids is not None:
            if self.docExt is None:
                from utils import DocExtracter
                self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
            retrieved_snippets = self.docExt.extract(snippets_ids[:k])
            scores = []
        elif snippet is not None:
            retrieved_snippets = [snippet]
            scores = []
        else:
            assert self.retrieval_system is not None
            retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k, signal=signal)
        #contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], (retrieved_snippets[idx]["content"] if retrieved_snippets[idx]["full_text"] == '' else retrieved_snippets[idx]["full_text"])) for idx in range(len(retrieved_snippets))]
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["full_text"]) for idx in range(len(retrieved_snippets))]
        if len(contexts) == 0:
            contexts = [""]
        contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
        num_tokens = len(self.tokenizer.encode('\n'.join(contexts)))
        print(f"context length: {num_tokens}")

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        context = "".join(contexts)

        if document_checker:
             prompt = self.templates["user-checker"].render(context=context, cell_type=cell_type, signal=signal)
        else:
            prompt = self.templates["user"].render(context=context, cell_type=cell_type, signal=signal, other_cell_types=other_cell_types, other_signals=other_signals)

        with open("prompt.txt", "w") as f:
            f.write(f"system prompt: {self.templates['system']} \n\n")
            f.write(f"user prompt: {prompt} \n\n")
            
        answers = self.generate(prompt)
                   
        if save_dir is not None:
            with open(os.path.join(save_dir, f"{cell_type}_{signal}_snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, f"{cell_type}_{signal}_response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers, retrieved_snippets, scores
            

    def generate(self, prompt):
        '''
        generate response given messages
        '''
        response = self.model.generate_content(prompt, request_options={"timeout": 1000})
        ans = response.candidates[0].content.parts[0].text
        return ans
