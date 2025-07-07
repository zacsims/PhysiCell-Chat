from src.utils import RetrievalSystem
from urllib.request import urlretrieve
from urllib.error import HTTPError
from metapub import FindIt, PubMedFetcher
import os
import time
import requests
import json

def retrieve_article_by_pmid(pmid, api_key, output_format='json'):
    """
    Retrieve a full-text article from Elsevier API using PubMed ID
    
    Args:
        pmid (str): PubMed ID (e.g., '32172064')
        api_key (str): Your Elsevier API key
        output_format (str): 'json' or 'xml' - format for the response
    
    Returns:
        dict or str: Article data in specified format
    """
    
    # Construct the API URL for PubMed ID retrieval
    base_url = "https://api.elsevier.com/content/article/pubmed_id/"
    url = f"{base_url}{pmid}"
    
    # Set up headers
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json" if output_format == 'json' else "text/xml"
    }
    
    # Add view parameter for full content
    params = {
        "view": "FULL"
    }
    
    try:
        # Make the API request
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        if output_format == 'json':
            return response.json()
        else:
            return response.text
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    
def download_wiley_article(doi, token, output_filename=None):
    """
    Download a PDF article from Wiley API using TDM (Text and Data Mining) service
    
    Args:
        doi (str): The DOI of the article (e.g., "10.1111/1467-923X.12168")
        token (str): Your Wiley TDM Client Token
        output_filename (str): Optional filename for the output PDF. 
                              If None, will use the last part of DOI + .pdf
    
    Returns:
        bool: True if download successful, False otherwise
    """
    
    # Construct the API URL
    base_url = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/"
    url = f"{base_url}{doi}"
    
    # Set up headers (equivalent to -H in curl)
    headers = {
        "Wiley-TDM-Client-Token": token
    }
    
    # Generate filename if not provided
    if output_filename is None:
        # Extract the last part of DOI for filename (e.g., "12168" from "10.1111/1467-923X.12168")
        doi_parts = doi.split('.')
        output_filename = f"{doi_parts[-1]}.pdf"
    
    try:
        print(f"Downloading article from: {url}")
        print(f"Output file: {output_filename}")
        
        # Make the request (equivalent to curl -L for following redirects)
        response = requests.get(url, headers=headers, allow_redirects=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Check if the response contains PDF content
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower() and len(response.content) < 1000:
            print(f"Warning: Response may not be a PDF. Content-Type: {content_type}")
            print(f"Response content (first 500 chars): {response.text[:500]}")
        
        # Write the PDF content to file (equivalent to -o in curl)
        with open(output_filename, 'wb') as file:
            file.write(response.content)
        
        file_size = os.path.getsize(output_filename)
        print(f"Successfully downloaded {output_filename} ({file_size:,} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading article: {e}")
        return False
    except IOError as e:
        print(f"Error writing file: {e}")
        return False
    
    
def download_pdf(pmid, savename, score, rank):
     src = FindIt(pmid)
     logname = '/'.join(savename.split('/')[:-1])+'/retrieval_log.txt'
     if os.path.exists(savename):
         print(f"pdf already downloaded for: {pmid=}.")
     elif src.url is not None:
         print(f"downloading pdf for: {pmid=}.")
         max_retries = 3
         retry_delay = 2  # seconds
         
         for attempt in range(max_retries):
             try:
                 urlretrieve(src.url, savename)
                 break  # Success, exit the retry loop
             except HTTPError as e:
                 if e.code == 500:
                    print(f"HTTP 500 Error when downloading {pmid} (attempt {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:  # Not the last attempt
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"Max retries reached for {pmid}")
             except Exception as e:
                 print(f"Unexpected error downloading {pmid}: {e}")
                 if attempt < max_retries - 1:
                     print(f"Retrying in {retry_delay} seconds...")
                     time.sleep(retry_delay)
                     retry_delay *= 2
                 else:
                     print(f"Max retries reached for {pmid}")
     elif src.reason == 'DENIED: ScienceDirect did not provide pdf link (probably paywalled)':
         api_key = "da6e64428cc390dbaf58c2d56dea8806"
         full_text = retrieve_article_by_pmid(pmid, api_key, output_format='json')
         if full_text is None:
             with open(logname, "a") as f:
                 f.write(f"{pmid} failed to retrieve {rank=} {score=} reason={src.reason} ELSEVIER API failed \n")
         else:
             with open(savename.replace('pdf','txt'), "w") as f:
                    f.write(full_text['full-text-retrieval-response']['originalText'])
     elif src.reason.startswith('DENIED: Wiley E Publisher says no to'):
         api_key = "436f53e6-6d53-4b67-95c5-13f967f89ea5"
         resp = download_wiley_article(src.doi, api_key, output_filename=savename)
         if resp == False:
             with open(logname, "a") as f:
                 f.write(f"{pmid} failed to retrieve {rank=} {score=} reason={src.reason} WILEY API failed \n")
     else:
         try:
             fetch = PubMedFetcher()
             article = fetch.article_by_pmid(pmid)
             with open(logname, "a") as f:
                 f.write(f"{pmid} failed to retrieve {rank=} {score=} reason={src.reason} year={article.year} \n")
         except Exception as e:
             # Handle invalid PMIDs and other metapub errors
             print(f"Error fetching article metadata for PMID {pmid}: {e}")
             with open(logname, "a") as f:
                 f.write(f"{pmid} failed to retrieve {rank=} {score=} reason={src.reason} ERROR: {str(e)} \n")

def retrieve_papers(retrieval_system, query, savepath, k=100):
     retrieved_snippets, scores = retrieval_system.retrieve(query, k=k)
     #print(f"retrieved {len(retrieved_snippets)} snippets")
    
     snippets_scores = sorted(zip(retrieved_snippets, scores), key=lambda item:item[1])

     if not os.path.exists(savepath): os.makedirs(savepath)
     
     successful_downloads = 0
     failed_downloads = 0
     
     for i,(snippet,score) in enumerate(snippets_scores):
         try:
             pmid = snippet['id'].split(':')[1]
             savename = f"{savepath}/{pmid}.pdf"
             if not os.path.exists(savename): 
                 download_pdf(pmid, savename, score, i)
                 successful_downloads += 1
             else:
                 successful_downloads += 1  # Already exists
         except Exception as e:
             failed_downloads += 1
             print(f"Error processing snippet {i}: {e}")
             # Log the error but continue with next paper
             logname = f"{savepath}/retrieval_log.txt"
             with open(logname, "a") as f:
                 f.write(f"SNIPPET_ERROR {i}: {str(e)} \n")
     
     print(f"Paper retrieval complete: {successful_downloads} successful, {failed_downloads} failed")

'''
    ("epithelial tumor","pressure"),
    ("CD8+ PD-1 high CD137 high T", "contact with PD-L1 high tumor cell"),
    ("CD8+ PD-1 high CD137 high T", "contact with PD-L1 low tumor cell"),
    ("CD8+ PD-1 high CD137 high T", "anti inflammatory factor"),
    ("CD8+ PD-1 high CD137 high T", "pro inflammatory factor"),
    ("CD8+ PD-1 low CD137 high T", "contact with PD-L1 high tumor cell"),
    ("CD8+ PD-1 low CD137 high T", "contact with PD-L1 low tumor cell"),
    ("CD8+ PD-1 low CD137 high T", "anti inflammatory factor"),
    ("CD8+ PD-1 low CD137 high T", "pro inflammatory factor"),
    ("CD8+ PD-1 high CD137 low T", "contact with PD-L1 high tumor cell"),
    ("CD8+ PD-1 high CD137 low T", "contact with PD-L1 low tumor cell"),
    ("CD8+ PD-1 high CD137 low T", "anti inflammatory factor"),
    ("CD8+ PD-1 high CD137 low T", "pro inflammatory factor"),
    ("CD8+ PD-1 low CD137 low T", "contact with PD-L1 high tumor cell"),
    ("CD8+ PD-1 low CD137 low T", "contact with PD-L1 low tumor cell"),
    ("CD8+ PD-1 high CD137 low T", "anti inflammatory factor"),
    ("CD8+ PD-1 low CD137 low T", "pro inflammatory factor"),


#signal/cell type
rules = [
    ("M0 macrophage","contact with dead cell"),
    ("M0 macrophage","death"),
    #
    ("M1 macrophage","contact with dead cell"),
    ("M1 macrophage","death"),
    ("M1 macrophage","IFN-Gamma"),
    #
    ("M2 macrophage","contact with dead cell"),
    ("M2 macrophage","death"),
    ("M2 macrophage","IFN-Gamma"),
    ("M2 macrophage","IL-4"),
    #
    ("CD4+ PD-1 high T","contact with dead cell"),
    ("CD4+ PD-1 high T","anti inflammatory factor"),
    #
    ("CD4+ PD-1 low T","contact with dead cell"),
    ("CD4+ PD-1 low T","anti inflammatory factor"),
    #
    ("CD4+ TH2+ T","contact with M0 macrophage cell"),
    ("CD4+ TH2+ T","contact with M1 macrophage cell"),
    ("CD4+ TH2+ T","contact with M2 macrophage cell"),
    ("CD4+ TH2+ T","contact with tumor cell"),
    ("CD4+ TH2+ T","death"),
    ("CD4+ TH2+ T","IFN-Gamma"),
    ("CD4+ TH2+ T","IL-4"),
    #
    ("CD8+ T", "contact with tumor cell"),
    ("CD8+ T", "death"),
    ("CD8+ T", "anti inflammatory factor"),
    ("CD8+ T", "pro inflammatory factor"),
    ("CD8+ T", "IFN-Gamma"),
    ("CD8+ T", "IL-10"),
    #
    ("tumor","damage"),
    ("tumor","death"),
    ("tumor","EGF"),
    ("tumor","IFN-Gamma"),
    ("tumor","pressure")
]
 
#signal/cell type
rules = [
    ("CD8+ T", "IL-10")
]
    

retrieval_system = RetrievalSystem("MedCPT", corpus_name="PubMed")
#retrieve papers
for cell_type,signal in rules:
    retrieve_papers(retrieval_system, f"effect of {signal} on {cell_type} cells in triple negative breast cancer", f"ohsu-immune-papers/{'-'.join(cell_type.split(' '))}-{signal}")
'''