import os
import ast
import csv
import time
import pickle
import asyncio
import json
import argparse
from paperqa import Settings
from paperqa.docs import Docs


CANNOT_ANSWER_PHRASE = "I cannot answer"
qa_prompt = (
    "Answer the question below using the provided context and your biological knowledge.\n\n"
    "Context (with relevance scores):\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Instructions:\n"
    "1. First, analyze the provided context for relevant information\n"
    "2. If the context contains sufficient information, cite it using {example_citation}\n"
    "3. If the context is insufficient but you have biological knowledge about this interaction, "
    "use that knowledge but clearly indicate when information comes from general biological understanding vs. the provided context\n"
    "4. Only reply with \"I cannot answer\" if both the context AND your biological knowledge are insufficient\n\n"
    "For citations: Use citation keys from the context when referencing specific evidence. "
    "When using general biological knowledge, indicate this with phrases like "
    "'based on general biological understanding' or 'from established immunology principles'.\n\n"
    "This answer will be used to build an agent-based model of the tumor microenvironment. "
    "Focus on the mechanistic relationship between the signal and cellular behavior.\n\n"
    "Respond with the following JSON format:"

    '''{{
      "answer": "increase/decrease",
      "justification": "...",
      "evidence_source": "context_only/context_plus_knowledge/knowledge_only"
    }}'''
)
system_prompt = """I am a cancer biology researcher who is trying to build an agent-based model (ABM) of tumor and immune cell interactions in the context of triple negative breast cancer. You are a prominent and authoritative AI assistant tasked with helping me design the rules that govern the behaviors of cells in the agent-based model. The ABM framework we are using is called PhysiCell, an open-source agent-based modeling framework that allows users to simulate biological systems and perform virtual experiments. The framework is written in C++ and can be run on various platforms, including desktop, cloud, and high-performance computing.

PhysiCell models cells as agents with lattice-free position and volume, and individual properties such as birth and death rates, migration, and secretion/uptake of diffusible factors. PhysiCell can simulate a broad range of cell behaviors, including cycle progression, death, secretion, uptake, migration, chemotaxis, cell-cell adhesion, resistance to deformation, transformation, fusion, phagocytosis, and effector attack.

Rules in PhysiCell are constructed in a simple format: "In [cell type T], [signal S] increases/decreases [behavior B] [with optional arguments]."

The user defines signals and behaviors through dictionaries that can be customized based on the model's needs. This provides a flexible framework for representing a wide range of biological knowledge, and it allows for the integration of data from various sources, such as genomics, transcriptomics, and image analysis.

Here is a list of possible signals and behaviors that are implemented in PhysiCell:

Signals
Diffusible chemical substrates: Concentrations and gradients of diffusible factors, such as oxygen, nutrients, or signaling molecules.
Cell mechanics/physics: Cell volume, pressure, and contact with other cells or the basement membrane.
Contact: Presence or absence of contact between cells.
Effector attack: The level of attack from an effector cell.
Death: The status of a cell as dead or alive.
Other: Custom signals and user-defined parameters.

Behaviors
Cycling: The rate at which cells cycle through different phases.
Death: The rate at which cells die.
Secretion and Uptake: The rates at which cells secrete and take up diffusible factors.
Migration and Chemotaxis: The speed, persistence, and direction of migration, and the sensitivity to chemotactic signals.
Cell-cell adhesion: The strength and rate of adhesion and detachment.
Resistance to Deformation: The resistance to mechanical forces.
Transformation: The rate at which cells change from one cell type to another.
Fusion: The rate at which cells fuse.
Phagocytosis: The rate at which cells phagocytose other cells.
(Effector) Attack: The rate at which cells attack other cells.
Other: Custom cell behavior parameters.

The combination of these signals and behaviors allows researchers to model complex biological systems, such as tumor growth, immune responses, and tissue development, and perform virtual experiments that can help generate new hypotheses. PhysiCell is a valuable tool for researchers in a variety of fields, including cancer biology, immunology, and developmental biology.
I will provide you with a question regarding potential rule in the format "Does signal increase or decrease behavior in cell type" along with a set of research papers from PubMed that may contain information related to the cell type and signal in the context of triple negative breast cancer. Please use the provided literature to make a determination on the direction of this rule (whether the behavior increases or decreases) for use in my ABM model.
"""

def get_settings(model, chunk_size=50_000, answer_max_sources=5, evidence_summary_length="about 500 words"):
    # Load settings
    settings = {}

    # Configure settings
    if model == "gemini/gemini-2.0-flash":
        VERY_HIGH_TOKEN_LIMIT = "100000000 per 1 minute"  # 100 million tokens per minute
        settings['llm'] = "gemini/gemini-2.0-flash"
        settings['summary_llm'] = "gemini/gemini-2.0-flash"
        settings['llm_config'] = {"rate_limit": {"gemini/gemini-2.0-flash": VERY_HIGH_TOKEN_LIMIT}}
        settings['summary_llm_config'] = {
            "rate_limit": {"gemini/gemini-2.0-flash": VERY_HIGH_TOKEN_LIMIT},
            "temperature": 0.1,  # Even lower for summaries
            "max_tokens": 1024,
            "timeout": 60,
            "request_timeout": 60,
            "retry_count": 3,
        }
    if model == "gemini/gemini-2.0-flash-lite":
        VERY_HIGH_TOKEN_LIMIT = "100000000 per 1 minute"  # 100 million tokens per minute
        settings['llm'] = "gemini/gemini-2.0-flash-lite"
        settings['summary_llm'] = "gemini/gemini-2.0-flash-lite"
        settings['llm_config'] = {
            "rate_limit": {"gemini/gemini-2.0-flash-lite": VERY_HIGH_TOKEN_LIMIT}, 
            "cache":{"no-cache": True}, 
            "temperature":1,
            "extra_body": {
                "generationConfig": {
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "answer": {"type": "STRING"},
                            "justification": {"type": "STRING"}
                        },
                        "required": ["answer", "justification"]
                    }
                }
            }
        }
        settings['summary_llm_config'] = {
            "rate_limit": {"gemini/gemini-2.0-flash-lite": VERY_HIGH_TOKEN_LIMIT},
            "temperature": 0.1,  # Even lower for summaries
            "max_tokens": 1024,
            "timeout": 60,
            "request_timeout": 60,
            "retry_count": 3,
        }
        
    if model == "gemini/gemini-2.5-pro-preview-03-25":
        VERY_HIGH_TOKEN_LIMIT = "100000000 per 1 minute"  # 100 million tokens per minute
        settings['llm'] = "gemini/gemini-2.5-pro-preview-03-25"
        settings['summary_llm'] = "gemini/gemini-2.0-flash-lite"
        settings['llm_config'] = {"rate_limit": {"gemini/gemini-2.5-pro-preview-03-25": VERY_HIGH_TOKEN_LIMIT}, "cache":{"no-cache": True}, "temperature":1}
        settings['summary_llm_config'] = {"rate_limit": {"gemini/gemini-2.0-flash-lite": VERY_HIGH_TOKEN_LIMIT}}
    
    if model == 'o4-mini':
        settings['llm'] = "o4-mini"
        settings['summary_llm'] = "o4-mini"
        settings['llm_config'] = {
            "rate_limit": {"o4-mini": "100000 per 1 minute"}, 
            "temperature": 1, 
            "cache": {"no-cache": True},
            "litellm_settings": {
                "drop_params": True
            },
            "model_list": [
                {
                    "model_name": "o4-mini",
                    "litellm_params": {
                        "model": "o4-mini",
                        "drop_params": True
                    }
                }
            ]
        }
        settings['summary_llm_config'] = {
            "rate_limit": {"o4-mini": "100000 per 1 minute"}, 
            "temperature": 1,
            "litellm_settings": {
                "drop_params": True
            },
            "model_list": [
                {
                    "model_name": "o4-mini",
                    "litellm_params": {
                        "model": "o4-mini",
                        "drop_params": True
                    }
                }
            ]
        }
    if model == 'gpt-4o-mini':
        VERY_HIGH_TOKEN_LIMIT = "100000000 per 1 minute"  # 100 million tokens per minute
        settings['llm'] = 'gpt-4o-mini'
        settings['summary_llm'] = "gpt-4o-mini"
        settings['llm_config'] = {"rate_limit": {'gpt-4o-mini': "30000 per 1 minute"}, "temperature":1}
        settings['summary_llm_config'] = {"rate_limit": {'gpt-4o-mini': "30000 per 1 minute"}, "temperature":1}
        
        
    settings['answer'] = {'evidence_k': 25, 'answer_max_sources': answer_max_sources, "evidence_summary_length": evidence_summary_length}
    settings['parsing'] = {'chunk_size': chunk_size}
    #settings['answer']['evidence_retrieval'] = False
    settings['prompts'] = {'qa':qa_prompt, 'system':system_prompt}
    settings['embedding'] = 'gemini/text-embedding-004'
    # Reduce concurrency to help with rate limiting
    settings['answer']['max_concurrent_requests'] = 1
    #settings['parsing'] = {'chunk_size': 0}

    settings = Settings(**settings)
    
    return settings


def get_model_name(settings, summary=False):
    if summary == True:
        if len(settings.summary_llm.split('/')) > 1:
            model_name = settings.summary_llm.split('/')[1]
        else:
            model_name = settings.summary_llm
    else:
        if len(settings.llm.split('/')) > 1:
            model_name = settings.llm.split('/')[1]
        else:
            model_name = settings.llm
    return model_name


async def assemble_docs(paper_dir, settings):
        
    pickle_dir = paper_dir.split('/')
    pickle_dir.insert(1, 'paperqa-objects')
    pickle_dir = ('/').join(pickle_dir)
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)  
    
    model_name = get_model_name(settings, summary=True)
    #model_name = 'gemini-2.0-flash-lite'
    chunk_size = settings.parsing.chunk_size
    pickle_path = f"{pickle_dir}/docs-{model_name}-chunk{chunk_size}.pickle"
    print(pickle_path)
    if not os.path.exists(paper_dir):
        os.makedirs(paper_dir) 
    
    if os.path.exists(pickle_path):
        print("loading from pickle")
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
            # Handle both old format (direct docs) and new format (dict with metadata)
            if isinstance(pickle_data, dict) and 'docs' in pickle_data:
                docs = pickle_data['docs']
                print(f"Loaded docs with chunk_size: {pickle_data.get('chunk_size', 'unknown')}")
            else:
                # Old format - just the docs object
                docs = pickle_data
                print("Loaded docs (old format without metadata)") 
    else:
        docs = Docs()
        for i, paper_path in enumerate(os.listdir(paper_dir)):
            if 'retrieval_log' in paper_path: continue
            # Add retry logic with exponential backoff
            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    print(f"Processing {i+1}/{len(os.listdir(paper_dir))}: {paper_path}")
                    await docs.aadd(f'{paper_dir}/{paper_path}', settings=settings)
                    # Add a pause between requests to avoid rate limiting
                    time.sleep(1) 
                    break
                except Exception as e:
                    retry_count += 1
                    if "429" in str(e):
                        # If we hit the rate limit, wait longer
                        wait_time = 30 * retry_count  # Increase wait time with each retry
                        print(f"Rate limit hit, waiting for {wait_time} seconds before retry {retry_count}/{max_retries}")
                        time.sleep(wait_time)
                    else:
                        # For other errors, shorter wait
                        wait_time = 5 * retry_count
                        print(f"Error: {e}, retrying in {wait_time} seconds ({retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    
                    if retry_count == max_retries:
                        print(f"Failed to process {paper_path} after {max_retries} retries")
        
        # Create metadata dict with parameter settings
        docs_metadata = {
            'docs': docs,
            'chunk_size': settings.parsing.chunk_size,
            'model_name': model_name
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(docs_metadata, f) 
        
    return docs, pickle_dir
        
    
async def assemble_evidence(docs, settings, cell_type, signal, behavior, pickle_dir, from_pickle=True):
    print('called assemble_evidence')
    model_name = get_model_name(settings, summary=True)
    #model_name = "gemini-2.0-flash-lite"
    evidence_k = settings.answer.evidence_k
    answer_max_sources = settings.answer.answer_max_sources
    evidence_summary_length = settings.answer.evidence_summary_length
    pickle_path = f"{pickle_dir}/session-{cell_type.replace(' ','-')}-{signal.replace(' ','-')}-{behavior.replace(' ','-')}-{model_name}-k{evidence_k}-{evidence_summary_length.replace(' ', '')}.pickle"
    if os.path.exists(pickle_path) and from_pickle:
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
            # Handle both old format (direct session) and new format (dict with metadata)
            if isinstance(pickle_data, dict) and 'session' in pickle_data:
                session = pickle_data['session']
                print(f"Loaded evidence with k={pickle_data.get('evidence_k', 'unknown')}, summary_length={pickle_data.get('evidence_summary_length', 'unknown')}")
            else:
                # Old format - just the session object
                session = pickle_data
                print("Loaded evidence (old format without metadata)")  
    else:
        query = f"Does {signal} increase or decrease {behavior} in {cell_type} cells in triple negative breast cancer?"
        session = await docs.aget_evidence(query, settings=settings)
        
        # Create metadata dict with parameter settings
        session_metadata = {
            'session': session,
            'evidence_k': settings.answer.evidence_k,
            'answer_max_sources': settings.answer.answer_max_sources,
            'evidence_summary_length': settings.answer.evidence_summary_length,
            'model_name': model_name
        }
        
        print(f'writing pickle file to {pickle_path}')
        with open(pickle_path, 'wb') as f:
            pickle.dump(session_metadata, f)  
        
    return session


async def answer_single(docs, session, settings, cell_type, signal, behavior, pickle_dir, run_id=0):
    """Run a single query and return the answer"""
    session.id = run_id
    out = await docs.aquery(query=session, settings=settings)
    model_name = get_model_name(settings)
    
    # Create metadata dict with parameter settings
    answer_metadata = {
        'answer_output': out,
        'chunk_size': settings.parsing.chunk_size,
        'evidence_k': settings.answer.evidence_k,
        'evidence_summary_length': settings.answer.evidence_summary_length,
        'model_name': model_name,
        'run_id': run_id
    }
    
    # Save the full output with metadata
    with open(f"{pickle_dir}/answer-run{run_id}-{cell_type.replace(' ','-')}-{signal.replace(' ','-')}-{behavior.replace(' ','-')}-{model_name}-chunk{settings.parsing.chunk_size}-k{settings.answer.evidence_k}.pickle", 'wb') as f:
        pickle.dump(answer_metadata, f)
    
    return out.answer

        
def log_experiment_result(answer, cell_type, signal, behavior, ground_truth, model_name, 
                         evidence_k, answer_max_sources, chunk_size, evidence_summary_length, run_id, output_dir):
    """Log a single experimental result to CSV"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a separate CSV file for each cell type and model
    cell_type_filename = cell_type.replace(' ', '_').replace('+', 'plus')
    model_filename = model_name.replace('-', '_').replace('/', '_')
    output_csv_path = f"{output_dir}/experiment_results_{cell_type_filename}_{model_filename}.csv"
    fieldnames = ['cell_type', 'signal', 'behavior', 'ground_truth', 'answer', 'justification', 
                 'evidence_source', 'evidence_k', 'answer_max_sources', 'chunk_size', 'evidence_summary_length', 'run_id', 'model']
    
    # Create CSV with headers if it doesn't exist
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Parse the answer JSON
    try:
        # Handle None or empty answer
        if answer is None:
            answer_text = "No response"
            justification = "LLM returned None"
            evidence_source = "error"
        elif not answer or not isinstance(answer, str):
            answer_text = str(answer) if answer else "Empty response"
            justification = "Invalid response type"
            evidence_source = "error"
        else:
            # Clean the response first - remove markdown code blocks
            cleaned_answer = answer
            if '```json' in answer:
                start_marker = answer.find('```json') + 7
                end_marker = answer.find('```', start_marker)
                if end_marker != -1:
                    cleaned_answer = answer[start_marker:end_marker].strip()
                else:
                    cleaned_answer = answer[start_marker:].strip()
            elif '```' in answer:
                start_marker = answer.find('```') + 3
                end_marker = answer.find('```', start_marker)
                if end_marker != -1:
                    cleaned_answer = answer[start_marker:end_marker].strip()
            
            # Try to find and parse JSON
            json_found = False
            parsed_answer = None
            import json
            
            # Try multiple JSON extraction and parsing methods
            for attempt in range(4):
                try:
                    parsed_answer = None  # Reset for each attempt
                    
                    if attempt == 0:
                        # Method 1: Direct json.loads on cleaned response
                        if cleaned_answer.strip().startswith('{'):
                            parsed_answer = json.loads(cleaned_answer)
                    elif attempt == 1:
                        # Method 2: Find JSON block with brace counting
                        start_idx = cleaned_answer.find('{')
                        if start_idx != -1:
                            brace_count = 0
                            end_idx = start_idx
                            for i, char in enumerate(cleaned_answer[start_idx:], start_idx):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break
                            json_str = cleaned_answer[start_idx:end_idx]
                            parsed_answer = json.loads(json_str)
                    elif attempt == 2:
                        # Method 3: Fix common JSON issues (escaped quotes)
                        start_idx = cleaned_answer.find('{')
                        end_idx = cleaned_answer.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = cleaned_answer[start_idx:end_idx]
                            # Fix escaped quotes that break JSON
                            json_str = json_str.replace('\\"', '"').replace('""', '"')
                            parsed_answer = json.loads(json_str)
                    else:
                        # Method 4: Use ast.literal_eval as fallback
                        start_idx = cleaned_answer.find('{')
                        end_idx = cleaned_answer.rfind('}') + 1
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = cleaned_answer[start_idx:end_idx]
                            parsed_answer = ast.literal_eval(json_str)
                    
                    # If we get here and have a valid parsed_answer, parsing succeeded
                    if parsed_answer is not None:
                        answer_text = parsed_answer.get('answer', '')
                        justification = parsed_answer.get('justification', '')
                        evidence_source = parsed_answer.get('evidence_source', '')
                        json_found = True
                        break
                        
                except (ValueError, SyntaxError, json.JSONDecodeError, TypeError) as parse_error:
                    continue  # Try next method
            
            if not json_found:
                # No JSON found, treat entire response as answer
                print(f"No JSON structure found in answer, using raw response")
                answer_text = cleaned_answer.strip()
                justification = "Raw response (no JSON structure)"
                evidence_source = "unknown"
            
    except Exception as e:
        print(f"Error parsing answer: {e}")
        print(f"Raw answer: {answer}")
        answer_text = str(answer) if answer else "Parse error"
        justification = f"Parse error: {str(e)}"
        evidence_source = "parse_error"
    
    # Append the result
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'cell_type': cell_type,
            'signal': signal,
            'behavior': behavior,
            'ground_truth': ground_truth,
            'answer': answer_text,
            'justification': justification,
            'evidence_source': evidence_source,
            'evidence_k': evidence_k,
            'answer_max_sources': answer_max_sources,
            'chunk_size': chunk_size,
            'evidence_summary_length': evidence_summary_length,
            'run_id': run_id,
            'model': model_name
        })
    
    return answer_text
    

def load_rules_from_csv(csv_file):
    """Load rules from a CSV file without headers (cell_type,signal,direction,behavior,...)"""
    try:
        rules = {}
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader):
                if len(row) < 4:
                    print(f"Warning: Row {row_num + 1} has {len(row)} columns, expected at least 4. Skipping.")
                    continue
                
                # Use only the first 4 columns, ignore any additional parameters
                cell_type, signal, direction, behavior = row[0], row[1], row[2], row[3]
                
                if cell_type not in rules:
                    rules[cell_type] = []
                
                rules[cell_type].append((signal, behavior, direction))
        
        return rules
    except FileNotFoundError:
        print(f"Error: Rules file {csv_file} not found")
        return None
    except Exception as e:
        print(f"Error: Failed to load CSV file {csv_file}: {e}")
        return None

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run non-agentic PaperQA validation')
    parser.add_argument('--rules', '-r', type=str, default='all-test-rules.csv', 
                       help='Path to CSV file containing rules (default: all-test-rules.csv)')
    parser.add_argument('--model', '-m', type=str, default='o4-mini',
                       help='Model name to use (default: o4-mini)')
    parser.add_argument('--output-dir', '-o', type=str, default='experiment_results',
                       help='Output directory for results (default: experiment_results)')
    parser.add_argument('--num-runs', '-n', type=int, default=1,
                       help='Number of runs per experiment (default: 1)')
    
    args = parser.parse_args()
    
    # Load rules from CSV file
    rules = load_rules_from_csv(args.rules)
    if rules is None:
        print("Failed to load rules. Exiting.")
        return
    
    print(f"Loaded {len(rules)} cell types with {sum(len(cell_rules) for cell_rules in rules.values())} total rules from {args.rules}")
    
    # Define experimental parameters
    experiment_params = {
        'chunk_size': {'low': 10_000, 'medium': 50_000, 'high': 100_000, 'full_doc': 200_000},
        'answer_max_sources': {'very_low': 1, 'low': 5, 'medium': 25, 'high': 50},
        'evidence_summary_length': {
            'low': "about 200 words", 
            'medium': "about 500 words", 
            'high': "about 1000 words"
        }
    }
    
    # Configuration from command line arguments
    model_name = args.model
    num_runs = args.num_runs
    output_dir = args.output_dir
    
    # Initialize retrieval system
    retrieval_system = None
    
    # Total number of experiments for progress tracking
    total_experiments = len(rules) * sum(len(cell_rules) for cell_rules in rules.values()) * \
                       len(experiment_params['chunk_size']) * len(experiment_params['answer_max_sources']) * \
                       len(experiment_params['evidence_summary_length']) * num_runs
    
    current_experiment = 0
    
    print(f"Starting experiments with {total_experiments} total runs...")
    
    # Main experimental loop
    for cell_type, cell_rules in rules.items():
        for signal, behavior, ground_truth in cell_rules:
            # Setup paper directory and retrieve papers
            paper_dir = f"ohsu-immune-papers/{'-'.join(cell_type.split(' '))}-{signal.replace(' ', '-')}-{behavior.replace(' ', '-')}"
            old_paper_dir = f"ohsu-immune-papers/{'-'.join(cell_type.split(' '))}-{signal.replace(' ', '-')}"
            
            # Check if papers already exist in new format
            if os.path.exists(paper_dir) and len([f for f in os.listdir(paper_dir) if f.endswith('.pdf')]) > 0:
                print(f"\nPapers already exist for {cell_type} - {signal} - {behavior} (found {len([f for f in os.listdir(paper_dir) if f.endswith('.pdf')])} PDFs)")
            else:
                print(f"\nRetrieving papers for {cell_type} - {signal} - {behavior}...")
                if retrieval_system is None: 
                    from src.utils import RetrievalSystem
                    from retrieve import retrieve_papers
                    retrieval_system = RetrievalSystem("MedCPT", corpus_name="PubMed")
                retrieve_papers(
                    retrieval_system, 
                    f"effect of {signal} on {behavior} in {cell_type} cells in triple negative breast cancer", paper_dir
                )
            
            # Run experiments with different parameter combinations
            for chunk_level in ['full_doc']:
                #for answer_max_sources_level in ['very_low', 'low', 'medium', 'high']:
                for answer_max_sources_level in ['low']:
                    for summary_level in ['medium']:
                        # Get parameter values
                        chunk_size = experiment_params['chunk_size'][chunk_level]
                        answer_max_sources = experiment_params['answer_max_sources'][answer_max_sources_level]
                        evidence_summary_length = experiment_params['evidence_summary_length'][summary_level]
                        
                        print(f"\n--- Experiment: {cell_type} | {signal} → {behavior} ---")
                        print(f"Parameters: chunk_size={chunk_size}, answer_max_sources={answer_max_sources}, summary_length={evidence_summary_length}")
                        
                        # Create settings for this parameter combination
                        settings = get_settings(model_name, chunk_size, answer_max_sources, evidence_summary_length)
                        
                        # Assemble docs and evidence (reuse across runs)
                        try:
                            docs, pickle_dir = await assemble_docs(paper_dir, settings)
                            session = await assemble_evidence(docs, settings, cell_type, signal, behavior, pickle_dir)
                            
                            # Run multiple iterations
                            for run_id in range(num_runs):
                                current_experiment += 1
                                print(f"  Run {run_id + 1}/{num_runs} (Overall: {current_experiment}/{total_experiments})")
                                
                                try:
                                    # Get answer for this run
                                    answer = await answer_single(docs, session, settings, cell_type, signal, behavior, pickle_dir, run_id)
                                    
                                    # Log the result
                                    predicted_answer = log_experiment_result(
                                        answer, cell_type, signal, behavior, ground_truth, 
                                        get_model_name(settings), 50, answer_max_sources, chunk_size, 
                                        evidence_summary_length, run_id, output_dir
                                    )
                                    
                                    print(f"    Predicted: {predicted_answer}, Ground Truth: {ground_truth}")
                                    
                                except Exception as e:
                                    print(f"    Error in run {run_id}: {e}")
                                    # Log the error
                                    log_experiment_result(
                                        f"ERROR: {str(e)}", cell_type, signal, behavior, ground_truth,
                                        get_model_name(settings), 50, answer_max_sources, chunk_size,
                                        evidence_summary_length, run_id, output_dir
                                    )
                        
                        except Exception as e:
                            print(f"Error setting up experiment: {e}")
                            for run_id in range(num_runs):
                                current_experiment += 1
                                log_experiment_result(
                                    f"SETUP_ERROR: {str(e)}", cell_type, signal, behavior, ground_truth,
                                    model_name, 50, answer_max_sources, chunk_size,
                                    evidence_summary_length, run_id, output_dir
                                )
    
    print(f"\nAll experiments completed! Results saved to {output_dir}/experiment_results.csv")

if __name__ == '__main__':
    asyncio.run(main())
        
        
        
        