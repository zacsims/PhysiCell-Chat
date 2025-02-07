import os
import asyncio
from paperqa import Settings, ask

#llm = 'openai/gpt-4o-2024-08-06'
#llm = 'openai/o1-preview-2024-09-12'
llm = 'anthropic/claude-3-5-sonnet-20241022'
llm_config = {
    "model_list": [
        {
            "model_name": llm,
            "litellm_params": {
                "model": llm,
                "drop_params": "True",
            },
        }
    ]
}
#query = 'What is the effect of IL-10 on CD8+ T Cells in triple negative breast cancer?'
query = "Is IL-10 considered diffusable or membrane bound? and how is it activated?"
query_id = '-'.join(query.lower().split(' ')[:10])

settings = Settings(paper_directory=f'queries/{query_id}/papers', llm=llm, llm_config=llm_config)
answer = ask(query, settings=settings)

answer_dir = f'queries/{query_id}/answers'
if not os.path.exists(answer_dir): os.makedirs(answer_dir)
fname = f'{query_id}-{llm.split("/")[1]}.txt'
with open(f'{answer_dir}/{fname}', 'w') as f:
	f.write(answer.session.formatted_answer)
