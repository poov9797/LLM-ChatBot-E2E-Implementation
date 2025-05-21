import os
import time
import json

from openai import OpenAI

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


ELASTIC_URL = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/v1/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")


es_client = Elasticsearch(ELASTIC_URL)
# ollama_client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")


def elastic_search_text(query, course, index_name="course-questions"):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {"term": {"course": course}},
            }
        },
    }

    response = es_client.search(index=index_name, body=search_query)
    return [hit["_source"] for hit in response["hits"]["hits"]]


def elastic_search_knn(field, vector, course, index_name="course-questions"):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {"term": {"course": course}},
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"],
    }

    es_results = es_client.search(index=index_name, body=search_query)

    return [hit["_source"] for hit in es_results["hits"]["hits"]]


def build_prompt(query):
    prompt_template = """Analyze the following job description and determine whether it shows signs of potential human trafficking or labor exploitation. 
    Consider factors such as vague responsibilities, excessive control over workers, unrealistic promises, poor working conditions, lack of legal protections, 
    recruitment from vulnerable populations, or requirements to surrender personal documents. 
    Respond in JSON format with fields ‘RiskLevel’ (LOW, MEDIUM, HIGH), ‘Indicators’ (a list of red flags), and ‘Explanation’ (a brief summary of your reasoning).

Job Description:
{query}""".strip()

    # context = "\n\n".join(
    #     [
    #         f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}"
    #         for doc in search_results
    #     ]
    # )
    return prompt_template.format(query =query).strip()

import httpx

def query_ollama(prompt, model):
    url = f"{OLLAMA_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Content-Type": "application/json"}
    response = httpx.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def llm(prompt, model_choice):
    start_time = time.time()

    if model_choice.startswith('ollama/'):
        model_name = model_choice.split('/')[-1]
        response = query_ollama(prompt, model_name)
        answer = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })


    # if model_choice.startswith('ollama/'):
    #     response = ollama_client.chat.completions.create(
    #         model=model_choice.split('/')[-1],
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     answer = response.choices[0].message.content
    #     tokens = {
    #         'prompt_tokens': response.usage.prompt_tokens,
    #         'completion_tokens': response.usage.completion_tokens,
    #         'total_tokens': response.usage.total_tokens
    #     }
    # elif model_choice.startswith('openai/'):
    #     response = openai_client.chat.completions.create(
    #         model=model_choice.split('/')[-1],
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     answer = response.choices[0].message.content
    #     tokens = {
    #         'prompt_tokens': response.usage.prompt_tokens,
    #         'completion_tokens': response.usage.completion_tokens,
    #         'total_tokens': response.usage.total_tokens
    #     }
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return answer, tokens, response_time


def evaluate_relevance(question, answer):
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt, 'ollama/phi3')
    
    try:
        json_eval = json.loads(evaluation)
        return json_eval['Relevance'], json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens


def calculate_openai_cost(model_choice, tokens):
    openai_cost = 0

    if model_choice == 'openai/gpt-3.5-turbo':
        openai_cost = (tokens['prompt_tokens'] * 0.0015 + tokens['completion_tokens'] * 0.002) / 1000
    elif model_choice in ['openai/gpt-4o', 'openai/gpt-4o-mini']:
        openai_cost = (tokens['prompt_tokens'] * 0.03 + tokens['completion_tokens'] * 0.06) / 1000

    return openai_cost


def get_answer(query, model_choice):
    # if search_type == 'Vector':
    #     vector = model.encode(query)
    #     search_results = elastic_search_knn('question_text_vector', vector, course)
    # else:
    #     search_results = elastic_search_text(query, course)
    prompt = build_prompt(query)
    # prompt = build_prompt(query, search_results)
    answer, tokens, response_time = llm(prompt, model_choice)
    
    # relevance, explanation, eval_tokens = evaluate_relevance(query, answer)

    # openai_cost = calculate_openai_cost(model_choice, tokens)
 
    return {
        'answer': answer,
        'response_time': response_time,
        # 'relevance': relevance,
        # 'relevance_explanation': explanation,
        # 'model_used': model_choice,
        # 'prompt_tokens': tokens['prompt_tokens'],
        # 'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        # 'eval_prompt_tokens': eval_tokens['prompt_tokens'],
        # 'eval_completion_tokens': eval_tokens['completion_tokens'],
        # 'eval_total_tokens': eval_tokens['total_tokens'],
        # 'openai_cost': openai_cost
    }