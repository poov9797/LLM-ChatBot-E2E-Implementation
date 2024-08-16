from flask import Flask, render_template, request, jsonify
import time  # Just for simulating a delay in the rag function

app = Flask(__name__)

from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

import requests 

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

index_name = "course-questions"

from tqdm.auto import tqdm

for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)

def elastic_search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs

def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = client.chat.completions.create(
        model='gemma2:2b',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content




def rag(input_data):
    # query=input_data
    # search_results = elastic_search(query)
    # prompt = build_prompt(query, search_results)

    prompt="You are a chatbot, who answers the question asked?" + query
    answer = llm(prompt)
    # Simulate a delay
    # time.sleep(2)
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_data = request.form.get('input_data')
    result = rag(input_data)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
