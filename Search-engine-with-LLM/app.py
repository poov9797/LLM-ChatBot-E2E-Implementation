from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
from openai import OpenAI

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI and Elasticsearch clients
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

es_client = Elasticsearch('http://localhost:9200')

# Function to perform Elasticsearch search
def elastic_search(query, index_name="course-questions"):
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
    result_docs = [hit['_source'] for hit in response['hits']['hits']]
    
    return result_docs

# Function to build a prompt for LLM
def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = "\n".join(
        f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n"
        for doc in search_results
    )
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

# Function to get a response from the LLM
def llm(prompt):
    response = client.chat.completions.create(
        model='phi3',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Function to handle the full RAG process
def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

# Flask route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for processing the user query
@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    answer = rag(query)
    return jsonify({'response': answer})

if __name__ == "__main__":
    app.run(debug=True)
