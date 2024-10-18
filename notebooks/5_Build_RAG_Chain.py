# Databricks notebook source
# MAGIC %md
# MAGIC # Build RAG Chain
# MAGIC This will use LangChain to build the request model used the extra context pulled from the Vector Search and prompt engineering to build a custom LLM 
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.16 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.9.0 databricks-sdk==0.12.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Define your attributes
catalog_name = "stu_sandbox"
schema_name = "rag_model"
vector_search_endpoint_name = "stu-test-vector"
model_endpoint_name = "bge_small_en_v1_5"
chat_model_endpoint_name = "mixtral_8x7b_instruct_v0_1" ##TODO: Fix issues deploying this "dbrx_instruct"
our_model_name = "stu_rag_model"

encoded_table_name = f"{catalog_name}.{schema_name}.silver_pdfs_chunked_encoded"
encoded_table_name_vs_index = f"{encoded_table_name}_vs_index"

# COMMAND ----------

import os

# optional if wanting to set deployed model
os.environ["DATABRICKS_TOKEN"] = "<change me>"
os.environ["DATABRICKS_HOST"] = "https://adb-3800464929700097.17.azuredatabricks.net"

# Verify it's set
print(os.environ["DATABRICKS_TOKEN"])
print(os.environ["DATABRICKS_HOST"])

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
import os

# Define embedding model
embedding_model = DatabricksEmbeddings(endpoint=model_endpoint_name)

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=os.environ["DATABRICKS_HOST"], 
     personal_access_token=os.environ["DATABRICKS_TOKEN"],
     disable_notice=True                  
    )

    vs_index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=encoded_table_name_vs_index) 

    # Create the retriever
    vectorstore = DatabricksVectorSearch(index=vs_index, embedding=embedding_model, text_column="content")
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# test your retriever
vectorstore = get_retriever()
similar_documents = vectorstore.invoke("when is Christmas shut down?")
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# Import necessary libraries
from langchain.chat_models import ChatDatabricks

# Define foundation model for generating responses
chat_model = ChatDatabricks(endpoint=chat_model_endpoint_name, max_tokens = 300)
print(f"test chat model: {chat_model.invoke('When is christmas shutdown?')}")

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks


TEMPLATE = """
You are an AI teaching assistant with expertise in Generative AI. You are answering questions specifically related to Generative AI and its impact on human life. Please only answer questions that are directly related to these topics. If you don't know the answer, it's fine to say so rather than guess. 

Use the provided context to help form your answer. If the context does not contain the answer, acknowledge this and be concise in your response.

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# Use simple LangChain chain to answer questions
chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

question = {"query": "when is Christmas shut down?"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation the model (optional)

# COMMAND ----------

eval_set = """question,ground_truth,evolution_type,episode_done
"What are the limitations of symbolic planning in task and motion planning, and how can leveraging large language models help overcome these limitations?","Symbolic planning in task and motion planning can be limited by the need for explicit primitives and constraints. Leveraging large language models can help overcome these limitations by enabling the robot to use language models for planning and execution, and by providing a way to extract and leverage knowledge from large language models to solve temporally extended tasks.",simple,TRUE
"What are some techniques used to fine-tune transformer models for personalized code generation, and how effective are they in improving prediction accuracy and preventing runtime errors? ","The techniques used to fine-tune transformer models for personalized code generation include ﬁne-tuning transformer models, adopting a novel approach called Target Similarity Tuning (TST) to retrieve a small set of examples from a training bank, and utilizing these examples to train a pretrained language model. The effectiveness of these techniques is shown in the improvement in prediction accuracy and the prevention of runtime errors.",simple,TRUE
How does the PPO-ptx model mitigate performance regressions in the few-shot setting?,"The PPO-ptx model mitigates performance regressions in the few-shot setting by incorporating pre-training and fine-tuning on the downstream task. This approach allows the model to learn generalizable features and adapt to new tasks more effectively, leading to improved few-shot performance.",simple,TRUE
How can complex questions be decomposed using successive prompting?,"Successive prompting is a method for decomposing complex questions into simpler sub-questions, allowing language models to answer them more accurately. This approach was proposed by Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner in their paper 'Successive Prompting for Decomposing Complex Questions', presented at EMNLP 2022.",simple,TRUE
"Which entity type in Named Entity Recognition is likely to be involved in information extraction, question answering, semantic parsing, and machine translation?",Organization,reasoning,TRUE
What is the purpose of ROUGE (Recall-Oriented Understudy for Gisting Evaluation) in automatic evaluation methods?,"ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is used in automatic evaluation methods to evaluate the quality of machine translation. It calculates N-gram co-occurrence statistics, which are used to assess the similarity between the candidate text and the reference text. ROUGE is based on recall, whereas BLEU is based on accuracy.",simple,TRUE
"What are the challenges associated with Foundation SSL in CV, and how do they relate to the lack of theoretical foundation, semantic understanding, and explicable exploration?","The challenges associated with Foundation SSL in CV include the lack of a profound theory to support all kinds of tentative experiments, and further exploration has no handbook. The pretrained LM may not learn the meaning of the language, relying on corpus learning instead. The models cannot reach a better level of stability and match different downstream tasks, and the primary method is to increase data, improve computation power, and design training procedures to achieve better results. The lack of theoretical foundation, semantic understanding, and explicable exploration are the main challenges in Foundation SSL in CV.",simple,TRUE
How does ChatGPT handle factual input compared to GPT-3.5?,"ChatGPT handles factual input better than GPT-3.5, with a 21.9% increase in accuracy when the premise entails the hypothesis. This is possibly related to the preference for human feedback in ChatGPT's RLHF design during model training.",simple,TRUE
What are some of the challenges in understanding natural language commands for robotic navigation and mobile manipulation?,"Some challenges in understanding natural language commands for robotic navigation and mobile manipulation include integrating natural language understanding with reinforcement learning, understanding natural language directions for robotic navigation, and mapping instructions and visual observations to actions with reinforcement learning.",simple,TRUE
"How does chain of thought prompting elicit reasoning in large language models, and what are the potential applications of this technique in neural text generation and human-AI interaction?","The context discusses the use of chain of thought prompting to elicit reasoning in large language models, which can be applied in neural text generation and human-AI interaction. Specifically, researchers have used this technique to train language models to generate coherent and contextually relevant text, and to create transparent and controllable human-AI interaction systems. The potential applications of this technique include improving the performance of language models in generating contextually appropriate responses, enhancing the interpretability and controllability of AI systems, and facilitating more effective human-AI collaboration.",simple,TRUE
"Using the given context, how can the robot be instructed to move objects around on a tabletop to complete rearrangement tasks?","The robot can be instructed to move objects around on a tabletop to complete rearrangement tasks by using natural language instructions that specify the objects to be moved and their desired locations. The instructions can be parsed using functions such as parse_obj_name and parse_position to extract the necessary information, and then passed to a motion primitive that can pick up and place objects in the specified locations. The get_obj_names and get_obj_pos APIs can be used to access information about the available objects and their locations in the scene.",reasoning,TRUE
"How can searching over an organization's existing knowledge, data, or documents using LLM-powered applications reduce the time it takes to complete worker activities?","Searching over an organization's existing knowledge, data, or documents using LLM-powered applications can reduce the time it takes to complete worker activities by retrieving information quickly and efficiently. This can be done by using the LLM's capabilities to search through large amounts of data and retrieve relevant information in a short amount of time.",simple,TRUE
"""

import pandas as pd
from io import StringIO

obj = StringIO(eval_set)
eval_df = pd.read_csv(obj)
display(eval_df)

# COMMAND ----------

from datasets import Dataset


test_questions = eval_df["question"].values.tolist()
test_groundtruths = eval_df["ground_truth"].values.tolist()

answers = []
contexts = []

# answer each question in the dataset
for question in test_questions:
    # save the answer generated
    chain_response = chain.invoke({"query" : question})
    answers.append(chain_response["result"])
    
    # save the contexts used
    vs_response = vectorstore.invoke(question)
    contexts.append(list(map(lambda doc: doc.page_content, vs_response)))

# construct the final dataset
response_dataset = Dataset.from_dict({
    "inputs" : test_questions,
    "answer" : answers,
    "context" : contexts,
    "ground_truth" : test_groundtruths
})

display(response_dataset.to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC Calcuate Evaluation Metrics

# COMMAND ----------

import mlflow
from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

dbrx_answer_similarity = mlflow.metrics.genai.answer_similarity(
    model="endpoints:/databricks-dbrx-instruct"
)

dbrx_relevance = mlflow.metrics.genai.relevance(
    model="endpoints:/databricks-dbrx-instruct"   
)

results = mlflow.evaluate(
        data=response_dataset.to_pandas(),
        targets="ground_truth",
        predictions="answer",
        extra_metrics=[dbrx_answer_similarity, dbrx_relevance],
        evaluators="default",
    )

display(results.tables['eval_results_table'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Save the Model to Model Registery in UC

# COMMAND ----------

import mlflow
import langchain
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# set model registery to UC
mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog_name}.{schema_name}.{our_model_name}"

with mlflow.start_run(run_name=our_model_name) as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever, 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# Move the model in production
client = mlflow.tracking.MlflowClient()
print(f"Registering model version {model_info.registered_model_version} as production model")
client.set_registered_model_alias(
    name=model_name,
    alias="Production",
    version=model_info.registered_model_version
)

# COMMAND ----------

# This might need to be done in the UI as it sometimes errors.
# TODO: Add env vars to the serving (current doing this via UI)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

# Initialize the workspace client
workspace = WorkspaceClient()

# Define the model name and endpoint name
endpoint_name = our_model_name

# Create the serving endpoint
status = workspace.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        served_models=[
            ServedModelInput(
                name=our_model_name,
                model_name=model_name,
                model_version=model_info.registered_model_version,
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
)

# Print the status of the endpoint
print(status)

# COMMAND ----------

# test deployed model
import json
import requests

serving_endpoint_url = "https://adb-3800464929700097.17.azuredatabricks.net/serving-endpoints/stu_rag_model_2/invocations"
input_data = {"instances": [{"query": "when is Christmas shut down?"}]}

# setup header
access_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

response = requests.post(
    serving_endpoint_url,
    headers=headers,
    data=json.dumps(input_data)
)

if response.status_code == 200:
    prediction = response.json()
    print(f"Result: {prediction}")
else:
    print(f"Error: {response.status_code}, {response.text}")