import logging
import os
import streamlit as st
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVING_ENDPOINT = "https://adb-3800464929700097.17.azuredatabricks.net/serving-endpoints/stu_rag_model_2/invocations"

# Function to get user info from headers
def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# Streamlit session state for chat visibility and history
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.image(
    "https://mantelgroup.com.au/wp-content/uploads/2023/03/Mantel-Group-Logo-Pride-e1622092187264-1.png", 
    caption="Mantel Group", 
    width=300  # Adjust the width to your preference
)
st.title("Mantel AI Assistant")
st.write(f"A basic chatbot using your own serving endpoint")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to query the Databricks serving endpoint
def query_databricks_endpoint(query):
    input_data = {"instances": [{"query": query}]}

    # Setup headers for the request
    access_token = os.getenv('DATABRICKS_CLIENT_SECRET') ##TODO: fix this
    access_token = "<CHANGEME>"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Make the POST request
    response = requests.post(
        SERVING_ENDPOINT,
        headers=headers,
        data=json.dumps(input_data)
    )

    # Handle the response
    if response.status_code == 200:
        prediction = response.json()
        return prediction
    else:
        raise Exception(f"Error querying model: {response.status_code}, {response.text}")

# Accept user input and interact with the assistant
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the Databricks serving endpoint
    try:
        assistant_response = query_databricks_endpoint(prompt)
        assistant_message = assistant_response['predictions'][0]
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(assistant_message)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    except Exception as e:
        st.error(f"Error querying model: {e}")