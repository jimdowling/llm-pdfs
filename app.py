import streamlit as st
import hopsworks
from peft import AutoPeftModelForCausalLM
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import transformers
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from FlagEmbedding import FlagReranker
import torch
from functions.prompt_engineering import get_context_and_source
import warnings
warnings.filterwarnings('ignore')

st.title("💬 AI assistant")

@st.cache_resource()
def connect_to_hopsworks():
    # Initialize Hopsworks feature store connection
    project = hopsworks.login(
        host="snurran.hops.works",
        project="LLM"
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Retrieve the 'documents' feature view
    feature_view = fs.get_feature_view(
        name="documents", 
        version=1,
    )

    # Initialize serving
    feature_view.init_serving(1)
    
    # Get the Mistral model from Model Registry
    mistral_model = mr.get_model(
        name="mistral_model",
        version=1,
    )
    
    # Download the Mistral model files to a local directory
    saved_model_dir = mistral_model.download()

    return feature_view, saved_model_dir


@st.cache_resource()
def get_models(saved_model_dir):

    # Load the Sentence Transformer
    sentence_transformer = SentenceTransformer(
        'all-MiniLM-L6-v2',
    ).to('cuda')

    # Retrieve the fine-tuned model
    model = AutoPeftModelForCausalLM.from_pretrained(
      saved_model_dir,
      device_map="auto",
      torch_dtype=torch.float16,
    )

    # Retrieve the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        saved_model_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return sentence_transformer, model, tokenizer


@st.cache_resource()
def get_llm_chain(model, tokenizer):
    
    # Create a text generation pipeline using the loaded model and tokenizer
    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.5,
        return_full_text=True,
        max_new_tokens=750,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

    # Create a Hugging Face pipeline for Mistral LLM using the text generation pipeline
    mistral_llm = HuggingFacePipeline(
        pipeline=text_generation_pipeline,
    )

    # Define a template for generating prompts
    prompt_template = """
[INST] 
Instruction: Prioritize brevity and clarity in responses. 
Avoid unnecessary repetition and keep answers concise, adhering to a maximum of 750 characters. 
Eliminate redundant phrases and sentences. 
If details are repeated, provide them only once for better readability. 
Focus on delivering key information without unnecessary repetition. 
If a concept is already conveyed, there's no need to restate it. Ensure responses remain clear and to the point.
Make sure you do not repeat any sentences in your answer.
[/INST]

Previous conversation:
{chat_history}

### CONTEXT:

{context}

### QUESTION:
[INST]{question}[/INST]"""

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_template,
    )
    
    # Create a ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history", 
        input_key="question",
    )
    
    # Create the LLM chain 
    llm_chain = LLMChain(
        llm=mistral_llm, 
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    return llm_chain


@st.cache_resource()
def get_reranker():
    reranker = FlagReranker(
        'BAAI/bge-reranker-large', 
        use_fp16=True,
    ) 
    return reranker


def predict(user_query, sentence_transformer, feature_view, reranker, llm_chain):
    
    st.write('⚙️ Generating Response...')
    
    # Retrieve reranked context and source
    context, source = get_context_and_source(
        user_query, 
        sentence_transformer,
        feature_view, 
        reranker,
    )
    
    # Generate model response
    model_output = llm_chain.invoke({
        "context": context, 
        "question": user_query,
    })

    return model_output['text'].split('### RESPONSE:\n')[-1] + source

# Retrieve the feature view and the saved_model_dir
feature_view, saved_model_dir = connect_to_hopsworks()

# Load and retrieve the sentence_transformer, fine-tuned model and corresponding tokenizer
sentence_transformer, model, tokenizer = get_models(saved_model_dir)

# Create the LLM Chain
llm_chain = get_llm_chain(model, tokenizer)

# Retrieve the reranking model
reranker = get_reranker()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_query := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    response = predict(
        user_query, 
        sentence_transformer, 
        feature_view,
        reranker,
        llm_chain,
    )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
