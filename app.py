import streamlit as st
import stanza
import spacy
from flair.models import SequenceTagger
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import openai
import time

# Load Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize models only once to save resources
@st.cache_resource
def load_models():
    stanza.download('uk', processors='tokenize,ner')
    stanza_nlp = stanza.Pipeline(lang='uk', processors='tokenize,ner')
    spacy.cli.download("uk_core_news_sm")
    spacy_nlp = spacy.load("uk_core_news_sm")
    flair_tagger = SequenceTagger.load("lang-uk/flair-uk-ner")
    tokenizer = AutoTokenizer.from_pretrained("EvanD/xlm-roberta-base-ukrainian-ner-ukrner")
    ner_model = AutoModelForTokenClassification.from_pretrained("EvanD/xlm-roberta-base-ukrainian-ner-ukrner")
    nlp = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")
    return stanza_nlp, spacy_nlp, flair_tagger, nlp

stanza_nlp, spacy_nlp, flair_tagger, nlp = load_models()

# NER functions
def stanza_ner(text):
    doc = stanza_nlp(text)
    entities = [(entity.text, entity.type) for sentence in doc.sentences for entity in sentence.ents]
    return entities

def spacy_ner(text):
    doc = spacy_nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def flair_ner(text):
    sentence = flair.data.Sentence(text)
    flair_tagger.predict(sentence)
    entities = [(entity.text, entity.labels[0].value) for entity in sentence.get_spans('ner')]
    return entities

def gpt_ner(text):
    client = openai.OpenAI(api_key=openai.api_key)
    assistant = client.beta.assistants.retrieve("asst_0j2S7NlZWOKzpsrElrHphTbU")
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=text)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
    
    def loop_until_completed(clnt, thrd, run_obj):
        while run_obj.status not in ["completed", "failed", "requires_action"]:
            run_obj = clnt.beta.threads.runs.retrieve(thread_id=thrd.id, run_id=run_obj.id)
            time.sleep(10)
    
    loop_until_completed(client, thread, run)
    
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    return messages.data[0].content[0].text.value

# Streamlit UI
st.title("Ukrainian NER Web App")

input_text = st.text_area("Enter text in Ukrainian:")

if st.button("Analyze"):
    if input_text:
        st.write("Analyzing...")

        stanza_entities = stanza_ner(input_text)
        spacy_entities = spacy_ner(input_text)
        flair_entities = flair_ner(input_text)
        gpt_results = gpt_ner(input_text)
        xlm_roberta_results = nlp(input_text)
        formatted_xlm_roberta_results = [(result['word'], result['entity_group']) for result in xlm_roberta_results]

        st.subheader("GPT-3.5 NER:")
        st.write(gpt_results)

        st.subheader("Stanza NER:")
        st.write(stanza_entities)

        st.subheader("SpaCy NER:")
        st.write(spacy_entities)

        st.subheader("Flair NER:")
        st.write(flair_entities)

        st.subheader("XLM-RoBERTa NER:")
        st.write(formatted_xlm_roberta_results)
