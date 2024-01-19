import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
import torch
from langchain_community.vectorstores import Chroma

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, DistilBertForQuestionAnswering, pipeline

model = None
if not model:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    model.train()

print ("It started again,,,,,,....")
# Load model directly

pipe = pipeline("question-answering", 
                model=model, 
                tokenizer=tokenizer,
                temperature=0.2,
                max_new_tokens=1024)

llm = HuggingFacePipeline(
    pipeline = pipe,
    model_kwargs={"temperature": 0.2, 
                  "max_length": 1024},
)

persist_directory = './vector_db/chroma/'

with st.sidebar:
    st.title = "Chat with data!"
    st.markdown('''
                ### About
                This is a sample implementation of chat with data built with straemlit, langchain and Chroma
                ''')
    add_vertical_space(5)
    st.write("Demonstration..")

def main():
    st.write("Hello, this is document chatbot !")

    pdf = st.file_uploader("Upload your pdf !!")

    if pdf:
        f = open('uploaded_files/'+pdf.name, "wb")
        f.write(pdf.getvalue())
        f.close()
        loader = PyPDFLoader('uploaded_files/'+pdf.name)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100, length_function=len )

        vectordb = Chroma.from_documents(
            documents=text_splitter.split_documents(pages),
            embedding=embeddings,
            persist_directory=persist_directory
        )

        query = st.text_input("Enter your query")
        
        matching_docs = vectordb.similarity_search(query, k=3)

        prompt = '''
        [CLS]Use the context as the only input and generate minimum 250 words answer.
        Donot generate sentences outside the context.  

        Question : ''' + query
        if query:
            context = '' 
            for i in matching_docs:
                context += " " + i.page_content

            inputs = tokenizer(prompt, context[:512], return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            answer_start_index = outputs.start_logits.argmax()
            answer_end_index = outputs.end_logits.argmax()
            if answer_start_index > answer_end_index:
                answer_end_index, answer_start_index = answer_start_index, answer_end_index
            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            ans = tokenizer.decode(predict_answer_tokens)
            
            # st.write(ans)

            if "[SEP]" in ans and ans.index("[SEP]") > 0:
                ans = ans[ans.index("[SEP]")+5:]
            st.write(ans)

if __name__ == '__main__':
    main()