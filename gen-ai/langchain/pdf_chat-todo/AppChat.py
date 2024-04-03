import torch

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from pdfminer.high_level import extract_text

model = None
if not model:
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad', temperature=0)

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
        text = extract_text('uploaded_files/'+pdf.name)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents([text])
        vectordb = Chroma(persist_directory=persist_directory).from_documents(docs)
        retriever = vectordb.as_retriever(search_kwargs={"k": 1})

        query = st.text_input("Enter your query")
        
        pages = retriever.get_relevant_documents(query, search_kwargs={"k": 1})
        context = ""
        import re 
        for p in pages: 
            context +=  '\n' +  re.sub('[\(\[].*?[\)\]]', "", p.page_content)
            context = context.replace('"', '')
        prompt = query
        if query:
            def query(context, question, verbose=True, context_max_len=512):
                print (context[:100])
                context = context[:context_max_len]
                # print (context)
                encoding = tokenizer.encode_plus(question, context)
                input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
                ret = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))  # bug work-around
                start_scores, end_scores = ret[0], ret[1]
                ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
                answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
                answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
                if verbose:
                    print ("\nQuestion: ",question)
                    print ("Answer: ", answer_tokens_to_string)
                return [question, answer_tokens_to_string]

            q,a = query(context=context, question=prompt)
            print (q, a)
            # st.write(ans)
            st.write(a)

if __name__ == '__main__':
    main()