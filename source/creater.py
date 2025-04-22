from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from source.prompt import *

# OpenAI authetication 
load_dotenv()
OPEN_API_KEY = os.getenv["OPEN_API_KEY"]
os.environ["OPEN_API_KEY"] = OPEN_API_KEY

class llm_pipeline:
    def file_processing(file_path):
        #load the data
        loader = PyPDFLoader(file_path=file_path)
        data = loader.load()

        questions_generater = "" 
        
        for page in data:
            questions_generater += page.page_content

        split_questions_generater = TokenTextSplitter(
            model_name= "gpt-3.5-turbo",
            chunk_size =10000,
            chunk_overlap = 200
        )
        chunks_ques_generator = split_questions_generater.split_text(questions_generater)

        document = [Document(page_content= i) for i in chunks_ques_generator]


        split_answer_generater= TokenTextSplitter(
            model_name = "gpt-3.5-turbo",
            chunk_size =1000,
            chunk_overlap = 100
        )
        chunks_ans_generater = split_answer_generater.split_documents(document)

        return chunks_ques_generator, chunks_ans_generater, document


    def pipline_llms(file_path):

        chunks_ques_generator, chunks_ans_generater = file_processing(file_path)

        llm_que_gen_pipline = ChatOpenAI(
            model = "gpt-3.5-turbo",
            temperature = 0.3
        )

        PROMPT_QUESTION = PromptTemplate(template=prompt_template, input_variables=["text"])

        REFINE_PROMPT_TEMPLATE = PromptTemplate(
            input_variables= ["existing", "text"],
            template= refine_template
        )

        que_gen_chain = load_summarize_chain(llm= llm_que_gen_pipline,
                                            chain_type= "refine",
                                            verbose=True,
                                            question_prompt = PROMPT_QUESTION,
                                            refine_template = REFINE_PROMPT_TEMPLATE
                                            )

        questions = que_gen_chain.run(chunks_ques_generator)

        embeddings = OpenAIEmbeddings()

        vector_stores = FAISS.from_documents(chunks_ans_generater, embeddings)

        llm_ans_generate = ChatOpenAI(model = "gpt-3.5-turbo",
                                    temepreture = 0.1
                                    )

        question_list = questions.split("\n")

        filtered_ques_list = [element for element in question_list if element.endwith("?") or element.endwith(".")]

        answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_ans_generate, 
                                                        chain_type="stuff", 
                                                        retriever=vector_stores.as_retriever())

        return answer_generation_chain, filtered_ques_list


