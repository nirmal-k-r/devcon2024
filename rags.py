from langchain_community.llms import Ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class AugmentedAI:
    def __init__(self,prompt=None,logger=False,model="llama3", gpu="mps",data_directory="data/"):
        if prompt is None:
            self.prompt = """
            1. Use the following parts of context to answer the question at the end.
            2. If you don't know the answer, then just say that "I don't know" but don't make up an answer on your own.
            3. Keep the answer clear and limited to 3 or 4 sentences. There is no need to say "According to the context"

            Context: {context}

            Question: {question}

            Helpful Answer:"""
        
        self.logger = logger

        #core components
        self.embedder = HuggingFaceEmbeddings(model_kwargs={'device':gpu}) #swap to metal GPU
        self.llm = Ollama(model=model) #ollama3 model
        self.retriever=None #set later on
        self.contextual_prompt = PromptTemplate.from_template(self.prompt) #set contextual prompt

        #create pipeline
        self.load_documents(directory=data_directory)
        self.create_vector_embeddings()
        self.create_llm_pipeline()
        

    def load_documents(self,directory):
        loader = DirectoryLoader("data/", glob="**/*.txt", loader_cls=TextLoader,use_multithreading=True,show_progress=False)
        self.docs = loader.load()

        if self.logger==True:
            # Check the number of pages
            print("Number of pages in ingested data:",len(self.docs))


    def create_vector_embeddings(self):
        #split the documents into chunks and create embeddings
        self.text_splitter = SemanticChunker(self.embedder)
        self.documents = self.text_splitter.split_documents(self.docs)

        # Check number of chunks created
        if self.logger==True:
            print("Number of chunks created: ", len(self.documents))

            # Printing first few chunks
            for i in range(len(self.documents)):
                print()
                print(f"CHUNK : {i+1}")
                print(self.documents[i].page_content)

        # Create the vector embedding store 
        vector = FAISS.from_documents(self.documents, self.embedder)
        self.retriever=vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    #create the LLM pipeline
    def create_llm_pipeline(self):
        self.llm_chain = LLMChain(
                  llm=self.llm, 
                  prompt=self.contextual_prompt, 
                  callbacks=None, 
                  verbose=True)

        self.rag_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )

        self.retrieval = StuffDocumentsChain(
                        llm_chain=self.llm_chain,
                        document_variable_name="context",
                        document_prompt=self.rag_prompt,
                        callbacks=None)
        
        #chat with the AI
        self.chat=RetrievalQA(
                combine_documents_chain=self.retrieval,
                verbose=self.logger,
                retriever=self.retriever,
                return_source_documents=self.logger)
    
    
ai=AugmentedAI(logger=True)
result=ai.chat("Does paul graham call animals?")
print(result['result'])
