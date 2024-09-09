import os
import torch
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pprint import pprint


def split_text_files(folder_path: str) -> tuple[list[Document], dict[str: str]]:
    """
    Load txt files from the specified folder, split them into overlapping chunks and return the chunks as list of
    Documents. Additionally return a mapping between the chunk text and its source (file name + chunk id).
    """
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n")
    text_chunks = []
    for file in tqdm(os.listdir(folder_path), desc="splitting documents"):
        loader = TextLoader(f"{folder_path}/{file}", encoding="utf-8")
        text_chunks.extend(text_splitter.split_documents(loader.load()))
    chunk2source = {}
    for chunk_index, document in tqdm(enumerate(text_chunks), desc="indexing chunks"):
        source_file = document.metadata["source"]
        text = document.page_content
        if text not in chunk2source:
            chunk2source[text] = []
        chunk2source[text].append((source_file, chunk_index))
    for chunk, source_ids in chunk2source.items():
        source_ids = [f"{file} - chunk {chunk_id}" for file, chunk_id in source_ids]
        source_ids = "/".join(source_ids)
        chunk2source[chunk] = source_ids
    return text_chunks, chunk2source


def load_embedding_model(model_id: str) -> Embeddings:
    """
    Load the specified embedding model from Hugging Face.
    """
    return HuggingFaceBgeEmbeddings(model_name=model_id, model_kwargs={"device": "cuda:0"},
                                    encode_kwargs={"normalize_embeddings": True}, show_progress=True)


def load_quantized_text_generation_model(model_id: str) -> HuggingFacePipeline:
    """
    Load the specified generation model from Hugging Face. Apply quantization to reduce the memory load in inference.
    Wrap the generation model into a HuggingFacePipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_new_tokens=512)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, attn_implementation='eager')
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)


def build_retrieval_qa_chain(texts: list[Document], embedding_model: Embeddings, generation_model: HuggingFacePipeline) -> RetrievalQA:
    """
    Use the embedding model and generation model to build a RetrievalQA chain.
    First construct a vector database for the chunk texts using FAISS, then wrap the generation model and the vector
    database into a RetrievalQA.
    Select 'stuff' as chain_type, 'similarity' as search_type and NUM_CHUNKS as k.
    This means for each query, first the NUM_CHUNKS text chunks with the most similar embedding to the query are
    retrieved using the vector database, then these chunks are stuffed into the prompt as pieces of context for the
    instruction model to use to answer the query.
    """
    vector_db = FAISS.from_documents(texts, embedding_model)
    return RetrievalQA.from_chain_type(llm=generation_model, chain_type="stuff", return_source_documents=True,
                                       retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": NUM_CHUNKS}))


def answer_queries(queries: list[str], qa_chain: RetrievalQA, chunk2source: dict[str: str]) -> dict[str: str]:
    """
    Use the RetrievalQA chain to answer queries based on the provided files.
    After the model generates a retrieval-augmented response and the text of the corresponding file chunk, use the
    chunk2source mapping to get the file name and chunk index.
    Return a list of dicts with query, model answer and source chunk id for each query.
    """
    answers = []
    for query in queries:
        output = qa_chain.invoke(query)
        source_chunks = [doc.to_json()['kwargs']['page_content'] for doc in output["source_documents"]]
        source_chunk_ids = [chunk2source[c] for c in source_chunks]
        start_of_answer = f"\nQuestion: {query}\nHelpful Answer: "
        answer = output["result"][output["result"].index(start_of_answer) + len(start_of_answer):].strip()
        answers.append({"query": query, "answer": answer, "source_chunk_id": source_chunk_ids})
    return answers


if __name__ == '__main__':

    # configurations
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    NUM_CHUNKS = 5
    EMBEDDING_MODEL_ID = "BAAI/bge-m3"
    GENERATION_MODEL_ID = "google/gemma-2-9b-it"

    # queries to run through the retrieval chain (collected from various trivia quizzes)
    queries = ["Which wizard lived in Orthanc?",
               "What was the name of the inn in the village of Bree?",
               "Who married Aragorn?",
               "Which type of blade was Frodo stabbed with?",
               "What was Gollum's real name?",
               "What did Frodo see on the ring after Gandalf threw it into the fire?",
               "What was the full name of Pippin?",
               "What was Gandalf's sword's name?",
               "What food does Gollum like?",
               "Which eagle rescued Gandalf from the tower of Isengard?"]

    # read input txt files and split them to chunks
    text_chunks, text2source = split_text_files("source_documents")

    # load embedding model
    bge_embeddings = load_embedding_model(EMBEDDING_MODEL_ID)

    # load text generation model in 4-Bit mode
    generation_model = load_quantized_text_generation_model(GENERATION_MODEL_ID)

    # construct a RetrievalQA chain from embedding model, text chunks and generation model
    qa_chain = build_retrieval_qa_chain(text_chunks, bge_embeddings, generation_model)

    # run the queries through the RetrievalQA chain
    answers = answer_queries(queries, qa_chain, text2source)
    pprint(answers)
