import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.voyageai import VoyageEmbeddings
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (RunnableBranch,
                                       RunnableLambda, RunnableMap)
from langchain.vectorstores.docarray import DocArrayInMemorySearch
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langsmith import Client
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

RESPONSE_TEMPLATE = """\
You are (gay) JRR Tolkein here to provide a bunch of homoerotic subtext to your GReat WOrks. 

BOY YOU CAN"T WAIT TO TALK BOUT GAY FRODO

<context>
    {context} 
<context/>


"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def get_embeddings_model() -> Embeddings:
    if os.environ.get("VOYAGE_API_KEY") and os.environ.get("VOYAGE_AI_MODEL"):
        return VoyageEmbeddings(model=os.environ["VOYAGE_AI_MODEL"])
    return OpenAIEmbeddings(chunk_size=200)


def get_text_from_url(
        url="https://raw.githubusercontent.com/ganesh-k13/shell/master/test_search/www.glozman.com/TextPages/01%20-%20The%20Fellowship%20Of%20The%20Ring.txt"):
    response = requests.get(url)
    response.raise_for_status()  # This will raise an exception if there's an error.
    return response.text


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    chunks = []
    start_index = 0
    text_as_words = text.split(" ")
    total_chunks = (len(text_as_words) - chunk_overlap) // (chunk_size - chunk_overlap)

    for _ in tqdm(range(total_chunks), desc="Splitting text into chunks"):
        end_index = min(start_index + chunk_size, len(text_as_words))
        chunks.append(" ".join(text_as_words[start_index:end_index]))
        start_index = end_index - chunk_overlap
        if end_index >= len(text_as_words):
            break

    return chunks


def get_retriever() -> BaseRetriever:
    big_text = get_text_from_url()

    chunks = split_text_into_chunks(big_text)

    # db = Chroma.from_texts(
    #     texts=chunks,
    #     embedding=get_embeddings_model(),
    # )
    # db_retriever = db.as_retriever()
    vectorstore = DocArrayInMemorySearch.from_texts(chunks, embedding=get_embeddings_model())
    retriever = vectorstore.as_retriever()

    return retriever


def create_retriever_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)

    condense_question_chain = (
            CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )

    conversation_chain = condense_question_chain | retriever

    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
                RunnableLambda(itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                )
                | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
            {
                "question": RunnableLambda(itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                ),
                "chat_history": RunnableLambda(serialize_history).with_config(
                    run_name="SerializeHistory"
                ),
            }
            | _context
            | response_synthesizer
    )


llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    streaming=True,
    temperature=0,
)
retriever = get_retriever()
answer_chain = create_chain(llm, retriever, )
