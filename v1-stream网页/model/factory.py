from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=rag_conf["base_url"],
            api_key=rag_conf["api_key"]
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return HuggingFaceEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
