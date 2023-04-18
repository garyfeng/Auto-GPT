"""Base class for memory providers."""
import abc
from config import AbstractSingleton, Config
import openai
from logger import logger

cfg = Config()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    # very crude handling of OpenAPI errors: we simply return none
    try:
        if cfg.use_azure:
            return openai.Embedding.create(input=[text], engine=cfg.azure_embeddigs_deployment_id, model="text-embedding-ada-002")["data"][0]["embedding"]
        else:
            return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]
    except openai.error.APIError as e:
        #Handle API error here, e.g. retry or log
        logger.debug(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        #Handle connection error here
        logger.debug(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.error.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        logger.debug(f"OpenAI API request exceeded rate limit: {e}")
        pass
    # in case of the above errors, returns none
    return None
    


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def get(self, data):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
