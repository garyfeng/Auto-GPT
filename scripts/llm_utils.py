import openai
from logger import logger
from config import Config

cfg = Config()

openai.api_key = cfg.openai_api_key

# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    """Create a chat completion using the OpenAI API"""
    # very crude handling of OpenAPI errors: we simply return an empty string 
    try:
        if cfg.use_azure:
            response = openai.ChatCompletion.create(
                deployment_id=cfg.azure_chat_deployment_id,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

        return response.choices[0].message["content"]
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
    # in case of the above errors, returns an empty string
    return ""
    
