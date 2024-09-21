"""
This module provides utility functions for working with language models, S3, and file operations.

The module includes the following main components:
- get_llm: Creates and configures a language model instance
- get_s3_uri: Generates S3 URIs
- measure_execution_time: A decorator for measuring function execution time
- load_jsonl: Loads data from a JSONL file
- save_jsonl: Saves data to a JSONL file

These functions are designed to support various operations in the fine-tuning process,
including model configuration, data handling, and performance measurement.
"""

import functools
import json
import time
from typing import Callable, Dict, List, Optional
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackHandler
from langchain_aws import ChatBedrock
from .constants import ChatModelId


def get_llm(
    chat_model_id: ChatModelId,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    profile_name: Optional[str] = None,
    region_name: Optional[str] = None,
    top_k: int = 50,
    top_p: float = 0.95,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    **kwargs,
) -> BaseLanguageModel:
    """
    Create and configure a language model instance.

    Args:
        chat_model_id (ChatModelId): The ID of the chat model to use.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 4096.
        temperature (float): Sampling temperature. Defaults to 0.2.
        profile_name (Optional[str]): AWS profile name. Defaults to None.
        region_name (Optional[str]): AWS region name. Defaults to None.
        top_k (int): Top-k sampling parameter. Defaults to 50.
        top_p (float): Top-p sampling parameter. Defaults to 0.95.
        callbacks (Optional[List[BaseCallbackHandler]]): List of callback handlers. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        BaseLanguageModel: Configured language model instance.
    """
    common_params = {
        "model": chat_model_id.value,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "region_name": region_name,
        "credentials_profile_name": profile_name,
        "callbacks": callbacks,
        **kwargs,
    }

    try:
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(stop=["\n\nHuman:"], **common_params)
    except ImportError:
        return ChatBedrock(
            model_id=chat_model_id.value,
            model_kwargs={
                "stop_sequences": ["\n\nHuman:"],
                "top_k": top_k,
                **{
                    k: v
                    for k, v in common_params.items()
                    if k in ["max_tokens", "temperature", "top_p"]
                },
            },
            **{
                k: v
                for k, v in common_params.items()
                if k not in ["max_tokens", "temperature", "top_p"]
            },
        )


def get_s3_uri(bucket_name: str, key: Optional[str] = None) -> str:
    """
    Generate an S3 URI from a bucket name and optional key.

    Args:
        bucket_name (str): The name of the S3 bucket.
        key (Optional[str]): The object key within the bucket. Defaults to None.

    Returns:
        str: The generated S3 URI.
    """
    return f"s3://{bucket_name}/{key}" if key else f"s3://{bucket_name}"


def measure_execution_time(func: Callable) -> Callable:
    """
    A decorator that measures and prints the execution time of a function.

    Args:
        func (Callable): The function to be measured.

    Returns:
        Callable: The wrapped function that measures execution time.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds.")
        return result

    return wrapper


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        List[Dict]: A list of dictionaries, each representing a JSON object from the file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(file_path: str, data: List[Dict]) -> None:
    """
    Save data to a JSONL file.

    Args:
        file_path (str): The path to save the JSONL file.
        data (List[Dict]): A list of dictionaries to be saved as JSON objects.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
