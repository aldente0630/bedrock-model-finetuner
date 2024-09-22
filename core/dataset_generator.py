"""
This module provides classes for generating Q&A datasets for fine-tuning language models.

The module includes the following main components:
- BaseQaDatasetGenerator: A base class for Q&A dataset generation
- QaDatasetGenerator: A class that extends BaseQaDatasetGenerator with additional functionality

Key features:
- Generation of Q&A pairs from given contexts and metadata
- Parsing and formatting of Q&A outputs
- Sampling and filtering of generated data
- Local saving and S3 uploading of datasets

Usage:
    llm = get_llm(...)
    generator = QaDatasetGenerator.from_jsonl(llm, "documents.jsonl")
    dataset = generator.generate()
    generator.save_and_upload(dataset, "dataset.jsonl", boto_session, "my-bucket")

Dependencies:
- boto3
- langchain
- json
- random
- typing
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
from langchain.base_language import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain_community.callbacks.manager import get_bedrock_anthropic_callback

from .logger import Loggable
from .prompts import get_qa_generation_prompt
from .utils import get_s3_uri, load_jsonl, measure_execution_time, save_jsonl

MAX_TRAINING_LINES = 10000
MAX_VALIDATION_LINES = 1000
DEFAULT_DATASETS_PREFIX = "datasets"
DEFAULT_SYSTEM_PROMPT = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
"""


class BaseQaDatasetGenerator:
    """
    Base class for generating Q&A datasets.

    This class provides the basic structure and functionality for generating
    question-answer pairs from given contexts and metadata.

    Attributes:
        llm_chain: A chain of language model operations for generating Q&A pairs.

    Methods:
        parse_qa_output: Static method to parse the output from the language model.
    """

    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize the BaseQaDatasetGenerator.

        Args:
            llm (BaseLanguageModel): The language model to use for generating Q&A pairs.
        """
        self.llm_chain = (
            get_qa_generation_prompt() | llm | StrOutputParser() | self.parse_qa_output
        )

    @staticmethod
    def parse_qa_output(output: str) -> Dict[str, Any]:
        """
        Parse the output from the language model into a structured format.

        Args:
            output (str): The raw output string from the language model.

        Returns:
            Dict[str, Any]: A dictionary containing parsed Q&A pairs or an error message.
        """
        output_start_token = "<output>"
        output_end_token = "</output>"

        json_str = output.strip()
        if output_start_token in json_str and output_end_token in json_str:
            start = json_str.index(output_start_token) + len(output_start_token)
            end = json_str.rindex(output_end_token)
            json_str = json_str[start:end].strip()
        else:
            json_str = output.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"qa_pairs": [], "error": "Failed to parse output"}


class QaDatasetGenerator(BaseQaDatasetGenerator, Loggable):
    """
    A class for generating Q&A datasets with advanced functionality.

    This class extends BaseQaDatasetGenerator and Loggable to provide a comprehensive
    solution for generating, formatting, and managing Q&A datasets.

    Attributes:
        contexts (List[str]): List of context strings for generating Q&A pairs.
        metadata (List[str]): List of metadata strings corresponding to each context.

    Methods:
        from_jsonl: Class method to create an instance from a JSONL file.
        generate: Generate Q&A pairs from the given contexts and metadata.
        save_and_upload: Save the generated dataset locally and optionally upload to S3.
    """

    def __init__(
        self, llm: BaseLanguageModel, contexts: List[str], metadata: List[str]
    ):
        """
        Initialize the QaDatasetGenerator.

        Args:
            llm (BaseLanguageModel): The language model to use for generating Q&A pairs.
            contexts (List[str]): List of context strings.
            metadata (List[str]): List of metadata strings.
        """
        BaseQaDatasetGenerator.__init__(self, llm)
        Loggable.__init__(self)
        self.contexts = contexts
        self.metadata = metadata

    @staticmethod
    def _extract_metadata(
        documents: List[Dict[str, Any]], extra_fields: List[str]
    ) -> List[str]:
        """
        Extract metadata from documents based on specified fields.

        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries.
            extra_fields (List[str]): List of field names to extract.

        Returns:
            List[str]: Extracted metadata strings.
        """
        return [
            (
                " > ".join(
                    filter(
                        None,
                        [
                            str(
                                document.get(field.split(".")[0], {}).get(
                                    field.split(".")[1], ""
                                )
                            )
                            for field in extra_fields
                        ],
                    )
                )
                if extra_fields
                else ""
            )
            for document in documents
        ]

    @classmethod
    def from_jsonl(
        cls,
        llm: BaseLanguageModel,
        documents_path: str,
        content_field: str = "page_content",
        extra_fields: List[str] = [],
        **kwargs,
    ) -> "QaDatasetGenerator":
        """
        Create a QaDatasetGenerator instance from a JSONL file.

        Args:
            llm (BaseLanguageModel): The language model to use.
            documents_path (str): Path to the JSONL file containing documents.
            content_field (str): Field name for the main content in documents.
            extra_fields (List[str]): Additional fields to extract as metadata.
            **kwargs: Additional keyword arguments.

        Returns:
            QaDatasetGenerator: An instance of QaDatasetGenerator.
        """
        documents = load_jsonl(documents_path)
        contexts = [doc[content_field] for doc in documents]
        metadata = cls._extract_metadata(documents, extra_fields)
        return cls(llm, contexts, metadata, **kwargs)

    def _format_for_claude(
        self, qa_sets: List[Dict[str, Any]], system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> List[Dict[str, Any]]:
        """
        Format Q&A sets for Claude model input.

        Args:
            qa_sets (List[Dict[str, Any]]): List of Q&A set dictionaries.
            system_prompt (str): System prompt to include in formatted output.

        Returns:
            List[Dict[str, Any]]: Formatted Q&A sets.
        """
        formatted_qa_sets = []
        for qa_set in qa_sets:
            if isinstance(qa_set, dict) and "qa_pairs" in qa_set:
                for qa in qa_set["qa_pairs"]:
                    formatted = self._format_qa_pair(qa, system_prompt)
                    if formatted:
                        messages = formatted["messages"]
                        if len(messages) > 1:
                            is_valid = all(
                                messages[i]["role"] != messages[i + 1]["role"]
                                for i in range(len(messages) - 1)
                            )
                            if is_valid and messages[0]["role"] == "user":
                                formatted_qa_sets.append(formatted)
        return formatted_qa_sets

    @staticmethod
    def _format_qa_pair(
        qa: Dict[str, Any], system_prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Format a single Q&A pair.

        Args:
            qa (Dict[str, Any]): A dictionary containing a Q&A pair.
            system_prompt (str): System prompt to include in formatted output.

        Returns:
            Optional[Dict[str, Any]]: Formatted Q&A pair or None if invalid.
        """
        formatted = {"system": system_prompt, "messages": []}

        if qa.get("type") == "single-turn":
            question, answer = qa.get("question"), qa.get("answer")
            if question and answer:
                formatted["messages"] = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
        elif qa.get("type") == "multi-turn":
            formatted["messages"] = [
                {"role": turn["role"], "content": turn["content"]}
                for turn in qa.get("turns", [])
                if "role" in turn and "content" in turn
            ]

        return formatted if formatted["messages"] else None

    def _generate_qa_pairs(
        self, contexts: List[str], metadata: List[str], max_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs from given contexts and metadata.

        Args:
            contexts (List[str]): List of context strings.
            metadata (List[str]): List of metadata strings.
            max_workers (int): Maximum number of concurrent workers.

        Returns:
            List[Dict[str, Any]]: Generated Q&A pairs.
        """
        with get_bedrock_anthropic_callback() as cb:
            results = self.llm_chain.batch(
                inputs=[
                    {"context": context, "metadata": meta}
                    for context, meta in zip(contexts, metadata)
                ],
                config={"max_concurrency": max_workers},
            )
            self.logger.info("Total Tokens: %d", cb.total_tokens)
            self.logger.info("Prompt Tokens: %d", cb.prompt_tokens)
            self.logger.info("Completion Tokens: %d", cb.completion_tokens)
            self.logger.info("Total Cost (USD): %.4f", cb.total_cost)

        self.logger.info("Generated %d results", len(results))
        return results

    def _sample_data(self, sampling_rate: float) -> Tuple[List[str], List[str]]:
        """
        Sample data based on the given sampling rate.

        Args:
            sampling_rate (float): The rate at which to sample data.

        Returns:
            Tuple[List[str], List[str]]: Sampled contexts and metadata.
        """
        if sampling_rate < 1.0:
            num_samples = int(len(self.contexts) * sampling_rate)
            sampled_indices = random.sample(range(len(self.contexts)), num_samples)
            return [self.contexts[i] for i in sampled_indices], [
                self.metadata[i] for i in sampled_indices
            ]
        return self.contexts, self.metadata

    def _sample_final_results(
        self,
        formatted_results: List[Dict[str, Any]],
        dataset_type: str,
        sample_final_results: bool,
    ) -> List[Dict[str, Any]]:
        """
        Sample final results based on dataset type and sampling flag.

        Args:
            formatted_results (List[Dict[str, Any]]): Formatted Q&A pairs.
            dataset_type (str): Type of dataset ('train' or 'validation').
            sample_final_results (bool): Whether to sample final results.

        Returns:
            List[Dict[str, Any]]: Sampled final results.
        """
        max_lines = (
            MAX_TRAINING_LINES if dataset_type == "train" else MAX_VALIDATION_LINES
        )
        if sample_final_results and len(formatted_results) > max_lines:
            self.logger.warning(
                "Dataset size exceeds %d lines. Randomly sampling %d lines.",
                max_lines,
                max_lines,
            )
            return random.sample(formatted_results, max_lines)
        return formatted_results

    @measure_execution_time
    def generate(
        self,
        sampling_rate: float = 1.0,
        max_workers: int = 4,
        dataset_type: str = "train",
        sample_final_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate Q&A dataset.

        Args:
            sampling_rate (float): Rate at which to sample input data.
            max_workers (int): Maximum number of concurrent workers.
            dataset_type (str): Type of dataset ('train' or 'validation').
            sample_final_results (bool): Whether to sample final results.

        Returns:
            List[Dict[str, Any]]: Generated Q&A dataset.
        """
        contexts, metadata = self._sample_data(sampling_rate)
        self.logger.info("Using %d samples for dataset generation", len(contexts))

        results = self._generate_qa_pairs(contexts, metadata, max_workers)
        formatted_results = self._format_for_claude(results)
        self.logger.info("Formatted %d results", len(formatted_results))

        return self._sample_final_results(
            formatted_results, dataset_type, sample_final_results
        )

    def _save_locally(self, dataset: List[Dict[str, Any]], dataset_path: str):
        """
        Save the dataset locally.

        Args:
            dataset (List[Dict[str, Any]]): The dataset to save.
            dataset_path (str): Path where the dataset will be saved.
        """
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        save_jsonl(dataset_path, dataset)
        self.logger.info("Dataset saved locally to: %s", dataset_path)

    def _upload_to_s3(
        self,
        dataset_path: str,
        boto_session: boto3.Session,
        bucket_name: str,
        datasets_prefix: Optional[str],
    ) -> str:
        """
        Upload the dataset to S3.

        Args:
            dataset_path (str): Local path of the dataset file.
            boto_session (boto3.Session): Boto3 session for S3 operations.
            bucket_name (str): Name of the S3 bucket.
            datasets_prefix (Optional[str]): Prefix for the S3 key.

        Returns:
            str: S3 URI of the uploaded dataset.

        Raises:
            ClientError: If the upload to S3 fails.
        """
        try:
            s3_client = boto_session.client("s3")
            key = os.path.basename(dataset_path)
            if datasets_prefix:
                key = f"{datasets_prefix.rstrip('/')}/{key}"

            s3_client.upload_file(dataset_path, bucket_name, key)
            s3_uri = get_s3_uri(bucket_name, key)
            self.logger.info("Dataset uploaded to S3: %s", s3_uri)
            return s3_uri
        except ClientError as e:
            self.logger.error("Failed to upload dataset to S3: %s", str(e))
            raise

    def save_and_upload(
        self,
        dataset: List[Dict[str, Any]],
        dataset_path: str,
        boto_session: Optional[boto3.Session] = None,
        bucket_name: Optional[str] = None,
        datasets_prefix: Optional[str] = DEFAULT_DATASETS_PREFIX,
    ) -> str:
        """
        Save the dataset locally and optionally upload to S3.

        Args:
            dataset (List[Dict[str, Any]]): The dataset to save and upload.
            dataset_path (str): Local path to save the dataset.
            boto_session (Optional[boto3.Session]): Boto3 session for S3 operations.
            bucket_name (Optional[str]): Name of the S3 bucket.
            datasets_prefix (Optional[str]): Prefix for the S3 key.

        Returns:
            str: Path of the saved dataset (local path or S3 URI).
        """
        self._save_locally(dataset, dataset_path)
        if boto_session and bucket_name:
            return self._upload_to_s3(
                dataset_path, boto_session, bucket_name, datasets_prefix
            )
        return dataset_path
