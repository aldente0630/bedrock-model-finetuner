"""
This module provides functionality for validating Q&A datasets used in fine-tuning language models.

The module includes the following main components:
- Message: A Pydantic model representing a single message in a conversation.
- DataEntry: A Pydantic model representing a single data entry, including system prompt and messages.
- QaDatasetValidator: A class for validating Q&A datasets, including file size, line count, and data structure.

Constants:
- MIN_LINES: Minimum number of lines required in a dataset file.
- MAX_TRAIN_LINES: Maximum number of lines allowed in a training dataset file.
- MAX_VALIDATION_LINES: Maximum number of lines allowed in a validation dataset file.
- MAX_TOTAL_LINES: Maximum total number of lines allowed across all dataset files.
- MAX_TOKENS: Maximum number of tokens allowed in a single data entry.
- CHARS_PER_TOKEN: Estimated number of characters per token for token counting.
- MAX_TRAIN_SIZE_GB: Maximum allowed size in GB for a training dataset file.
- MAX_VALIDATION_SIZE_GB: Maximum allowed size in GB for a validation dataset file.
- RESERVED_KEYWORDS: List of keywords that are not allowed in the dataset content.

This module helps ensure that the datasets used for fine-tuning meet the required
structure and constraints, improving the quality and consistency of the training data.
"""

import json
import os
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator, ValidationError

from .logger import Loggable

MIN_LINES = 32
MAX_TRAIN_LINES = 10000
MAX_VALIDATION_LINES = 1000
MAX_TOTAL_LINES = 10000
MAX_TOKENS = 32000
CHARS_PER_TOKEN = 3
MAX_TRAIN_SIZE_GB = 10
MAX_VALIDATION_SIZE_GB = 1
RESERVED_KEYWORDS = ["\nHuman:", "\nAssistant:"]


class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        role (str): The role of the message sender, either "user" or "assistant".
        content (str): The content of the message.
    """

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class DataEntry(BaseModel):
    """
    Represents a single data entry in the dataset, including system prompt and messages.

    Attributes:
        system (Optional[str]): The optional system prompt for the conversation.
        messages (List[Message]): A list of messages in the conversation.

    Raises:
        ValueError: If the message structure is invalid.
    """

    system: Optional[str] = None
    messages: List[Message] = Field(..., min_items=2)

    @model_validator(mode="after")
    def check_message_structure(self) -> "DataEntry":
        """
        Validates the structure of the messages in the data entry.

        Returns:
            DataEntry: The validated DataEntry object.

        Raises:
            ValueError: If the message structure is invalid.
        """
        if not self.messages:
            raise ValueError("Messages list cannot be empty")

        if self.messages[0].role != "user" or self.messages[-1].role != "assistant":
            raise ValueError("Messages must start with user and end with assistant")

        if any(
            self.messages[i].role == self.messages[i + 1].role
            for i in range(len(self.messages) - 1)
        ):
            raise ValueError("Messages must alternate between user and assistant")

        for keyword in RESERVED_KEYWORDS:
            if self.system and keyword in self.system:
                raise ValueError(f"Reserved keyword '{keyword}' found in system prompt")
            if any(keyword in message.content for message in self.messages):
                raise ValueError(
                    f"Reserved keyword '{keyword}' found in message content"
                )

        return self


class QaDatasetValidator(Loggable):
    """
    A class for validating Q&A datasets used in fine-tuning language models.

    This class provides methods to validate dataset files, including file size,
    line count, and data structure.
    """

    def __init__(self):
        """
        Initialize the QaDatasetValidator.
        """
        super().__init__()

    @staticmethod
    def _count_tokens(text: str) -> int:
        """
        Estimates the number of tokens in a given text.

        Args:
            text (str): The input text to count tokens for.

        Returns:
            int: The estimated number of tokens.
        """
        return len(text) // CHARS_PER_TOKEN

    def _log_validation_results(self, file_path: str, errors: List[str]) -> None:
        """
        Logs the validation results for a given file.

        Args:
            file_path (str): The path of the file being validated.
            errors (List[str]): A list of error messages, if any.
        """
        if errors:
            self.logger.error("Validation FAILED for %s. Errors:", file_path)
            for error in errors:
                self.logger.error("- %s", error)
        else:
            self.logger.info("Validation SUCCESSFUL for %s.", file_path)

    @staticmethod
    def _validate_data_entry(entry: dict) -> Tuple[bool, List[str]]:
        """
        Validates a single data entry in the dataset.

        Args:
            entry (dict): The data entry to validate.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating whether
            the entry is valid, and a list of error messages if any.
        """
        try:
            DataEntry(**entry)
            return True, []
        except ValidationError as e:
            errors = [
                (
                    "Structure error"
                    if error["type"] == "value_error"
                    else f"Field {'.'.join(str(loc) for loc in error['loc'])}"
                )
                + f": {error['msg']}"
                for error in e.errors()
            ]
            return False, errors

    def _validate_file(
        self, file_path: str, is_train: bool = True
    ) -> Tuple[List[str], int]:
        """
        Validates a single dataset file.

        This method performs various checks on the given file, including:
        - File size validation
        - JSON structure validation for each line
        - Token count validation for each entry
        - Line count validation

        Args:
            file_path (str): The path to the file to be validated.
            is_train (bool): Whether the file is a training dataset. Defaults to True.

        Returns:
            Tuple[List[str], int]: A tuple containing a list of error messages (if any)
            and the total number of lines in the file.
        """
        errors = []
        line_count = 0

        file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
        max_size_gb = MAX_TRAIN_SIZE_GB if is_train else MAX_VALIDATION_SIZE_GB
        if file_size_gb > max_size_gb:
            errors.append(
                f"{'Train' if is_train else 'Validation'} file size ({file_size_gb:.2f} GB) exceeds the maximum allowed size ({max_size_gb} GB)"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                try:
                    data = json.loads(line)
                    is_valid, entry_errors = self._validate_data_entry(data)
                    if not is_valid:
                        errors.extend(
                            f"Line {line_num}: {error}" for error in entry_errors
                        )

                    total_tokens = self._count_tokens(data.get("system", "")) + sum(
                        self._count_tokens(msg["content"]) for msg in data["messages"]
                    )
                    if total_tokens > MAX_TOKENS:
                        errors.append(
                            f"Line {line_num}: Exceeds maximum token count ({total_tokens} > {MAX_TOKENS})"
                        )
                except json.JSONDecodeError:
                    errors.append(f"Line {line_num}: Invalid JSON")

        max_lines = MAX_TRAIN_LINES if is_train else MAX_VALIDATION_LINES
        if not MIN_LINES <= line_count <= max_lines:
            errors.append(
                f"File has {line_count} lines. {'Train' if is_train else 'Validation'} data should have between {MIN_LINES} and {max_lines} lines."
            )

        return errors, line_count

    def validate_data(
        self, train_dataset_path: str, validation_dataset_path: Optional[str] = None
    ) -> None:
        """
        Validates both training and validation datasets.

        This method performs validation on the training dataset and optionally on the
        validation dataset if provided. It checks for various criteria including file
        structure, content validity, and adherence to size limits.

        Args:
            train_dataset_path (str): The path to the training dataset file.
            validation_dataset_path (Optional[str]): The path to the validation dataset file.
                If not provided, only the training dataset will be validated.

        Returns:
            None

        Raises:
            None, but logs errors, warnings, and information about the validation process.
        """
        self.logger.info("Validating Train Data...")
        train_errors, train_lines = self._validate_file(train_dataset_path)
        self._log_validation_results(train_dataset_path, train_errors)

        total_lines = train_lines
        if validation_dataset_path:
            self.logger.info("\nValidating Validation Data...")
            validation_errors, validation_lines = self._validate_file(
                validation_dataset_path, is_train=False
            )
            self._log_validation_results(validation_dataset_path, validation_errors)
            total_lines += validation_lines

        if total_lines > MAX_TOTAL_LINES:
            self.logger.error(
                "\nError: Total number of lines (%d) exceeds the maximum allowed (%d).",
                total_lines,
                MAX_TOTAL_LINES,
            )
        elif not train_errors and (
            not validation_dataset_path or not validation_errors
        ):
            self.logger.info("\nAll data passed validation!")
        else:
            self.logger.warning(
                "\nPlease correct the errors and run the validation again."
            )
