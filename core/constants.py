from enum import Enum


class ChatModelId(str, Enum):
    """
    Enumeration of chat model IDs for various AI models.

    This enum provides string constants for different versions and types of chat models
    from various providers such as Anthropic, Cohere, Meta, and Mistral.
    """

    CLAUDE_V1: str = "anthropic.claude-instant-v1"
    CLAUDE_V2_1: str = "anthropic.claude-v2:1"
    CLAUDE_V3_HAIKU: str = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_SONNET: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_OPUS: str = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_V3_5_SONNET: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_V3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    COMMAND_R: str = "cohere.command-r-v1:0"
    COMMAND_R_PLUS: str = "cohere.command-r-plus-v1:0"

    META_LLAMA_3_8B_INST: str = "meta.llama3-8b-instruct-v1:0"
    META_LLAMA_3_70B_INST: str = "meta.llama3-70b-instruct-v1:0"
    META_LLAMA_3_1_8B_INST: str = "meta.llama3-1-8b-instruct-v1:0"
    META_LLAMA_3_1_70B_INST: str = "meta.llama3-1-70b-instruct-v1:0"
    META_LLAMA_3_1_405B_INST: str = "meta.llama3-1-405b-instruct-v1:0"

    MISTRAL_7B_INST: str = "mistral.mistral-7b-instruct-v0:2"
    MISTRAL_8X7B_INST: str = "mistral.mixtral-8x7b-instruct-v0:1"
    MISTRAL_LARGE: str = "mistral.mistral-large-2402-v1:0"
    MISTRAL_LARGE_2407: str = "mistral.mistral-large-2407-v1:0"
    MISTRAL_SMALL: str = "mistral.mistral-small-2402-v1:0"
