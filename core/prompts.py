"""
This module provides functions for generating prompts and examples for Q&A generation tasks.

The module includes the following main components:
- get_qa_generation_prompt: Generates a ChatPromptTemplate for Q&A generation
- get_few_shot_examples: Provides few-shot examples for Q&A generation

These functions are designed to create prompts and examples for AI models to generate
high-quality Q&A pairs based on given contexts and metadata.
"""

from typing import Dict, List
from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

OUTPUT_START_TOKEN = "<output>"
OUTPUT_END_TOKEN = "</output>"


def get_qa_generation_prompt(
    output_start_token: str = OUTPUT_START_TOKEN,
    output_end_token: str = OUTPUT_END_TOKEN,
) -> ChatPromptTemplate:
    """
    Generate a ChatPromptTemplate for Q&A generation tasks.

    This function creates a structured prompt template that includes:
    - A system message with guidelines for generating Q&A pairs
    - Few-shot examples to demonstrate the expected output format
    - A human message template for providing context and metadata
    - An AI message template for initiating the thought process

    Args:
        output_start_token (str): The token to mark the start of the output. Defaults to "<output>".
        output_end_token (str): The token to mark the end of the output. Defaults to "</output>".

    Returns:
        ChatPromptTemplate: A complete prompt template for Q&A generation tasks.
    """
    system_prompt_template = """
    You are an expert AI assistant tasked with generating high-quality Q&A pairs 
    for fine-tuning language models. Your goal is to create diverse and informative Q&A sets 
    based on the given context and metadata. Follow these guidelines:

    1. Analyze the context and metadata thoroughly.
    2. Generate 3-5 Q&A pairs, mixing single-turn and multi-turn formats.
    3. Cover key concepts, important details, and potential areas of interest.
    4. Provide clear, accurate, and comprehensive answers.
    5. Use a mix of question types (e.g., factual, analytical, hypothetical).
    6. Ensure that multi-turn Q&As build upon each other logically.
    7. In multi-turn Q&As, strictly alternate between user and assistant roles.
    8. Format your output as a valid JSON structure.

    Remember to generate all Q&A content in the same language as the provided context and metadata.
    Provide your final output within {output_start_token} and {output_end_token} tags.

    See the examples below for the expected format and thought process.
    """

    human_prompt_template = """
    Context: {context}
    Metadata: {metadata}

    Generate a set of Q&A pairs based on the given context and metadata. 
    Follow the guidelines provided in the system message.
    """

    ai_prompt_template = """Here's my thought process for generating Q&A pairs based on 
    the given context and metadata:"""

    few_shot_examples = get_few_shot_examples(
        output_start_token=output_start_token, output_end_token=output_end_token
    )

    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(
                "Context: {context}\nMetadata: {metadata}\n\n"
                "Generate a set of Q&A pairs based on the given context and metadata."
            ),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=few_shot_examples,
    )

    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                system_prompt_template.format(
                    output_start_token=output_start_token,
                    output_end_token=output_end_token,
                )
            ),
            few_shot_prompt,
            HumanMessagePromptTemplate.from_template(
                human_prompt_template,
                input_variables=["context", "metadata"],
            ),
            AIMessagePromptTemplate.from_template(ai_prompt_template),
        ]
    )


def get_few_shot_examples(
    output_start_token: str = OUTPUT_START_TOKEN,
    output_end_token: str = OUTPUT_END_TOKEN,
) -> List[Dict[str, str]]:
    """
    Generate few-shot examples for Q&A generation tasks.

    This function creates a list of example Q&A pairs based on a given context about the Industrial Revolution.
    The examples demonstrate the expected format and content for Q&A generation tasks.

    Args:
        output_start_token (str): The token to mark the start of the output. Defaults to "<output>".
        output_end_token (str): The token to mark the end of the output. Defaults to "</output>".

    Returns:
        List[Dict[str, str]]: A list containing a dictionary with the context, metadata, and formatted output
        for the few-shot example.
    """
    output_str = """
    Here's my thought process for generating Q&A pairs based on the given context and metadata:

    1. Analyze the context: The text discusses the Industrial Revolution, its timeframe, origin, spread, 
    and impact.
    2. Identify key concepts: Timeframe, origin, spread, mechanization, productivity, economic growth.
    3. Formulate diverse questions: When and where did it start? What were its characteristics? How did 
    it spread?
    4. Create a mix of single-turn and multi-turn Q&As to cover different aspects.
    5. Ensure logical flow in the multi-turn Q&A by connecting characteristics to their impacts.
    6. Generate clear and comprehensive answers in the same language as the context.
    7. Format the output as a valid JSON structure.

    Now, I'll generate the Q&A pairs:

    {output_start_token}
    {{
        "qa_pairs": [
            {{
                "type": "single-turn",
                "question": "산업 혁명은 언제, 어디에서 시작되었나요?",
                "answer": "산업 혁명은 18세기 후반부터 19세기 초반에 걸쳐 영국에서 시작되었습니다."
            }},
            {{
                "type": "multi-turn",
                "turns": [
                    {{   
                        "role": "user",
                        "content": "산업 혁명의 주요 특징은 무엇이었나요?"
                    }},
                    {{
                        "role": "assistant",
                        "content": "산업 혁명의 주요 특징은 대규모 산업화, 혁신, 그리고 제조 과정의 기계화였습니다."
                    }},
                    {{
                        "role": "user",
                        "content": "이러한 특징들이 사회에 어떤 영향을 미쳤나요?"
                    }},
                    {{
                        "role": "assistant",
                        "content": "이러한 특징들은 생산성과 경제 성장을 크게 증가시켰고, 이는 노동 패턴의 변화, 도시화, 사회 구조의 변화 등 사회를 크게 변화시켰습니다."
                    }}
                ]
            }},
            {{
                "type": "single-turn",
                "question": "산업 혁명은 어떻게 확산되었나요?",
                "answer": "산업 혁명은 영국에서 시작되어 빠르게 유럽과 북미 전역으로 퍼져나갔습니다."
            }}
        ]
    }}
    {output_end_token}
    """

    return [
        {
            "context": "산업 혁명은 18세기 후반부터 19세기 초반에 걸쳐 일어난 주요 산업화와 혁신의 시기였습니다. "
            "산업 혁명은 영국에서 시작되어 빠르게 유럽과 북미 전역으로 퍼져나갔습니다. "
            "이 시기에는 제조 과정의 기계화가 이루어져 생산성과 경제 성장이 크게 증가했습니다.",
            "metadata": "산업 혁명",
            "output": output_str.format(
                output_start_token=output_start_token, output_end_token=output_end_token
            ),
        }
    ]
