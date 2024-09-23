# Bedrock Model Fine-tuner

A helper library for fine-tuning Amazon Bedrock models. This toolkit assists in generating Q&A datasets from documents and streamlines the LLM fine-tuning process.

## Features

- Wrapper classes for easy fine-tuning and deployment of Bedrock models using boto3 (currently limited to Claude 3 Haiku model)
- Q&A dataset generation from documents for Claude 3 Haiku model fine-tuning
- Dataset validation to ensure compliance with Claude 3 Haiku model fine-tuning format and constraints

## Usage

### Step 1: Generate Q&A Dataset from Documents (Optional)

```python
import boto3
from core import ChatModelId, QaDatasetGenerator, get_llm

boto_session = boto3.Session(region_name="us-west-2")

llm = get_llm(
    ChatModelId.CLAUDE_V3_5_SONNET,
    region_name="us-west-2",
)
qa_dataset_generator = QaDatasetGenerator.from_jsonl(llm, "../assets/docs.jsonl")

train_dataset = qa_dataset_generator.generate(dataset_type="train")
_ = qa_dataset_generator.save_and_upload(
    train_dataset,
    "../assets/train_dataset.jsonl",
    boto_session=boto_session,
    bucket_name="<YOUR-S3-BUCKET-NAME>",
)
```

### Step 2: Validate Q&A Dataset

```python
from core import QaDatasetValidator

qa_dataset_validator = QaDatasetValidator()
qa_dataset_validator.validate_data("../assets/train_dataset.jsonl")
```

### Step 3: Fine-tune and Deploy Model

```python
from core import BedrockModelFinetuner

bedrock_model_finetuner = BedrockModelFinetuner(aws_region_name="us-west-2")

_ = bedrock_model_finetuner.finetune("s3://<YOUR-S3-BUCKET-NAME>/datasets/train_dataset.jsonl")
_ = bedrock_model_finetuner.deploy()
```

### Step 4: Delete Provisioned Model (Optional)

```python
_ = bedrock_model_finetuner.delete()
```

For detailed usage examples, please refer to the notebook files in the `samples` directory.

## Additional Resources

- [Amazon Bedrock Custom Models Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html)
- [Claude Haiku Code Samples](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/bedrock-fine-tuning/claude-haiku)
- [Fine-tune Anthropic's Claude 3 Haiku in Amazon Bedrock to Boost Model Accuracy and Quality](https://aws.amazon.com/ko/blogs/machine-learning/fine-tune-anthropics-claude-3-haiku-in-amazon-bedrock-to-boost-model-accuracy-and-quality/)

## License

[MIT License](LICENSE)
