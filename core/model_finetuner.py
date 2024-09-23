"""
This module provides a BedrockModelFinetuner class for fine-tuning and managing Amazon Bedrock models.

The BedrockModelFinetuner class offers functionality to:
- Initialize and configure the fine-tuning environment
- Create and manage IAM roles for Bedrock access
- Prepare and submit fine-tuning jobs
- Monitor job progress and handle completion
- Deploy fine-tuned models
- Manage provisioned model throughput
- Delete provisioned models

Key components:
- AWS Bedrock client for interacting with the Bedrock API
- IAM role management for necessary permissions
- Hyperparameter configuration for fine-tuning jobs
- Logging and execution time measurement utilities

Usage:
    finetuner = BedrockModelFinetuner()
    finetuner.finetune(train_dataset_s3_uri, validation_dataset_s3_uri)
    finetuner.deploy()
    finetuner.delete()

Dependencies:
- boto3
- botocore
"""

import json
import time
from datetime import datetime
from pprint import pformat
from typing import Dict, Tuple, Optional

import boto3
from botocore.exceptions import ClientError

from .logger import Loggable
from .utils import get_s3_uri, measure_execution_time

BASE_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0:200k"
CUSTOMIZATION_TYPE = "FINE_TUNING"
DEFAULT_IAM_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
BEDROCK_SERVICE = "bedrock.amazonaws.com"
DEFAULT_REGION = "us-west-2"
DEFAULT_ROLE_NAME = "model-finetune-role"
DEFAULT_OUTPUT_PREFIX = "outputs"


class BedrockModelFinetuner(Loggable):
    """
    A class for fine-tuning and managing Amazon Bedrock models.

    This class provides methods to initialize the fine-tuning environment,
    submit and monitor fine-tuning jobs, deploy models, and manage
    provisioned model throughput.

    Attributes:
        region_name (str): AWS region name
        role_name (str): IAM role name for Bedrock access
        outputs_prefix (str): S3 prefix for output data
        boto_session (boto3.Session): Boto3 session for AWS API calls
        bedrock_client (boto3.client): Bedrock client for API interactions
        account_id (str): AWS account ID
        role_arn (str): ARN of the IAM role for Bedrock access
        job_name (str): Name of the current fine-tuning job
        custom_model_name (str): Name of the custom model being fine-tuned
        hyperparameters (Dict): Hyperparameters for fine-tuning
        custom_model_arn (str): ARN of the fine-tuned custom model
        provisioned_model_arn (str): ARN of the provisioned model
    """

    def __init__(
        self,
        aws_region_name: str = DEFAULT_REGION,
        iam_role_name: str = DEFAULT_ROLE_NAME,
        outputs_s3_prefix: str = DEFAULT_OUTPUT_PREFIX,
        base_model_id: str = BASE_MODEL_ID,
        boto_kwargs: Optional[Dict] = None,
        hyperparameters: Optional[Dict] = None,
    ):
        """
        Initialize the BedrockModelFinetuner.

        Args:
            aws_region_name (str): AWS region name (default: DEFAULT_REGION)
            iam_role_name (str): IAM role name for Bedrock access (default: DEFAULT_ROLE_NAME)
            outputs_s3_prefix (str): S3 prefix for output data (default: DEFAULT_OUTPUT_PREFIX)
            base_model_id (str): Base model ID for fine-tuning (default: BASE_MODEL_ID)
            boto_kwargs (Optional[Dict]): Additional kwargs for boto3.Session
            hyperparameters (Optional[Dict]): Initial hyperparameters for fine-tuning
        """
        super().__init__()
        self.region_name = aws_region_name
        self.role_name = iam_role_name
        self.outputs_prefix = outputs_s3_prefix
        self.boto_session = boto3.Session(**(boto_kwargs or {}))
        self.bedrock_client = self.boto_session.client(
            "bedrock", region_name=self.region_name
        )
        self.account_id = self._get_account_id()
        self.role_arn = self._get_or_create_iam_role()
        self.job_name = None
        self.custom_model_name = None
        self.base_model_id = base_model_id
        self.provider = base_model_id.split(".")[0]
        self.hyperparameters = hyperparameters or {}
        self.custom_model_arn = None
        self.provisioned_model_arn = None

    def _create_iam_role(self) -> str:
        """
        Create an IAM role for Bedrock access.

        Returns:
            str: ARN of the created IAM role

        Raises:
            ClientError: If there's an error creating the IAM role
        """
        iam_client = self.boto_session.client("iam")
        trust_relationship = self._get_trust_relationship()
        try:
            role = iam_client.create_role(
                RoleName=self.role_name,
                AssumeRolePolicyDocument=json.dumps(trust_relationship),
                Description="IAM role for Amazon Bedrock model customization",
            )
            iam_client.attach_role_policy(
                RoleName=self.role_name, PolicyArn=DEFAULT_IAM_POLICY_ARN
            )
            self.logger.info("Created IAM role: %s", role["Role"]["Arn"])
            return role["Role"]["Arn"]
        except ClientError as e:
            self.logger.error("Error creating IAM role: %s", e)
            raise

    def _get_account_id(self) -> str:
        """
        Get the AWS account ID.

        Returns:
            str: AWS account ID
        """
        return self.boto_session.client("sts").get_caller_identity()["Account"]

    def _get_hyperparameters(self) -> Dict:
        """
        Get the hyperparameters for fine-tuning.

        Returns:
            Dict: Hyperparameters with default values if not specified
        """
        return {
            "epochCount": str(self.hyperparameters.get("epoch_count", 3)),
            "batchSize": str(self.hyperparameters.get("batch_size", 8)),
            "learningRateMultiplier": str(
                self.hyperparameters.get("learning_rate_multiplier", 2)
            ),
            "earlyStoppingThreshold": str(
                self.hyperparameters.get("early_stopping_threshold", 0.01)
            ),
            "earlyStoppingPatience": str(
                self.hyperparameters.get("early_stopping_patience", 3)
            ),
        }

    def _get_or_create_iam_role(self) -> str:
        """
        Get an existing IAM role or create a new one if it doesn't exist.

        Returns:
            str: ARN of the IAM role
        """
        iam_client = self.boto_session.client("iam")
        try:
            return iam_client.get_role(RoleName=self.role_name)["Role"]["Arn"]
        except iam_client.exceptions.NoSuchEntityException:
            return self._create_iam_role()

    def _get_trust_relationship(self) -> Dict:
        """
        Get the trust relationship policy for the IAM role.

        Returns:
            Dict: Trust relationship policy document
        """
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": BEDROCK_SERVICE},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": self.account_id},
                        "ArnEquals": {
                            "aws:SourceArn": f"arn:aws:bedrock:{self.region_name}:{self.account_id}:model-customization-job/*"
                        },
                    },
                }
            ],
        }

    def _prepare_job_config(
        self,
        train_dataset_uri: str,
        validation_dataset_uri: Optional[str],
        job_name: Optional[str],
        custom_model_name: Optional[str],
        base_model_id: str,
    ) -> Dict:
        """
        Prepare the configuration for a fine-tuning job.

        Args:
            train_dataset_uri (str): S3 URI of the training dataset
            validation_dataset_uri (Optional[str]): S3 URI of the validation dataset
            job_name (Optional[str]): Name for the fine-tuning job
            custom_model_name (Optional[str]): Name for the custom model

        Returns:
            Dict: Job configuration for the fine-tuning job
        """
        bucket_name = train_dataset_uri.split("//")[1].split("/")[0]
        output_data_uri = get_s3_uri(
            bucket_name, f"{self.outputs_prefix}/{custom_model_name}"
        )
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.job_name = job_name or f"model-finetune-job-{timestamp}"
        self.custom_model_name = custom_model_name or f"finetuned-model-{timestamp}"
        job_config = {
            "customizationType": CUSTOMIZATION_TYPE,
            "jobName": self.job_name,
            "customModelName": self.custom_model_name,
            "roleArn": self.role_arn,
            "baseModelIdentifier": base_model_id,
            "hyperParameters": self._get_hyperparameters(),
            "trainingDataConfig": {"s3Uri": train_dataset_uri},
            "outputDataConfig": {"s3Uri": output_data_uri},
        }
        if validation_dataset_uri:
            job_config["validationDataConfig"] = {
                "validators": [{"s3Uri": validation_dataset_uri}]
            }
        return job_config

    def set_hyperparameters(self, hyperparameters: Dict) -> None:
        """
        Set or update hyperparameters for fine-tuning.

        Args:
            hyperparameters (Dict): Hyperparameters to set or update
        """
        self.hyperparameters.update(hyperparameters)

    @measure_execution_time
    def finetune(
        self,
        train_dataset_s3_uri: str,
        validation_dataset_s3_uri: Optional[str] = None,
        job_name: Optional[str] = None,
        custom_model_name: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Start a fine-tuning job and wait for its completion.

        Args:
            train_dataset_s3_uri (str): S3 URI of the training dataset
            validation_dataset_s3_uri (Optional[str]): S3 URI of the validation dataset
            job_name (Optional[str]): Name for the fine-tuning job
            custom_model_name (Optional[str]): Name for the custom model

        Returns:
            Tuple[str, Optional[str]]: Tuple containing the custom model name and ARN (if successful)
        """
        job_config = self._prepare_job_config(
            train_dataset_s3_uri,
            validation_dataset_s3_uri,
            job_name,
            custom_model_name,
            self.base_model_id,
        )
        response = self.bedrock_client.create_model_customization_job(**job_config)
        self.logger.info("Fine-tuning job created: %s", response["jobArn"])
        job_identifier = response["jobArn"]
        success = self._wait_for_job_completion(job_identifier)
        return (
            (self.custom_model_name, self.custom_model_arn)
            if success
            else (self.custom_model_name, None)
        )

    def _wait_for_job_completion(self, job_identifier: str) -> bool:
        """
        Wait for the completion of a fine-tuning job and handle its outcome.

        Args:
            job_identifier (str): Identifier of the fine-tuning job

        Returns:
            bool: True if the job completed successfully, False otherwise
        """
        while True:
            try:
                fine_tune_job = self.bedrock_client.get_model_customization_job(
                    jobIdentifier=job_identifier
                )
                status = fine_tune_job["status"]
                self.logger.info("Fine-tuning job status: %s", status)
                if status == "Completed":
                    self.logger.info("Fine-tuning job completed successfully")
                    self.logger.debug(
                        "Fine-tuning job details: %s", pformat(fine_tune_job)
                    )
                    self.custom_model_arn = fine_tune_job.get("outputModelArn")
                    if not self.custom_model_arn:
                        self.logger.error("Model ARN not found in the job details")
                        return False
                    return True
                if status in ["Failed", "Stopped"]:
                    self.logger.error("Fine-tuning job failed or was stopped")
                    self.logger.debug(
                        "Fine-tuning job details: %s", pformat(fine_tune_job)
                    )
                    return False
                if status not in ["InProgress", "Completed", "Failed", "Stopped"]:
                    self.logger.warning("Unexpected job status: %s", status)
                    return False
                time.sleep(60)
            except ClientError as e:
                self.logger.error("Error while checking job status: %s", str(e))
                return False

    @measure_execution_time
    def deploy(
        self,
        model_id: Optional[str] = None,
        provisioned_model_name: Optional[str] = None,
        model_units: int = 1,
    ) -> str:
        """
        Deploy a fine-tuned model by creating provisioned throughput.

        Args:
            model_id (Optional[str]): ID of the model to deploy (default: self.custom_model_arn)
            provisioned_model_name (Optional[str]): Name for the provisioned model (default: self.custom_model_name)
            model_units (int): Number of model units to provision (default: 1)

        Returns:
            str: ARN of the provisioned model

        Raises:
            ValueError: If model_id and provisioned_model_name are not provided and not available
        """
        model_id = model_id or self.custom_model_arn
        provisioned_model_name = provisioned_model_name or self.custom_model_name

        if not model_id or not provisioned_model_name:
            raise ValueError(
                "Model is not fine-tuned yet or model_id and provisioned_model_name are not provided"
            )

        response = self.bedrock_client.create_provisioned_model_throughput(
            modelId=model_id,
            provisionedModelName=provisioned_model_name,
            modelUnits=model_units,
        )

        provisioned_model_id = response["provisionedModelArn"]
        self.logger.info("Provisioned Throughput created: %s", provisioned_model_id)

        self._wait_for_provisioning(provisioned_model_id)
        self.provisioned_model_arn = provisioned_model_id

        return self.provisioned_model_arn

    def _wait_for_provisioning(self, provisioned_model_id: str) -> None:
        """
        Wait for the provisioning of a model to complete.

        Args:
            provisioned_model_id (str): ID of the provisioned model
        """
        while True:
            status = self.bedrock_client.get_provisioned_model_throughput(
                provisionedModelId=provisioned_model_id
            )["status"]
            self.logger.info("Provisioning status: %s", status)
            if status != "Creating":
                break
            time.sleep(60)

    def delete(self, provisioned_model_id: Optional[str] = None) -> None:
        """
        Delete a provisioned model.

        Args:
            provisioned_model_id (Optional[str]): ID of the provisioned model to delete (default: self.provisioned_model_arn)

        Raises:
            ValueError: If no provisioned model ID is provided or available
            ClientError: If there's an error deleting the provisioned model
        """
        if not provisioned_model_id and not self.provisioned_model_arn:
            raise ValueError("No provisioned model ID provided or available")
        provisioned_model_id = provisioned_model_id or self.provisioned_model_arn
        try:
            self.bedrock_client.delete_provisioned_model_throughput(
                provisionedModelId=provisioned_model_id
            )
            self.logger.info(
                "Provisioned model deleted successfully: %s", provisioned_model_id
            )
            self.provisioned_model_arn = None
        except ClientError as e:
            self.logger.error("Failed to delete provisioned model: %s", str(e))
            raise
