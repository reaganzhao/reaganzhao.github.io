---
layout: post
title:  Build Your Own Sagemaker Project
image:
  feature: gate_crop.png
tags:   programming
date:   2022-01-03 14:25
---

Sagemaker has its own templates for machine learning model training, deployment, and monitoring. One of the very important features is the sagemaker project, which has the default deployment and monitoring environment and is able to handle the git repo between users. While there seems to be limited resources and documentation regarding the how to set up your own project apart from the abalone tutorial, I would like to walk through how to 1. build a inference pipeline model, 2. create your own image for a [sagemaker project](https://reaganzhao.github.io/use-updated-scikit-learn-in-sagemaer-project/)


## Inference Pipeline

The inference pipeline saves the combined sklearn preprocessor and machine learning model, and trigger them after you deploy the model. There is a very good tutorial about inference pipeline in [sagemaker notebook](https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/Inference%20Pipeline%20with%20Scikit-learn%20and%20Linear%20Learner.ipynb). It walks through how the inference pipeline should be saved and applied in a Jupiter notebook. While it doesn't talk too much about how to transform it in a sagemaker project using the defined steps. You may have seen the [post](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html), and under "RegisterModel" there is a pipeline model defined simply as the above, but there is way too few information about how to build it. Here is my way to build it:

**1.Create a sklearn estimator**

```python
from sagemaker.sklearn.estimator import SKLearn

sklearn_preprocessor = SKLearn(
        entry_point=os.path.join(BASE_DIR, "sklearn_preprocess.py"),
        role=role,
        instance_type="ml.m5.large",
        sagemaker_session=sagemaker_session,
        base_job_name='FitSkLearnPreprocessorJob',
        image_uri=sklearn_image_uri
        )
```
The section is the same as the notebook tutorial. Remember to put the input_fn, out_fn, predict_fn, and model_fn as the [`notebook`](https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_inference_pipeline/Inference%20Pipeline%20with%20Scikit-learn%20and%20Linear%20Learner.ipynb) in the entry point script.

**2.Fit the sklearn estimator**

```python
from sagemaker.workflow.steps import TrainingStep
step_train_sklearn = TrainingStep(
        name="FitSkLearnPreprocessor",
        estimator=sklearn_preprocessor,
        inputs={
            "train": TrainingInput(
                s3_data=input_data.default_value,
                content_type="text/csv",
            )}
        )
```
This section is the ".fit({"train": train_input})" in the notebook. While sagemaker project doesn't support a .fit(), we will use a training step instead.

**3.(Optional) Convert the fit result**

This step is optional because if you hardcode your output data file in the entry_point in step 1, then you can use the hardcoded path directly in the next step. If you hate to use any hardcoded path (like me), you can use an additional step to avoid that.

So in your sklearn_preprocess.py, save your data using
```python
df_train.to_csv('/opt/ml/output/data/train.csv', index=False, header=False)
df_test.to_csv('/opt/ml/output/data/test.csv', index=False, header=False)
```
/opt/ml/output/data/ is the default output path and the data will then be saved into the s3 bucket. It will be a zipped file, so you will need an additional step like this to unzip it to the right format for model training

```
save_data_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=sagemaker_session,
        base_job_name='FormatPreprocessorOutput',
        role=role,
    )
step_save_data = ProcessingStep(
        name="FormatPreprocessorOutputJob",
        processor=save_data_processor,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "pipelines/leadscoresignup/format_preprocessor_output.py"),
        job_arguments=["--preprocessor-model", step_train_sklearn.properties.ModelArtifacts.S3ModelArtifacts],
        depends_on=['FitSkLearnPreprocessor']
    )
```
The key parameter here is the job_arguments, it helps you to find the location from the previous step without hardcoding.
```python
import os
import io
import boto3
import tarfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--preprocessor-model", type=str, required=True)
args = parser.parse_args()
preprocessor_model = args.preprocessor_model

output_dir = os.path.dirname(preprocessor_model)
output_bucket = output_dir.split('//')[1].split('/')[0]
output_file_key = output_dir.split('//')[1].split('/', 1)[1] + '/output.tar.gz'

s3 = boto3.client('s3')
s3_object = s3.get_object(Bucket=output_bucket, Key=output_file_key)
buffer = io.BytesIO(s3_object['Body'].read())

tarf = tarfile.open(fileobj=buffer)
tarf.extract(member='train.csv', path='/opt/ml/processing/train/')
tarf.extract(member='test.csv', path='/opt/ml/processing/test/')
```
There can be other ways to unzip and extract the files, this is my way to do that. Remember to save the result into /opt/ml/processing/ so that you can pass that later using the properties.



**4.Train your machine learning model**

You can then train the model using the way the same as the abalone tutorial. I use xgboost as the example here:

```python
step_train_xgboost = TrainingStep(
        name="TrainXGBoostModel",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_save_data.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_save_data.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        depends_on=['FormatPreprocessorOutputJob']
    )
```

**5.Final Step: Create the pipeline model for inference**

This time it will be similar to the [`documentation`](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html)

```python
model_sklearn_preprocessor = SKLearnModel(
        model_data=step_train_sklearn.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point=os.path.join(BASE_DIR, "pipelines/leadscoresignup/sklearn_preprocess.py"),
        sagemaker_session=sagemaker_session,
        role=role,
        name='SkLearnPreprocessorModel',
        image_uri=sklearn_image_uri,
    )


model_xgboost = Model(
        image_uri=xgboost_image_uri,
        model_data=step_train_xgboost.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role,
        name='XGBoostModel'
    )

model_inference_pipeline = PipelineModel(
        name="LeadScoreSignupInferencePipeline", 
        role=role, 
        sagemaker_session=sagemaker_session,
        models=[
            model_sklearn_preprocessor, 
            model_xgboost])

step_register = RegisterModel(
        name="RegisterLeadScoreSignupModel",
        model=model_inference_pipeline,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        depends_on=['TrainXGBoostModel']
    )
```
The model will be created under model registry in the sagemaker resources.
