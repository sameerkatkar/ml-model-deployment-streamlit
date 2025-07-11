import streamlit as st
import boto3
import os
import torch
from transformers import pipeline

bucket_name = "mlops-sameer"

local_path = "tinybert-sentiment-analysis"
s3_prefix = "s3_data"
s3 = boto3.client('s3', region_name = 'ap-south-1')
def download_dir(loacl_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket = bucket_name, Prefix = s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']

                local_file = os.path.join(loacl_path, os.path.relpath(s3_key, s3_prefix))

                s3.download_file(bucket_name, s3_key, local_file)

st.title("Machine Learning Model Deploymeny at the Server!!!")
button = st.button("Download Model")
if button:
    with st.spinner("Downloading.."):
        download_dir(local_path, s3_prefix)

input_text = st.text_area("Enter your review","Type here")
predict_button = st.button("Predict")
device = torch.device('cpu')
#device = torch.device('cuda')
classifier = pipeline('text-classification', model='tinybert-sentiment-analysis', device=device)

if predict_button:
    with st.spinner("Predicting.."):
        output = classifier(input_text)
        st.write(output[0]['label'])

#from transformers.pipelines import PIPELINE_REGISTRY

#print(PIPELINE_REGISTRY.get_supported_tasks())

        