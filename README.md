# Serverless Data Engineering Pipeline
[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)




# Introduction
* This project is about predicting if the client will subscribe to a term deposit or not based on various features. 
* The main idea of this code is to show how to use Amazon SageMaker to train a model on this dataset and deploy it as an endpoint to make predictions on new data.

## Install all the libraries

```py
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker import image_uris
from sagemaker.session import s3_input, Session
```

## Downloading The Dataset And Storing in S3

```py
import pandas as pd
import urllib
try:
    urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ',e)

try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
```
![s3](images/Snip20230502_33.png)


## Train Test split

```py
import numpy as np
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
```

## Test Data Into Buckets

```py
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')

```


## Building Models

```py
container = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name,version ="latest")

# construct a SageMaker estimator that calls the xgboost-container
estimator = sagemaker.estimator.Estimator(image_uri=container,
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_type = 'ml.m5.2xlarge',
                                          instance_count= 1,
                                          volume_size = 5,
                                          output_path=output_path,
                                           use_spot_instances=True,
                                           max_run=300,
                                           max_wait= 600)

estimator.fit({'train': s3_input_train,'validation': s3_input_test})                                     
```

![model1](images/Snip20230502_28.png)
![model1](images/Snip20230502_29.png)

## Deploy Machine Learning Model 

```py
xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```

## Prediction of the Test Data
```py
from sagemaker.predictor import csv_serializer
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.content_type_csv = 'text/csv' # set the data type for an inference
xgb_predictor.serializer = csv_serializer # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array

cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
```

![prediction](images/Snip20230502_30.png)

## Deleting The Endpoints

```py
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
```

![endpoints](images/Snip20230502_31.png)
