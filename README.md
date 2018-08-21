# Spark_Image_Classification
Example Image Classification Program Using Spark


## Requires Intel Analytics Zoo 
[Link to Intel Analytics Zoo GitHub Project](https://github.com/intel-analytics/analytics-zoo)

[Link to Install Instructions for Intel Analytics Zoo](https://analytics-zoo.github.io/master/#PythonUserGuide/install/)

## Requires Pyspark
[Link to Apache Spark Website](https://spark.apache.org/)

[Link to Pyspark Documentation](https://spark.apache.org/docs/2.3.1/api/python/index.html)

## Requires Keras
[Link to Keras GitHub](https://github.com/keras-team/keras)

[Link to Keras Documentation](https://keras.io/)

## Reads Files from
./caltech-256-image-dataset/256_ObjectCategories/\*/\*

### Dataset Download
[Link to Caltech256 Website for Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)


## Running the code
Create the docker container using

$ docker run -it -p 8888:8888 -e ACCEPT_EULA=yes microsoft/mmlspark

Add python and model files to docker container

$ docker cp [image-dataset-folder] [container id]:\notebooks\

$ docker cp [local-ipynb-file] [containerid]:\notebooks\

The .ipynb notebook can be started in docker container and should run without issue.

To run the .py file use docker exec: 

$ docker exec [container id] spark-submit --packages Azure:mmlspark:0.13 /notebooks/WorksInDocker-ExistingModel.py


<a href="http://www.vision.caltech.edu/Image_Datasets/Caltech256/" target="_blank">Link to Caltech256 Website for Dataset</a>
