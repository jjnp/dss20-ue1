# dss20-ue1
Data Stewardship Summer Term 2020 Exercise 1
Jacob Palecek
Matr.Nr. 01526624
# About this project
This GitHub project hosts the code for exercise 1 of the Data Stewardship course in the summer term 2020 of TU Vienna.
In this sample project data from mobile text messages (SMS) is analyzed as to whether it is a spam message or not.
# How to run
## Directly
You require a relatively recent installation of Python 3 to run this project. Start by installing the necessary dependencies. Assuming you have pip installed with Python navigate to the project folder and run `pip3 install -r requirements.txt` to install the necessary dependencies.

Once dependencies are install you can run the project by executing the file main.py using the Python 3 interpreter using this command inside the folder: `python3 main.py`. If you want the intermediate step to be part of the results as well (the fully transformed input data separated into training data, testing data, training labels, and testing labels) you simply have to specify an additional argument e.g. `python main.py 1`. While not the most elegant, it is by far the simplest solution as the program simply checks for the presence of an argument.

NOTE: It is **not** recommended to output the intermediary data as it dramatically increases the runtime, since the transformed data is much larger when represented in text form and writing it to disk can take up to a few minutes on a slow hard drive.

If you want intermediary steps 
## With Docker
You need to have docker installed on your system in order to run this using docker. You can build the image using the following command in the folder:
`docker build -t dss20ue1 .`. This may take a while as it needs to download all required dependencies.
You can then run the image using this command: `docker run --rm -v "<output-folder>:/app/results" dss20ue1`

Please note that you should replace `<output-folder>` with a valid path on your system where you want the output to be saved. 

Just like in the local run you can include intermediary data(fully transformed input data) by supplying an argument like so:
`docker run --rm -v "<output-folder>:/app/results" dss20ue1 1`, although just like locally it is not recommended for performance reasons.
# What is done in the project
The project tries to automate the decision of whether a text message is spam or not. To do this a 

# Results
Sample results can be seen in the `comparison_results` folder. When you run it your result should be exactly the same.
In the comparison_results folder intermediary results are also provided, namely the fully transformed training and testing input, as well as the according labels. This data is exactly what is used for training the subsequent model. For performance reasons this data is only written to disk if a flag is specified.
The `training_input.csv` file is only available as a zip because it would otherwise exceed githubs upload limit. The preprocessed files are:
 * training_input.csv (from training_input.zip)
 * training_labels.csv
 * test_input.csv
 * test_labels.csv
# Licenses
## Code
The code and images are provided under the MIT license. You can see the terms in the `LICENSE` file
## Data
The data does not have a specific license attached to it and has been obtained from [this](https://www.kaggle.com/uciml/sms-spam-collection-dataset "Spam SMS data on Kaggle") Kaggle repository, and is itself an agglomeration of other datasets.
