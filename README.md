# Classifying Reddit posts using FinBert

Link to publication: _____

This program is an MVP to test out the possibilities of using ML in analyzing Finnish social media conversations. It can analyze any Reddit post which has a valid link. The output of the program is an analysis of the post's title, contents and comments. The sentiment and NER tags are identified. One of the output files includes the combination of these two that highlights the most frequent NER tags with the sentiment of the context from which the NER tag was found. Using this information one can get an overall idea of the conversation that happened in the post.

## Demo

Link to a demo video: ?


## Manual

### Setup

To use this program you need to either train the models or download them from OneDrive. If training the models is done by yourself, the training data needs to be set to the correct folders. This is instructed in the [2. Training data](#2-training-data). If the models are not wanted to be trained yourself they can be downloaded. Downloading the models is instructed in [4. Training/ Loading models](#4-training-loading-models)

#### 1. Prerequisite

- The program is developed and tested by using Python 3.8.0. The same version is recommended if this program is used.
- Having at least 15GB of free memory on the drive this Program is used at
  * Training data ~8MB
  * Trained models ~8GB
  * Venv ~6.5GB
- The repository is copied to your own machine
- Internet connection

#### 2. Training data

The models are trained using [turku-one corpus](https://github.com/TurkuNLP/turku-one) for the NER tagger and [FinnSentiment 1.1 corpus](https://metashare.csc.fi/repository/browse/finnsentiment-11-source/aae3853ea5ff11ed8b7cfa163eb87b84db6dcd26d78145808f85231b123053cb/) for sentiment analysis.

1. Download the [turku-one corpus](https://github.com/TurkuNLP/turku-one) as a zip file
2. Create a folder named 'data_ner' at the root of the project.
3. Extract the zip file to './data_ner'
    1. There should be a path './data_ner/data' in which there are 'dev.tst', 'test.tsv' and 'train.tsv'
4. Download the [FinnSentiment 1.1 corpus](https://metashare.csc.fi/repository/browse/finnsentiment-11-source/aae3853ea5ff11ed8b7cfa163eb87b84db6dcd26d78145808f85231b123053cb/) as a zip file
5. Create a folder named 'data' at the root of the project.
6. Extract the zip file to './data'
    1. There should be a path './data/finsen-v1-1-src/' in which there is 'FinnSentiment-1.1.tsv'
   

#### 3. Setting Python virtual environment

The requirements for the virtual environment are in a file called './requirements.txt'. To create and install the requirements follow these steps:
1. Open the terminal and go to the root of the project
2. Install virtual Python environment by running

   ```console
   python -m venv .venv
   ```
3. Activate and step into the virtual environment by running
    ```console
    .venv\Scripts\activate
    ```
4. Upgrade pip inside the virtual environment
    ```console
   pip install --upgrade pip
   ```
5. Install the requirements in the virtual environment by running. If this fails, the requirements need to be installed manually. Might fail if other than Python 3.8 is used!
    ```console
    pip install -r requirements.txt
    ```


#### 4. Training/ Loading models

The program uses two trained neural networks to analyze the posts. These need to be either trained or downloaded before using the program. If the models are downloaded by the user, the training data needs to be downloaded to the correct folders first. This is shown in [2. Training data](#2-training-data). After the training data has been downloaded and the virtual environment is set, the program can be started. Go to section [Usage](#usage)

If the models are not wanted to be trained they can be downloaded from OneDrive. Follow these instructions to start downloading the models:
1. Go to OneDrive and download the models [Link to OneDrive folder](https://lut-my.sharepoint.com/:f:/g/personal/vili_raunola_student_lut_fi/EmDKNLQmoStKrBRtCJ5sdvIBC1yVo4ii-F9MYuVDL6x8IQ?e=3lvHnf)
2. Navigate to './code/models/' folder
3. Copy the downloaded folders 'model' and 'ner_model' to this folder
    1. In './code/models/' there should now be 'model/', 'ner_model/', 'classifier.py' and 'tagger.py'


#### 5. Reddit API key
 To use this program a Reddit API key is needed to fetch the posts for analysis. To get yourself a Reddit API key follow the instructions here: https://www.jcchouinard.com/reddit-api/

 Once you have obtained the API key, create a file called '.env' to the root of the project. An example of the contents is shown in './.envExample'-file. Paste your client_id, client_secret and user_agent to your '.env'-file.



### Usage

Once the setup is done the program can be started.

1. Start the Python virtual environment if it is not running from the root of the project:
   ```console
    .venv\Scripts\activate
    ```
2. Go to './code/'
3. Run the main file:
    ```console
    python main.py
    ```
4. Follow the instructions given by the program
5. Happy analyzing!
