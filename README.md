# Classifying Reddit posts using FinBert


Link to published thesis: _____

## Demo

Link to a demo video: ?


## Manual

### Setup

To use this program you need to set the training data to the correct folders. This is instructed in the [2. Training data](#2-training-data)

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

#### 4. Training/ Loading models

Link to OneDrive folder: ___________

### Usage
