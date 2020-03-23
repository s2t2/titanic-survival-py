# Titanic Survival Predictions (Python)

![](https://media.nationalgeographic.org/assets/photos/000/273/27302.jpg)

A binary classification model, inspired by the ["Titanic" Kaggle Challenge](https://www.kaggle.com/c/titanic).

Predicts whether or not a given passenger will survive, based on personal characteristics such as age, gender, and how much money their ticket cost.


## [Data Dictionary](DATA.md)

## Setup

Prerequisites:

  + Anaconda and Python 3.7
  + Graphviz (`brew install graphviz`)
  + Orca (`conda install -c plotly plotly-orca`)

Setup virtual environment:

```sh
conda create -n titanic-env python=3.7
conda activate titanic-env
```

Install package dependencies:

```sh
pip install -r requirements.txt
```

## Usage

Import the data, generate profile reports, and train and score the classifier:

```sh
python -m app.importer
python -m app.profiler
python -m app.classifier
```

## Results

Feature Importances:

![](/reports/feature_importances.png)


Decision Tree Logic:

![](/reports/decision_tree.png)
