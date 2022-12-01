# speaker-verification-test

Tested using Python 3.8.15

## Setup

Set up virtual env:

```
python -m venv venv
source venv/bin/activate
```

Install modules

```
pip install -r requirements.txt
```

## Run experiments

Make sure to have a dataset in the `./dataset` location,
which contains subdirectories representing speakers, which
contain `.wav` files.

The the experiments can be run using the following command:

```
python main.py
```

The output of the program will be a CSV file `all_scores.csv`,
containing all experiments scores and information.
