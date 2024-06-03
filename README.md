# BH-CodingAssignment
Coding Assignment for BH RA Application - NLP for tweet sentiment classification. We build a simple pipeline to process tweets and create binary classification models. The repo is structured as follows

# BH-CodingAssignment
Coding Assignment for BH RA Application - NLP for tweet sentiment classification. We build a simple pipeline to process tweets and create binary classification models. 

> A `requirements.txt` is available to set up a clean environment. After that, running `main.sh` should recreate the results.

The repo is structured as follows:

- `01_Data/`: This directory contains the dataset files used for training and testing the models.
    - `Raw`: Contains the raw datasets as downloaded from Kaggle
    - `Processed`: Contains processed training, validation and testing datasets
- `02_Code/`: This directory contains the source code for the tweet processing pipeline and classification models.
    - `data_prep.py`: Contains the pipeline to clean the dataset, generate features and create training, validation and testing sets
    - `modeling.py`: Contains pipeline to fit and evaluate models
    - `main.sh`: Orchestrator that runs the rest of the scripts
    - `generate_report_figures.py`: Small script to generate graph based on results
- `03_Results/`: This folder stores the final report, as well as the figures and tables that feed it
    - `Report.md`: Final report on project
    - Other figures and csv
- `04_Notebooks`: Empty, used it to store scratchpad notebooks


