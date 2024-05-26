# keys-study

Code for the paper "Testing a Distributional Account of Grammatical Gender Effects on Semantic Gender Perception" (Flint, Ivanova 2024)
The [manuscript](https://drive.google.com/file/d/15lXIXJTCap1NosJ2HYCpYWwwRSHvEo68/view?usp=sharing) will appear in CogSci 2024.

## Overview
- This study investigated the effect of grammatical gender on gender semantics intralinguistically in distributional semantic spaces as well as behaviorally.
- This repository contains the scripts, analyses, and materials used in the study as well as the data produced.

## Usage
### Requirements
- Use Python 3.8 or later.
- Use `pip install -r requirements.txt` to install code dependencies.
### Materials
- We used FastText and BERT models for embeddings analyses, and human participants recruited on Prolific for behavioral studies. We investigated English (control), Spanish, and German.
  - FastText models for English, Spanish, and German should be downloaded from the FastText website and should be placed in the `models` directory. (E.g. `models/cc.de.300.bin`)
  - The multilingual BERT model should be loaded through Python. (See `scripts/extensions_playground.ipynb`, `contextual word embeddings` section.)
 ### Execution
- To run the core procedures of the project, run the command `python scripts/main.py` from the root directory.
- It is recommended to use the default parameters, but different parameters can be set to run different procedures and/or change how those procedures are run. Reference the docstring for the `run()` function in `scripts.py` for documentation on these parameters, or, run the command `python main.py --help`. 

## Project Structure
- `analyses` contains analyses in R for embeddings and behavioral experiments and their outputs.
- `data` contains data collected from embeddings and behavioral experiments.
    - `data/embeddings` contains the output for experiments run on FastText embeddings. 
    - `data/contextual-embeddings` contains the output for experiments run on BERT embeddings.
    - `data/adjective-ratings` contains the data collected from asking participants to rate words for semantic gender association.
    - `data/matchings` contains the data collected from asking participants to between two adjectives (one rated masculine, one rated feminine) given a noun (of masculine or feminine grammatical gender).
- `materials` contains the materials collected for the running of experiments.
    - `materials/nouns.csv` for the nouns,
    - `materials/adjectives` for the adjectives, where
        - `materials/adjectives/stimulus_files` were those adjectives selected to be rated by human participants, and
        - the full adjective lists can be found in Parquet and CSV form, separated into files for masculine and feminine gender associations* for each language. (E.g. `materials/adjectives/es_masculine_adjectives.csv`)
    - `materials/matchings` contains those nouns and adjectives selected to be used in the binary choice behavioral experiment, where
        - `materials/matchings/reference` contains these materials as lists, and
        - `materials/matchings/stimulus` contains these materials organized into questions for upload to Qualtrics for survey generation.
- `models` is the directory where the FastText `.bin` models should be placed.
- `scripts` contains the scripts used to run embeddings experiments.
    - `scripts/scripts.py` contains all of the functions and static variables used to run the FastText embeddings experiments.
    - `scripts/extensions_playground.ipynb` contains all of the functions and static variables used to run the FastText embeddings experiments. It also has some preliminary code for LLM completion experiments, although this was not included in the CogSci 2024 submission.

*determined by cosine similarity to their language's words for "man" and "woman" respectively in FastText models

## Future Work
This repository will only ever contain the materials for the CogSci 2024 submission and will not be updated with extensions. Extensions on this work will be located in another repository.

## Contact
- George Flint
    - georgeflint@berkeley.edu
    - [georgeflint.com](https://www.georgeflint.com/)
- Anna Ivanova
    - a.ivanova@gatech.edu
    - [anna-ivanova.net](https://anna-ivanova.net/)
