# comparative reviews classification

This repo contains the code of my Master's thesis, which is about comparative comments classification.

The repo has several parts included:

1. Data folder contains the training dataset and some badcase files. Please use "jd_comp_final_v5.xlsx"

2. Result folder contains some attention visulaization html files and some model structure picture.

3. Old folder contains some original scripts, just for keeping for backup(will be removed in the next commit)

4. Python scripts start with "baidu" use Baidu API to complete word segment and embedding tasks.

5. Text Preprocessing scripts: utils.py, langconv.py, zh_wiki.py

6. Char/Word embedding script: embedding.py(You need to train the embeddings first for the first time)

7. Traditional models script: traditional_ml_models.py

8. Deep Learning models scripts:
    
    * config.py: model hyperparameters class
    * evaluator.py: model evaluation class
    * layers.py: DL text classification model used in thesis
    * main.py: the main program for training, more details please see the code comments(the command line version is coming soon)
    * metrics.py: model evaluation class during training
    * reader.py: data generator
    * tools.py: some useful fuctions and attention mechanism implementation
    * trainer.py: model training class

9. Average embedding model: average_embedding.py

10. Some model results and attention visualization: visualization.py
