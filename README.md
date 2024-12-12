# nlp-final
### Extractive Model

To run this code, simply open the directory **Jacks Extractive Model** and run the `training.py` file. This will output two files: one JSON file to store training and validation results, and another file saving the PyTorch model created. File names can be edited at the bottom of the training loop.

### Pre-Trained Testing
To run the Pre-Trained Model testing, open the notebook named Pre Trained HF Model Testing.ipynb . This file has 4 Sections, you can run the first section to load the environment. Sections 2 and 3 are for initial EDA and testing so these can be skipped. The final section named Final Testing can be run to generate outputsfor the model performance. 

**Note:** If you are using the government reporting dataset, you must ensure the dataset class is updated due to a mismatch in keys (`article` vs. `report` and `abstract` vs. `summary`).

The testing file will run a loop over all the models in the model list provided at the top and will output a JSON file containing the ROUGE scores for each model and each dataset.

