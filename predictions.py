import pandas as pd
import numpy as  np

def classified_correct(model, i):
    return (model["sentiment"][i] == 0 and int(model["id"][i].split("_")[1]) <= 5 ) or\
           (model["sentiment"][i] == 1 and int(model["id"][i].split("_")[1]) > 5 )

model = pd.read_csv("Bag_of_Words_model.csv")
correct = np.array([classified_correct(model,i) for i in range(model.shape[0])])
print(correct.sum() / model.shape[0])