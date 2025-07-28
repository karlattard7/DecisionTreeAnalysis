from preprocessing import *
from decision_tree_id3 import buildDecisionTreeID3, postPruning, classificationReport
import warnings
warnings.filterwarnings('ignore')

""""
Car Dataset Attributes (Categorical)
-------------------------------------
buying       v-high, high, med, low
maint        v-high, high, med, low
doors        2, 3, 4, 5-more
persons      2, 4, more
lug_boot     small, med, big
safety       low, med, high


Wine Dataset Attributes (Continuous)
------------------------------------
Alcohol
Malic acid
Ash
Alcalinity of ash  
Magnesium
Total phenols
Flavanoids
Nonflavanoid phenols
Proanthocyanins
Color intensity
Hue
OD280/OD315 of diluted wines
Proline          
"""

# Car Evaluation Dataset
car_labels = ['buying', 'maintenance', 'doors', 'persons', 'lug_boot', 'safety', 'Class']
car_df = readParseDataset('car.data', car_labels)
# ---------- end of Car Dataset reading and parsing


# Wine Evaluation Dataset
wine_labels = ['Class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_ash', 'magnesium', 'total_phenols', 'flavanoids',
               'nonflavanoid_phenols', 'proanthocyanins', 'colour_intensity', 'hue', 'diluted_wines', 'proline']
wine_df = readParseDataset('wine.data', wine_labels)
# ---------- end of Wine Dataset reading


# Splitting Datasets to Train & Test Sets
df_train_car, df_test_car = splitTrainTest(car_df)
df_train_wine, df_test_wine = splitTrainTest(wine_df)

# Building Decision Trees for each respective Dataset
decision_tree_car = buildDecisionTreeID3(df_train_car)
decision_tree_wine = buildDecisionTreeID3(df_train_wine)

# Evaluating the Car test set -> Pre and Post Pruning
classificationReport(decision_tree_car, df_test_car, "Car Evaluation (Before Pruning)")
decision_tree_car_pruned = postPruning(decision_tree_car, df_train_car, df_test_car)
classificationReport(decision_tree_car_pruned, df_test_car, "Car Evaluation (After Pruning)")


# Evaluating the Wine test set -> Pre and Post Pruning
classificationReport(decision_tree_wine, df_test_wine, "Wine (Before Pruning)")
decision_tree_wine_pruned = postPruning(decision_tree_wine, df_train_wine, df_test_wine)
classificationReport(decision_tree_wine_pruned, df_test_wine, "Wine (After Pruning)")



