# Census Income Classification
An analysis of census data and prediction of income class (<=50K or >50K).

## Data Source
This data was obtained from [UCI Machine Learning Repositor](https://archive.ics.uci.edu/ml/datasets.php). Data source: [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/Adult). This dataset contains census information related to occupation, education and personal information like race, gender and family status.

## Data Preparation
Data was provided in comma-seperated format in 2 different files. One named "adult.data" having 32561 rows which we will use as training data and other named "adult.test" having 16281 which we will use as testing data. Subset of training data only will be used for cross-validation. Data was observed to be fairly clean having only a small fraction of rows containing missing data for a few columns. Missing value was represented by "?" character. Since the size of dataset is big enough given that total number of features is just 13, we decided to drop "native-country" column because only less 2% rows were missing value for this feature. Same reasoning was applied when droping rows not having values for workclass.
We also decided to drop 2 columns: "fnlwgt" and "education". Education feature is actually also present as label encoding with name "education-num". So their was no use of retaining it. "fnlwgt" column was decided to be dropped after analysing its description given in "adult.name" file.
After performing EDA, all categorical columns were one-hot encoded for better model training.

## Data Analysis.
Univariate analysis and Bivariate analysis with income column of multiple columns were performed and results observed are reported in the jupyter notebook. Important thing observed was there was imbalanced data across target feature of income category as well other features. For example more rows were observed for people having US as their native country or white race was most common in "race" feature.

## Data  Modelling
Basic logistic regression model was initially created using GLM and cut-off probability was adjusted to give optimal results having comparable recall and precision.
Then we applied Decision Tree, Random Forest and AdaBoost trying to improve results achieved from Logistic Regression.
In the end we decided to see if neural networks were able to get better results. After trying networks with different neurons and connections, we reported final results in the notebook.

## Final results
Out of all models, only Random Forest was unable to properly handle class-imbalance problem. We faced a bottle-neck in acheiving balanced accuracy of around 83% with all models.