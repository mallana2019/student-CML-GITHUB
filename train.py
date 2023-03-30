#visuallization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#modelling 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

#set random seed
seed = 42


##################################################
########## DATA PREP AND FEATURIZATION ###########
##################################################


df = pd.read_csv("exams.csv")

X = df.drop(columns=['math_score'],axis=1)
y= df['math_score']

num_features = X.select_dtypes(exclude="object").columns
cat_features= X.select_dtypes(include="object").columns

numeric_transformer  = StandardScaler()
oh_transformer = OneHotEncoder()


preprocessor = ColumnTransformer(
    [
    ("OneHotEncoder",oh_transformer,cat_features),
    ("StandardScaler",numeric_transformer,num_features)

    
    ]
)

X = preprocessor.fit_transform(X)

# Split into train and test sections
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = DecisionTreeRegressor(random_state=seed)
regr.fit(X_train, y_train)

# Report training set score
train_R2_score = r2_score(y_train,regr.predict(X_train)) 
# Report test set score
test_R2_score = r2_score(y_test,regr.predict(X_test)) 

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write(f"R2 score Training Data: {train_R2_score:.2f} \n" )
        outfile.write(f"R2 score Test Data: {test_R2_score:.2f}\n"  )


##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = regr.feature_importances_
labels = df.drop(columns=['math_score'],axis=1)
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()


