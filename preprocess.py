# I grabbed this from a kaggle entry
# https://www.kaggle.com/vincentlugat/titanic-neural-networks-keras-81-8
# wanted to illustrate the concept rather than doing the work to preprocess
# the titanic dataset into something easily useful

import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

data = pd.read_csv('train.csv')


# Title
data['Title'] = data['Name']

# Cleaning name and extracting Title
for name_string in data['Name']:
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    
# Replacing rare titles 
mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 
           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 
           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 
           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}
           
data.replace({'Title': mapping}, inplace=True)
titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']

# Replacing missing age by median/title 
for title in titles:
    age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]
    data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute
    
# New feature : Family_size
data['Family_Size'] = data['Parch'] + data['SibSp'] + 1
data.loc[:,'FsizeD'] = 'Alone'
data.loc[(data['Family_Size'] > 1),'FsizeD'] = 'Small'
data.loc[(data['Family_Size'] > 4),'FsizeD'] = 'Big'

# Replacing missing Fare by median/Pclass 
fa = data[data["Pclass"] == 3]
data['Fare'].fillna(fa['Fare'].median(), inplace = True)

#  New feature : Child
data.loc[:,'Child'] = 1
data.loc[(data['Age'] >= 18),'Child'] =0

# New feature : Family Survival (https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83)
data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
DEFAULT_SURVIVAL_VALUE = 0.5

data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
                               
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
                
for _, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
                    
####################################
# Encoding and pre-modeling
####################################                  

# dropping useless features
data = data.drop(columns = ['Age','Cabin','Embarked','Name','Last_Name',
                            'Parch', 'SibSp','Ticket', 'Family_Size'])

# Encoding features
target_col = ["Survived"]
id_dataset = ["Type"]
cat_cols   = data.nunique()[data.nunique() < 12].keys().tolist()
cat_cols   = [x for x in cat_cols ]
# numerical columns
num_cols   = [x for x in data.columns if x not in cat_cols + target_col + id_dataset]
# Binary columns with 2 values
bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()
# Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]
# Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    data[i] = le.fit_transform(data[i])
# Duplicating columns for multi value columns
data = pd.get_dummies(data = data,columns = multi_cols )
# Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(data[num_cols])
scaled = pd.DataFrame(scaled,columns = num_cols)
# dropping original values merging scaled values for numerical columns
df_data_og = data.copy()
data = data.drop(columns = num_cols,axis = 1)
data = data.merge(scaled,left_index = True,right_index = True,how = "left")
data = data.drop(columns = ['PassengerId'],axis = 1)

# Target = 1st column
cols = data.columns.tolist()
cols.insert(0, cols.pop(cols.index('Survived')))
data = data.reindex(columns= cols)

data.to_pickle('titanic.pkl')