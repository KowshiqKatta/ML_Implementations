import pandas as pd

df = pd.DataFrame({
    "Color": ["Red", "Green", "Blue", "Red"]
})

# One-hot encoding using pandas 

ohe_df = pd.get_dummies(df, columns=["Color"], drop_first = True)

print(ohe_df)

# One-hot encoding using scikit-learn

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

one_hot_encoded = encoder.fit_transform(df[["Color"]])

print(one_hot_encoded)

encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(["Color"]))

print(encoded_df)

# label encoding using scikit-learn

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

print(df)

df["Color_encoded"] = encoder.fit_transform(df["Color"])

print(df)

# ordinal encoding using scikit-learn

data = {
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['S', 'M', 'L', 'M', 'S'],
    'price': [10, 20, 30, 20, 10]
}

df = pd.DataFrame(data)

print(df)

from sklearn.preprocessing import OrdinalEncoder

size_categories = [['S', 'M', 'L']] #increasing order of size

ordinal_encoder = OrdinalEncoder(categories=size_categories)

size_values = df[["size"]].values # creating a numpy error as it is needed for the fit_transform method of the ordinal encoder

print(size_values)

df["size_encoded"] = ordinal_encoder.fit_transform(size_values)

print(df)
