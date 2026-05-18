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