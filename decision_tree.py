import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('train.csv')

# Select relevant columns
selected_columns = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 'Vehicle_condition', 'Type_of_order', 'Type_of_vehicle']

df_selected = df[selected_columns]

label_encoder = LabelEncoder()
df_selected['Type_of_order'] = label_encoder.fit_transform(df_selected['Type_of_order'])
df_selected['Type_of_vehicle'] = label_encoder.fit_transform(df_selected['Type_of_vehicle'])

# Separate features and target variables
X = df_selected.drop(['Restaurant_latitude', 'Restaurant_longitude'], axis=1)
y_latitude = df_selected['Restaurant_latitude']
y_longitude = df_selected['Restaurant_longitude']

# Split the data into training and testing sets
X_train, X_test, y_latitude_train, y_latitude_test, y_longitude_train, y_longitude_test = train_test_split(
    X, y_latitude, y_longitude, test_size=0.2, random_state=42
)

# Scale numerical features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize DecisionTreeRegressor models for latitude and longitude
dt_latitude = DecisionTreeRegressor(random_state=42)
dt_longitude = DecisionTreeRegressor(random_state=42)

# Fit the models
dt_latitude.fit(X_train_scaled, y_latitude_train)
dt_longitude.fit(X_train_scaled, y_longitude_train)

# Create a new data point for recommendation
new_data_point = pd.DataFrame({
    'Delivery_location_latitude': [13.043041],
    'Delivery_location_longitude': [77.813237],
    'Vehicle_condition': [2],
    'Type_of_order': [3],
    'Type_of_vehicle': [1]
})

# Predict values for latitude and longitude
predicted_latitude = dt_latitude.predict(scaler.transform(new_data_point))
predicted_longitude = dt_longitude.predict(scaler.transform(new_data_point))

# Calculate R2 scores for latitude and longitude on the test set
predicted_latitude_test = dt_latitude.predict(X_test_scaled)
predicted_longitude_test = dt_longitude.predict(X_test_scaled)

r2_latitude = r2_score(y_latitude_test, predicted_latitude_test)
print(f'R2 Score for Latitude (Decision Tree): {r2_latitude}')

r2_longitude = r2_score(y_longitude_test, predicted_longitude_test)
print(f'R2 Score for Longitude (Decision Tree): {r2_longitude}')

# Calculate Mean Squared Errors for latitude and longitude on the test set
mse_latitude = mean_squared_error(y_latitude_test, predicted_latitude_test)
print(f'Mean Squared Error for Latitude (Decision Tree): {mse_latitude}')

mse_longitude = mean_squared_error(y_longitude_test, predicted_longitude_test)
print(f'Mean Squared Error for Longitude (Decision Tree): {mse_longitude}')

# Visualize the data
plt.figure(figsize=(10, 6))

plt.scatter(df_selected['Restaurant_latitude'], df_selected['Restaurant_longitude'], c='blue', marker='o', label='Existing Restaurants')
plt.scatter(y_latitude_test, y_longitude_test, c='red', marker='x', label='Actual Test Data')
plt.scatter(predicted_latitude, predicted_longitude, c='green', marker='s', label='Recommended Restaurant')

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Latitude and Longitude Prediction with Decision Tree')
plt.legend()
plt.show()

# Visualize feature importance for latitude
plt.figure(figsize=(10, 6))
plt.barh(X.columns, dt_latitude.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Latitude Prediction')
plt.show()

# Visualize feature importance for longitude
plt.figure(figsize=(10, 6))
plt.barh(X.columns, dt_longitude.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Longitude Prediction')
plt.show()