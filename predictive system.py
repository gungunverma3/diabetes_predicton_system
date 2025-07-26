# import numpy as np
# import pickle
# from sklearn.preprocessing import StandardScaler

# # loading the save model

# loaded_model = pickle.load(open("trained_model.sav","rb"))
# input_data = (0,131,0,0,0,43.2,0.27,26)

# # # changing the input data to numpy array

# # input_data_numpy_array = np.asarray(input_data)

# # # reshape the array as we predicted for one instance

# # input_data_reshaped = input_data_numpy_array.reshape(1,-1)

# # prediction = loaded_model.predict(input_data_reshaped)
# # print(prediction)

# # if(prediction[0] == 0):
# #     print("This is not diabetic person")


# # else:
# #     print("This person is diabetic ")




#     # changing the input data to numpy array

# input_data_numpy_array = np.asarray(input_data)

#     # reshape the array as we predicted for one instance

# input_data_reshaped = input_data_numpy_array.reshape(1,-1)

# scaler = StandardScaler()

# scaler.fit(input_data_reshaped)

# std_data = scaler.transform(input_data_reshaped)
    

# prediction = loaded_model.predict(std_data)
    

# if(prediction[0] == 0):
#     print("This is not diabetic person")

# else:
#      print( "This person is diabetic ")

import numpy as np
import pickle

# Load the model and scaler
with open("trained_model_with_scaler.sav", "rb") as f:
    loaded_model, scaler = pickle.load(f)

# Input data for prediction
input_data = (0, 131, 0, 0, 0, 43.2, 0.27, 26)

# Convert input data to numpy array and reshape it
input_data_reshaped = np.asarray(input_data).reshape(1, -1)

# Standardize the input data using the loaded scaler
std_data = scaler.transform(input_data_reshaped)

# Predict using the loaded model
prediction = loaded_model.predict(std_data)

if prediction[0] == 0:
    print("This is not a diabetic person")
else:
    print("This person is diabetic")
