import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dbn.tensorflow import SupervisedDBNRegression
from sklearn.inspection import permutation_importance


def data_prerocessing (Input,Output):
    #Input = (Input - np.mean(Input,axis = 0))/np.std(Input,axis = 0)
    Input = Input/np.std(Input,axis = 0)
    (qt,Removal,finalpH,LeachingMg) = (Output[:,0],Output[:,1],Output[:,2],Output[:,3])
    
    fixpoint = dict(qt =  2.03, 
                    Removal = np.max(Removal)- np.min(Removal),
                    finalpH = np.max(finalpH)- np.min(finalpH),
                    LeachingMg = 1.706)
    
    qt_processed = np.emath.logn(fixpoint['qt'],qt)
    Removal_processed = Removal/fixpoint['Removal']
    finalpH_processed = finalpH/fixpoint['finalpH']
    LeachingMg_processed = np.emath.logn(fixpoint['LeachingMg'],LeachingMg+1)    
    Output = np.stack((qt_processed,Removal_processed,finalpH_processed,LeachingMg_processed),axis = 1)
    
    return (Input,Output,fixpoint)

def cal_result(prediction,fix_point):
    
    qt_result = np.power(fix_point['qt'],prediction[:,0])
    Removal_result = prediction[:,1] * fix_point['Removal']
    finalpH_result = prediction[:,2] * fix_point['finalpH']
    LeachingMg_result = np.power(fix_point['LeachingMg'],prediction[:,3])-1
    result = np.stack((qt_result,Removal_result,finalpH_result,LeachingMg_result),axis = 1)
    
    return result


#Loading data
df= pd.read_csv('Data-for-P-MgO.csv')
data = pd.DataFrame.to_numpy(df)   
print(data.shape)


#Data preprocessing
#input of model
X = data[:,0:15]
#output of model
Y = data[:,15:19]
X_,Y_ ,fix_point= data_prerocessing(X,Y)
print(fix_point)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.3, random_state=1)

# Data scaling
#min_max_scaler = MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)


# Training
#regressor = SupervisedDBNRegression(hidden_layers_structure=[64,16,16,8,4],
                                    #learning_rate_rbm=0.001,
                                    #learning_rate=0.05,
                                    #n_epochs_rbm=100,
                                    #n_iter_backprop=200000,
                                    #batch_size=512,
                                    #activation_function='relu')
#regressor.fit(X_train, Y_train)
#regressor._fine_tuning(X_train, Y_train)

# Save models
#regressor.save('./Model/DBN/DBN_[64,32,16,8].h5')
#regressor.save('./Model/DBN/DBN_[64,16,16,8,4].h5')

# Load trained models
#regressor = SupervisedDBNRegression.load('./Model/DBN/DBN_[64,32,16,8].h5')
regressor = SupervisedDBNRegression.load('./Model/DBN/DBN_[64,16,16,8,4].h5')


# Evaluate model
predictions = regressor.predict(X_test)
predictions = cal_result(predictions,fix_point)

Y_t = cal_result(Y_test,fix_point)
y_test_f = np.ndarray.flatten(Y_t)
predictions_f = np.ndarray.flatten(predictions)

mae = mean_absolute_error(y_test_f, predictions_f)
rmse = np.sqrt(mean_squared_error(y_test_f, predictions_f))
R_score = r2_score(y_test_f, predictions_f)
print(mae,rmse,R_score)

fig = plt.figure()
plt.figure(figsize = (40, 20))
for index in range(4):
    plt.subplot(2,2,index+1)
    plt.plot(predictions[:,index],label = 'Predict')
    plt.plot(Y_t[:,index],label = 'Actual')
    plt.legend(['Predict', 'Real'])

# Save predicted test and train data sets
def evaluate_DBN_model(X_test, Y_test, fix_point, model_path):
    model = SupervisedDBNRegression.load(model_path)
    Y_pred = cal_result(model.predict(X_test),fix_point)
    Y_true = cal_result(Y_test,fix_point)
    metrics = {var: {
        'mae': mean_absolute_error(Y_true[:, i], Y_pred[:, i]),
        'rmse': np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i])),
        'r2': r2_score(Y_true[:, i], Y_pred[:, i])
    } for i, var in enumerate(['qt', 'Removal', 'finalpH', 'LeachingMg'])}
    return metrics

#test_metrics = evaluate_DBN_model(X_test, Y_test, fix_point, './Model/DBN/DBN_[64,32,16,8].h5')
test_metrics = evaluate_DBN_model(X_test, Y_test, fix_point, './Model/DBN/DBN_[64,16,16,8,4].h5')
print(test_metrics)

train_predictions = cal_result(regressor.predict(X_train), fix_point)
test_predictions = cal_result(regressor.predict(X_test), fix_point)

np.savetxt("Predict_Train_DBNv2.csv", train_predictions, delimiter=",")
np.savetxt("Predict_Test_DBNv2.csv", test_predictions, delimiter=",")
print('Predict_Train and Predict_Test were saved')


# Feature importances
result = permutation_importance(regressor, X_test, Y_test, n_repeats=10, random_state=0)

importances = result.importances_mean
feature_importances = pd.DataFrame(data={'feature': range(X_train.shape[1]), 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
feature_importances['importance'] = feature_importances['importance'] / feature_importances['importance'].sum()

feature_importances.to_csv('FeIm-DBN_[64,32,16,8].csv', index=False)
print('Feature importances were saved')