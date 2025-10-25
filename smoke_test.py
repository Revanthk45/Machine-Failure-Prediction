from joblib import load
import numpy as np
import pandas as pd

print('Loading device model...')
dev = load('device_failure_model.joblib')
print('Device model loaded')

# device expects 5 attributes: attribute1..attribute5
device_sample = pd.DataFrame([[50,50,50,50,50]], columns=['attribute1','attribute2','attribute3','attribute4','attribute5'])
print('Device predict:', dev.predict(device_sample))
if hasattr(dev, 'predict_proba'):
    print('Device prob:', dev.predict_proba(device_sample)[0])

print('\nLoading machine model...')
mac = load('machine_failure_model.joblib')
print('Machine model loaded')

# machine features per train_machine.py
features = ['Quality','Product Code','Process T (C)','Ambient T (C)','T Difference Squared (C^2)','Tool Lifespan (min)','Tool Lifespan/Temp Increase^2 (min/C^2)','Rotation Speed (rpm)','Torque (Nm)','Horsepower (HP)']
# create a plausible sample
machine_sample = pd.DataFrame([[1,1,35.0,25.0,(25.0-35.0)**2,100,100/((25.0-35.0)**2 or 1),1500,40.0,(1500*40.0)/5252.0]], columns=features)
print('Machine predict:', mac.predict(machine_sample))
if hasattr(mac, 'predict_proba'):
    print('Machine prob:', mac.predict_proba(machine_sample)[0])

print('\nSmoke test completed')
