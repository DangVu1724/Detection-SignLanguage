import os
import pickle
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

# Load dữ liệu
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

processed_data = []
for sample in data_dict['data']:
    if len(sample) == 42:
        sample += [0] * 42  # Bổ sung thêm các giá trị 0 nếu mẫu thiếu tọa độ
    processed_data.append(sample)

data = np.asarray(processed_data)
labels = np.asarray(data_dict['labels'])

print(f'Loaded {len(data)} samples with {data.shape[1]} features per sample.')
print(f'Total {len(set(labels))} classes.')

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

start_time = time.time()

model = RandomForestClassifier(n_estimators=1000, random_state=1000)
model.fit(x_train, y_train)

end_time = time.time()
training_time = end_time - start_time
print(f'Training completed in {training_time:.2f} seconds.')

y_predict = model.predict(x_test)

accuracy = accuracy_score(y_predict, y_test)
print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

conf_matrix = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_predict)
print("Classification Report:")
print(class_report)

y_proba = model.predict_proba(x_test) 
logloss = log_loss(y_test, y_proba)
print(f'Log Loss: {logloss:.4f}')

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
