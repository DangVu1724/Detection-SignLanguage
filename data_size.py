import pickle

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

print(f'Data length: {len(data_dict["data"])}')
print(f'Labels length: {len(data_dict["labels"])}')
