import numpy as np
import pandas as pd


from collections import Counter

n = int(input())
sepal_length, sepal_width, petal_length, petal_width, cl = [], [], [], [], []

for i in range(n):
    s = list(input().split())
    sepal_length.append(float(s[0]))
    sepal_width.append(float(s[1]))
    petal_length.append(float(s[2]))
    petal_width.append(float(s[3]))
    cl.append(s[4])

df = pd.DataFrame({
    'sepal_length': sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width,
    'cl': cl
})

for col in ['sepal_length','sepal_width','petal_length','petal_width']:
    df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())

np.random.seed(42)
df = df.sample(frac=1).reset_index(drop=True)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

def manhattan(a, b):
    return np.sum(np.abs(a - b))

k_values = [1, 3, 5, 7, 9]
res = {k: [] for k in k_values}

K = 5
nfold = len(x) // K

for i in range(K):
    test_idx = np.arange(i * nfold, (i + 1) * nfold)
    train_idx = np.concatenate([np.arange(0, i * nfold), np.arange((i + 1) * nfold, len(x))])
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    for k in k_values:
        corr = 0
        for t in range(len(x_test)):
            distances = []
            for j, x_train_sample in enumerate(x_train):
                d = manhattan(x_test[t], x_train_sample)
                distances.append((d, y_train[j]))
            distances.sort(key=lambda z: z[0])
            k_ne = [lab for _, lab in distances[:k]]
            mc = Counter(k_ne).most_common(1)
            if mc[0][0] == y_test[t]:
                corr += 1
        acc = corr / float(len(x_test))
        res[k].append(acc)

mean_res = {k: np.mean(res[k]) for k in res}
print(mean_res)

train_size = int(0.8 * len(x))
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

final_k=1
preds=[]
for t in range(len(x_test)):
    distances=[]
    for j, x_train_sample in enumerate(x_train):
        d = manhattan(x_test[t], x_train_sample)
        distances.append((d, y_train[j]))
    distances.sort(key=lambda z: z[0])
    k_ne = [lab for _, lab in distances[:final_k]]
    mc = Counter(k_ne).most_common(1)
    preds.append(mc[0][0])

final_acc = np.mean(preds == y_test)
print("\nFINAL EVALUATION ON TEST SET")
print(f"Test set accuracy with k={final_k}: {final_acc:.2f}")