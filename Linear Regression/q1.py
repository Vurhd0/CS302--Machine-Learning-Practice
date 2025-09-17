import pandas as pd

n=int(input())
rows=[]
for i in range(n):
    x_val, y_val=map(float, input().split())
    rows.append([x_val,y_val])

df=pd.DataFrame(rows, columns=['x', 'y'])

for i in range(min(5, len(df))):
    print(f"{df.loc[i,'x']:.1f} {df.loc[i,'y']:.1f}")

print(f"({len(df)},{len(df.columns)})")
print(f"{df['x'].mean():.2f} {df['x'].std(ddof=0):.2f} {df['x'].min():.2f} {df['x'].max():.2f}")
print(f"{df['y'].mean():.2f} {df['y'].std(ddof=0):.2f} {df['y'].min():.2f} {df['y'].max():.2f}")

lr = 0.01
epochs = 1000
theta0, theta1 = 0, 0
mean_x = df['x'].mean()
std_x = df['x'].std(ddof=0)
df['x'] = (df['x'] - mean_x) / std_x

def mse(y, y_hat):
    return ((y - y_hat) ** 2).mean() / 2

for _ in range(epochs):
    y_hat = theta1 * df['x'] + theta0
    lo = mse(df['y'], y_hat)
    err = y_hat - df['y']
    theta0 -= lr * err.mean()
    theta1 -= lr * (err * df['x']).mean()

print(f"Final theta0={theta0:.3f} | theta1={theta1:.3f} | Final MSE={lo:.2f}")
print(round(theta1 * ((150 - mean_x) / std_x) + theta0, 2))
print(round(theta1 * ((200 - mean_x) / std_x) + theta0, 2))
