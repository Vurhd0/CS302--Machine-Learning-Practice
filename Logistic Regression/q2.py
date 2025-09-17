import pandas as pd
import numpy as np

n=int(input())
temp=[]

for i in range(n):
    
    s=list(map(int,input().split()))
    temp.append(s)
    

df=pd.DataFrame(temp,columns=['exam1','exam2','hours_study','admitted'])
print("First 5 rows:")
print(df.head())
print("")
print(f"Shape (N, d): {df.shape}")
print("")
print("Summary statistics:")


print(f"exam1 -> Min: {df["exam1"].min():.0f}, Max: {df["exam1"].max():.0f} Mean:{df["exam1"].mean():0.2f} Std: {df["exam1"].std(ddof=1):0.2f}")
print(f"exam2 -> Min: {df["exam2"].min():.0f}, Max: {df["exam2"].max():.0f} Mean:{df["exam2"].mean():0.2f} Std: {df["exam2"].std(ddof=1):0.2f}")
print(f"hours_study -> Min: {df["hours_study"].min():.0f}, Max: {df["hours_study"].max():.0f} Mean:{df["hours_study"].mean():0.2f} Std: {df["hours_study"].std(ddof=1):0.2f}")
print("")


def sig(x):
    return 1/(1+ np.exp(-x))
    
def loss(y_hat,y):
    #y_hat=np.clip(y_hat,1e-5,1-1e-5)
    return (-1/n)*np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

x = df[['exam1', 'exam2', 'hours_study']]
y=df['admitted']
mean=x.mean()
std=x.std(ddof=0)
x_norm=(x-x.mean())/x.std(ddof=0)
x_norm = np.c_[np.ones(x_norm.shape[0]), x_norm]
theta=np.zeros(x_norm.shape[1])



epochs=1500
lr=0.01


for i in range(epochs):
    
    y_hat=sig(x_norm@theta)
    
    #y_hat=np.clip(y_hat,1e-5,1-1e-5)
    theta =theta- lr*(x_norm.T@(y_hat-y))/n

y_hat=sig(x_norm@theta)

print("Final theta:", np.round(theta, 2))
print(f"Final loss: {round(loss(y_hat,y),2)}")


t1=[72,80,11]
t2=[150,118,20]
t1=(t1-mean)/std
t2=(t2-mean)/std

t1=np.insert(t1,0,1)
t2=np.insert(t2,0,1)


print(f"Prediction for (exam1=72, exam2=80, hours_study=11): {sig(t1@theta):.2f}")
print(f"Prediction for (exam1=150, exam2=118, hours_study=20): {sig(t2@theta):.2f}")


