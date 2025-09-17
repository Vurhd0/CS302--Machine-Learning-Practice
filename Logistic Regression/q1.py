import pandas as pd
import numpy as np

n=int(input())
temp=[]

for i in range(n):
    
    s=list(map(int,input().split()))
    temp.append(s)
    

df=pd.DataFrame(temp,columns=['exam_score','admitted'])
print("First 5 rows:")
print(df.head())
print("")
print(f"Shape (N, d): {df.shape}")
print("")
print("Summary statistics for exam_score:")

print(f"Min: {df["exam_score"].min():.0f}")
print(f"Max: {df["exam_score"].max():.0f}")
print(f"Mean:{df["exam_score"].mean():0.2f}")
print(f"Std: {df["exam_score"].std(ddof=0):0.2f}")
print("")


def sig(x):
    return 1/(1+ np.exp(-x))
    
def loss(y_hat,y):
    y_hat=np.clip(y_hat,1e-5,1-1e-5)
    return (-1/n)*np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

theta0=0
theta1=0
epochs=1000
lr=0.01
x=df["exam_score"]
y=df["admitted"]

for i in range(epochs):
    z=theta0+theta1*x
    y_hat=sig(z)
    #y_hat=np.clip(y_hat,1e-5,1-1e-5)
    theta0=theta0- lr*(np.mean(((y_hat-y))))
    theta1=theta1- lr*(np.mean((y_hat-y)*x))
z_new=theta0+theta1*x
y_hat_new=sig(z_new)

los=loss(y_hat_new,y)    
print(f"Final theta0: {theta0:0.2f}")
print(f"Final theta1: {theta1:0.2f}")
print(f"Final Loss: {los:0.2f}")
print("")
print("")

t1=theta0+theta1*65
pred1=sig(t1)

print(f"Prediction for exam_score=65: {pred1:0.2f}")


t2=theta0+theta1*155
pred2=sig(t2)
print(f"Prediction for exam_Score=155: {pred2:0.2f}")
