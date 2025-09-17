import pandas as pd
import numpy as np
n=int(input())
size=[]
bedroom=[]
age=[]
price=[]
for i in range(n):
    s=input().split()
    size.append(float(s[0]))
    bedroom.append(float(s[1]))
    age.append(float(s[2]))
    price.append(float(s[3]))
df=pd.DataFrame({'size':size,'bedroom':bedroom,'age':age,'price':price})
for i in range(min(5,n)):
    print(f"{df.iloc[i,0]:.1f} {df.iloc[i,1]:.1f} {df.iloc[i,2]:.1f} {df.iloc[i,3]:.1f}")
print(f"({df.shape[0]},{df.shape[1]})")
for c in df.columns:
    print(f"{df[c].mean():.2f} {df[c].std(ddof=0):.2f} {df[c].min():.2f} {df[c].max():.2f}")
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x=(x-x.mean())/x.std(ddof=0)
x=pd.concat([pd.Series(1,index=x.index,name="bias"),x],axis=1)
x_np=x.values
y_np=y.values
theta=np.zeros(x_np.shape[1])
lr=0.01
epochs=300
def mse(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()/2
for _ in range(epochs):
    y_hat=x_np@theta
    theta-=(lr/n)*(x_np.T@(y_hat-y_np))
y_hat=x_np@theta
print(f"Final Theta={[round(t,3) for t in theta]}")
print(f"Final MSE={mse(y_np,y_hat):.2f}")
theta_pr=np.linalg.inv(x_np.T@x_np)@(x_np.T@y_np)
y_h=x_np@theta_pr
print(f"MSE Difference={round(abs(mse(y_np,y_h)-mse(y_np,y_hat)),5)}")
new_points=pd.DataFrame([[150,3,5],[200,4,2]],columns=["size","bedroom","age"])
new_points=(new_points-df.iloc[:,:-1].mean())/df.iloc[:,:-1].std(ddof=0)
new_points=pd.concat([pd.Series(1,index=new_points.index,name="bias"),new_points],axis=1)
for pred in new_points.values@theta:
    print(f"{pred:.2f}")
