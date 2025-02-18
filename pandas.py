import pandas as pd
import json
A={"marks1":[1,2,3,4],"marks":[2,3,4,56]}
var=pd.DataFrame(A)
print(var)
var1=var.to_csv("payal.csv")
var2=pd.read_csv("payal.csv")
print(var2.head(1))
print(var2.tail(3))
var4=var.to_json("payal1.json")
var5=pd.read_json("payal1.json")
print(var5.head(2))

