import pandas as pd
import json
A={"marks1":[1,2,3,4],"marks":[2,3,4,56]}
var=pd.DataFrame(A)
print(var)
#create a series from the given list
import pandas as pd
a=[1,2,3,4,5,6,7]
var=pd.Series(a)
print(var)
print("The minimum number among the given series is :-",var.min())
print("The maximum number among the given series is :-",var.max())
print("The mean of the given series is:-",var.mean())
print(type(var))



#Accessing the series Elements
print("The 1st index value of var is :-",var[0])
print("The 2nd index value of var is :-",var[1])
print("The 3rd  index value of var is :-",var[2])
print("The 4th index value of var is :-",var[3])
print("The 5th index value of var is :-",var[4])
print("The 6th index value of var is :-",var[5])
print("The 7th index value of var is :-",var[6])


# we can also change the label
var1=pd.Series(a,index=["Ist","IInd","IIIrd","IV","V","VI","VII"])
print(var1)


#create a series from the given dictonary

b={1:"payal",2:"Ekta",3:"neha"}
var2=pd.Series(b)
print(var2)


#Create a dataframe from the given Dictonary

data={
    "Data1":[1,2,3,4,5],
    "Data2":[2,4,6,8,10],
    "Data3":[1,2,3,4,5],
    "Data4":[4,5,6,7,8]}
var3=pd.DataFrame(data)
print(var3)

# we can also change the label
var4=pd.DataFrame(data,index=["Ist","IInd","IIIrd","IV","V"])
print(var4)
 
#use of loc method(accessing the specified no of rows)
print(var3.loc[[0,1,2]])


#create the csv file
var4.to_csv("payal.csv",index=False)

#read the csv file
var5=pd.read_csv("payal.csv")
print(var5.head(1))
print(var5.tail(1))
print(var5.info())
print(var5.describe())



var1=var.to_csv("payal.csv")
var2=pd.read_csv("payal.csv")
print(var2.head(1))
print(var2.tail(3))
var4=var.to_json("payal1.json")
var5=pd.read_json("payal1.json")
print(var5.head(2))

