import pandas as pd
import json
A={"marks1":[1,2,3,4],"marks":[2,3,4,56]}
var=pd.DataFrame(A)
print(var)
#create a series from the given list
import pandas as pd
a=[1,2,3,4,5,6,7]
vara=pd.Series(a)
print(vara)
print("The minimum number among the given series is :-",vara.min())
print("The maximum number among the given series is :-",vara.max())
print("The mean of the given series is:-",vara.mean())
print(type(vara))



#Accessing the series Elements
print("The 1st index value of var is :-",vara[0])
print("The 2nd index value of var is :-",vara[1])
print("The 3rd  index value of var is :-",vara[2])
print("The 4th index value of var is :-",vara[3])
print("The 5th index value of var is :-",vara[4])
print("The 6th index value of var is :-",vara[5])
print("The 7th index value of var is :-",vara[6])


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
var5=vara.to_csv("payal.csv",index=False)

#read the csv file
var6=pd.read_csv("payal.csv")
print(var6.head(1))
print(var6.tail(1))
print(var6.info())
print(var.describe())


print("json file creation")

# Create the json file
var7=var3.to_json("payal11.json")
# Read the json file
var8=pd.read_json("payal11.json")
print(var8.head(2))


# Use of dropna
var9=pd.read_csv("book1.csv")
var10=var9.dropna()
print(var10)
print(var9)


#Use of fillna
var11=pd.read_csv("book1.csv")
var12=var11.fillna(11)
print(var12)


#Use of fillna with inplace(inplace will do the permanent cange in the originl file)
var13=pd.read_csv("book1.csv")
var14=var13.fillna(11,inplace=True)
print(var14)
print(var13)


#Check where there is a duplicacy or not in the csv file
print(var13.duplicated())

# delete the duplicte value from the csv file
var13.drop_duplicates(inplace = True)
print(var13)

#Merge operation (use to merge 2dataframe based on common column)
d1={1:["Payal",23,5],"Age":[12,50,43],3:["Preeti",50,65]}
d2=pd.DataFrame(d1)
print(d2)
d3={1:["Payal",23,5],"Age1":[13,50,43],31:["Preeti",50,65]}
d4=pd.DataFrame(d3)
print(d4)
print(pd.merge(d2,d4))


#Concat operation
res=pd.concat([d2,d4])
print(res)
res1=pd.concat([d2,d4],axis=1)
print(res1)

# use of .sort_values(use to arrange in ascending and descending order)

#Arrange in ascending order

sort=d2.sort_values(by="Age",ascending=True)
print(sort)
print()

#Arrange in Descending order
res2=d2.sort_values(by="Age",ascending=False)
print(res2)
















