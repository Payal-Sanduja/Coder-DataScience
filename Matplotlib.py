import matplotlib.pyplot as mp
#Create the pie chart along with its label,explode,shadow
x=[1,2,3,4,5,6,7]
l=["Apple","Mango","Papaya","Orange","Guvava","lemon","cherr"]
e=[0,0,0,0.5,0,0,0]
mp.pie(x,labels=l,startangle=90,explode=e,shadow=True)
mp.legend()

mp.show()
#change the color by using colors parameter
c=["hotpink","orange","black","blue","red","yellow","brown"]
mp.pie(x,labels=l,startangle=90,explode=e,shadow=True,colors=c)
mp.legend()

mp.show()

