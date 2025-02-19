import matplotlib.pyplot as mp
import numpy  as np
#Create the pie chart along with its label,explode,shadow
x=[1,2,3,4,5,6,7]
l=["Apple","Mango","Papaya","Orange","Guvava","lemon","cherr"]
e=[0,0,0,0.5,0,0,0]
mp.pie(x,labels=l,startangle=90,explode=e,shadow=True)
mp.legend()

mp.show()
#change the color by using colors parameter and Add the title as well
c=["hotpink","orange","black","blue","red","yellow","brown"]
mp.pie(x,labels=l,startangle=90,explode=e,shadow=True,colors=c)
mp.legend(title="Fruits Name")

mp.show()
#create the bar graph
y=[10,20,76,40,50,60,70]
z=[28,50,76,54,54,76,32]
mp.bar(y,z,width=1.6)
mp.show()
# we can also create the bar graph  horizontally 
mp.barh(y,z,height=1.6)
mp.show()
# change the color of the bar graph
mp.barh(y,z,color="black",height=1.6)
mp.show()
mp.bar(y,z,color="red",width=1.6)
mp.show()
# chnage the width of the bar graph
mp.bar(y,z,color="brown",width=1.6)
mp.show()
# change the height of the bar graph in barh
mp.barh(y,z,color="brown",height=1.6)
mp.show()



# Scatter plot
mp.scatter(y,z)
mp.legend(title="Scatter Point")
mp.show()
# comparing the 2 scatterplot
mp.scatter(y,z)
mp.legend(title="Scatter Point")
mp.scatter(x,y)
mp.legend(title="Scatter Point")
mp.show()
# change the color of the scatter plot

mp.scatter(y,z,color="green")
mp.legend(title="Scatter Point")
mp.scatter(x,y,color="red")
mp.legend(title="Scatter Point")


mp.show()
mp.show()


# use of cmap
colors=[5, 20, 15, 40, 0, 50, 55]
mp.scatter(x, y, c=colors, cmap='viridis')
mp.legend(title="Scatter Point")

mp.colorbar()
mp.show()


