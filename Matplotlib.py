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


#create the bar graph
y=[10,20,30,40]
z=[28,50,76,54]
mp.bar(y,z)
mp.show()
# we can also create the bar graph  horizontally 
mp.barh(y,z)
mp.show()
# change the color of the bar graph
mp.barh(y,z,color="black")
mp.show()
mp.bar(y,z,color="red")
mp.show()
# chnage the width of the bar graph
mp.bar(y,z,color="brown",width=1.6)
mp.show()
# change the height of the bar graph in barh
mp.barh(y,z,color="brown",height=1.6)
mp.show()



# Scatter plot
mp.scatter(y,z)
mp.show()
# comparing the 2 scatterplot
mp.scatter(y,z)
mp.scatter(x,y)
mp.show()
# change the color of the scatter plot

mp.scatter(y,z,color="green")
mp.scatter(x,y,color="red")


mp.show()
mp.show()


# use of cmap
colors=[5, 20, 15, 40, 0, 50, 55]
mp.scatter(x, y, c=colors, cmap='viridis')

mp.colorbar()
mp.show()
