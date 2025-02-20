#importing the modules
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#load the dataset from the github
data=sns.load_dataset("attention")
print (data)


#plot the histogram with the help of seaborn
sns.displot(data)
plt.show()
#Generate the 2d matrix of 10 rows and 10 columns with the help of random no between 1 and 100
data1=np.random.randint(low=1,high=100,size=(10,10))
print(data1)

# create the heatmap of the 2d matrix

heat=sns.heatmap(data=data1)
plt.show()

#Anchoring the colour map
min=10
max=70
heat=sns.heatmap(data=data1,vmin=min,vmax=max)
plt.show()
#using the colour map
heat=sns.heatmap(data=data1,vmin=min,vmax=max,cmap="viridis_r")
plt.show()
heat=sns.heatmap(data=data1,vmin=min,vmax=max,cmap="plasma")
plt.show()


#centering the colourmap

heat=sns.heatmap(data=data1,vmin=min,vmax=max,cmap="plasma",center=100)
plt.show()


# Display the cell value with the help of annot
heat=sns.heatmap(data=data1,vmin=min,vmax=max,cmap="viridis_r",annot=True)
plt.show()


#Customizing the separating line

heat=sns.heatmap(data=data1,vmin=min,vmax=max,cmap="viridis_r",annot=True,linewidths=3, linecolor="red")
plt.show()

#Removing the label
heat=sns.heatmap(data=data1,vmin=min,vmax=max,cmap="viridis_r",annot=True,linewidths=3, linecolor="red",xticklabels=False,yticklabels=False)
plt.show()





