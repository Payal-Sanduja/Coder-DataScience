#Create a Simple Calculator using Function in Pyhton
def sum(a,b):
    print("The Sum of the Numbers are ",a+b)
def Subtract(a,b):
    print("The difference of the  numbers are",a-b)
def Multiplication(a,b):
    print("The multiplication of the two numbers are ",a*b)
def Divison(a,b):
    print("The dividon of the two numbers are",a/b)



n1=int(input("Enter the first number"))
n2=int(input("Enter the second number"))

print("Please select operation :\n"
      \
      "1.Add\n" \
      "2.Subtract\n"\
      "3.multiply\n"\
      "4.divide\n")

select=int(input("select  operations form 1,2,3,4:"))
if select==1:
       sum(n1,n2)
elif select==2:
    Subtract(n1,n2)
elif select==3:
    Multiplication(n1,n2)
elif select==4:
    Divison(n1,n2)
else:
    print("You enter the invalid input")
       
      
