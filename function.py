#Define a Function without  passing arguments
def fun():
    print("welcome to my Python programs")

fun()

#Define a function by passing an arguments
def sum(a,b):
    print(a+b)
sum(10,20)  
def subtract(c,d):
    print(c-d)
subtract(10,5)

def multiply(c,d):
    print(c*d)
multiply(10,5)
def divide(c,d):
    print(c/d)
divide(10,5)
def modulus(c,d):
    print(c%d)
modulus(10,5)

#function by using the default values
def d(a,b=5):
    print(a+b)
    
d(20)
#Function using return statement
def area(l,b):
   return l*b
s=area(5,4)
print s

# program of lambda Expression

x=lambda a,b:a+b
print(x(5,6))
          
