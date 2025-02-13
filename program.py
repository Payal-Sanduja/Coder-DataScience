#Program to Calculate the average of 2 Numbers
n1=int(input("Enter the first number "))
n2=int(input("Enter the Second Number "))
Average=(n1+n2)/2
print("The avraeg of the Two Numbers is== ",Average)       

#Program to Calculate the average of 3 Numbers
n3=int(input("Enter the first number "))
n4=int(input("Enter the Second Number "))
n5=int(input("Enter the first number "))
Avg=(n3+n4+n5)/3
print("The average of the Three Numbers is== ",Avg)

#Program to calculate the average function in a list
list=[10,20,30,40]
Sum=sum(list)
length=len(list)
Average=Sum/length
print( "The Average of the Given List is as :",Average)


# Program to calculate the Factorial of the Number
n=int(input("Enter the Number: "))
fact=1
i=1
while i<=n:
    fact=fact*i
    i=i+1
print("The Factotal of the given number is:",fact)  

#Program to check whether the student is pass or not
marks=int(input("Enter your Marks:-"))
if(marks>=33):
    print("Student is Pass")
else:
    print("Student is fail")


#Program to Calculate the Grade of the student
m=int(input("enter The Marks"))
if(m>=90 and m<=100):
    print("grade is A")
elif(m>=80 and m<90):
    print("Grade is B")
elif(m>=70 and m<80):
    print("Grade is C")
else:
    print("Grade is D")
    




