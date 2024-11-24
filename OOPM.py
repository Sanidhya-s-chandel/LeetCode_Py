# ---------------------------------------->>>>> creating class and object <<<<<-----------------------------------------------------

# class Person:
#     name = "Sanidhya"
#     occupation = "Student"
#     networth = 1000

# x = Person()  # object creation
# print(x.name) # calling variable value through object

# x.name = "shanu"        # changing value of the variable
# print(x.name) 

# ---------------------------------------------->>>>> ((methods:)) <<<<<----------------------------------------------------------

# class Person:
#     name = "Sanidhya"
#     occupation = "Student"
#     networth = 1000

#     def info(self):             # self is an object for the method is called
#         print(f"{self.name} is a {self.occupation} with networth {self.networth}")

# x = Person()
# x.info()        # method calling
# y = Person()
# y.name = "Shanu"
# y.occupation = "Developer"

# y.info()

# ---------------------------------------->>>>> ((Constructor)) <<<<<------------------------------------------------------------

# class Person:
#     # name = "Sanidhya"
#     # occupation = "Student"
#     # networth = 1000
#     def __init__(self,name,occupation,networth):             # dunder method (constructor)
#         # print("HELLO, i will run")                         
#         self.name = name                                    # self.variable --> self will assign the passed value to the variable 
#         self.occupation = occupation
#         self.networth = networth                        # (this is a parameterized constructor)

#     def info(self):
#         print(f"{self.name} is a {self.occupation} with networth {self.networth}")

# x = Person("Sanidhya","Student",1500) # every time the class will be calle the constructor will run (**)
# # print(x.name,x.networth)
# x.info()
# # (**) ---->> x will be passed in the self as an argument and is counted automatically so it means c.name,c.occ. etc respectively

# # types of constructors : (1) Parameterized constructor (2) default constructor

# ---------------------------------------->>>> ((Decorators)) <<<<-------------------------------------------------------------

# def greet(fx):
#     def mfx(x):
#         print("Good Morning.... :)")
#         fx(x)
#         print("Thank You for using this function...... :)")
#     return mfx

# @greet          # it actually modify the original function || it is short hand for greet(table)():
# def table(x):
#     for i in range(1,11):                          # *agrs is used to take tuples as arguments
#         print(f"{x} * {i} =",x*i)                  # **kwargs is used to take it as [key:value] pair

# x = int(input("Enter the number : "))
# table(x)

# ------------------------------------->>>> ((Getters and Setters)) <<<<----------------------------------------------------------

# class Myclass:
#     def __init__(self,value):
#         self._value = value
    
#     def show(self):
#         print(f"value is {self._value}")

#     @property        # (this is a Getter) --->> used to access the private variables of the class || (to show them only no changnes)
#     def ten_value(self):
#         return 10*self._value
    
#     @ten_value.setter       # (this is a setter) --->> used to change the value of the private variable
#     def ten_value(self,New_value):
#         self._value = New_value/10    
       
# obj = Myclass(10)
# # print(obj.ten_value)
# # obj.show()

# obj.ten_value = 67     # this will not change the value of getter without the setter 
# print(obj.ten_value)
# obj.show()

# ---------------------------------------->>>>> ((Inheritance)) <<<<<-------------------------------------------------------------

# class Employee:
#     def __init__(self,name,id):
#         self.name = name
#         self.id = id

#     def showDetails(self):
#         print(f"The name of employee is {self.name} and employee id is {self.id}")

# # x = Employee("Sanidhya singh",3108)
# # x.showDetails()

# # x2 = Employee("Shanu singh",3308)
# # x2.showDetails()

# # x3 = Employee("Dhramraj singh",4002)
# # x3.showDetails()

# # x4 = Employee("Aanya singh",2112)
# # x4.showDetails()

# class Programmer(Employee):
#     def showLanguage(self):
#         print("The default language is pyrhon")

# # x.showLanguage()      # this will not work and show that the class has no attribute

# x2 = Programmer("Sanidhya",3108)        # The programmer can access everything in employee but vice-vera is not possible
# x2.showDetails()                # this can show the details of employee while being created in programmer
# x2.showLanguage()

# #( key-Fact there is no such thing like pulbic, protected and private in python (no access modifiers)
# # it is just conventional by users not by the python)

# examples:-
# =======================================>> ((PUBLIC)) <<============================================================

# class Employee:
#     def __init__(self):
#         self.name = "Sanidhya"  #(public that can be accessed)

# x = Employee()
# print(x.name)  # accessable from outside so it can be treated as public

# =======================================>> ((PRIVATE)) <<============================================================

# class Employee:
#     def __init__(self):
#         self.__name = "Sanidhya"  #(private that can not be accessed) (__variable (double underscore) is used to make it private)

# x = Employee()
# # print(x.__name) ---->> can not be accessed directly but can be accessed indirectly
# print(x._Employee__name) # by this it is been accessed (this is name-mangling)

# print(x.__dir__()) # is used to see what all things can be accessed in the class

# =======================================>> ((PROTECTED)) <<============================================================

# # protected is defined with a _Single underscore

# class Student:
#     def __init__(self):
#         self._name = "Sanidhya"

#     def _funName(self):         # protected method
#         return "Saloni Gite"
    
# class Subject(Student):     # inherted
#     pass

# x = Student()
# x2 = Subject()

# print(x._name)      # calling by student class object
# print(x._funName())

# print(x2._name)  # calling by subject class object
# print(x2._funName())   

# # protected is not in python it is just for naming convention and __Private means for name_mangling

# ---------------------------------------- ((Static Methods)) ----------------------------------------------------------

# class Math:
#     def __init__(self,num):
#         self.num = num

#     def addofnum(self,num2):
#         self.num += num2

#     @staticmethod       # this can be directly used outside the class
#     def add(a,b):
#         return a+b
    
# # a = Math(5)
# # print(a.num)

# # a.addofnum(10)
# # print(a.num)
    
# print(Math.add(5,10)) # ---------->>> like this

# noramlly we do not use the name of the class directly but here we can use bcoz it is a static fun().

# ----------------------------->>>>> ((Instance variable VS class variable)) <<<<<--------------------------------------------

# class Employee:
#     company = "Samsoong"    # class variable (this is constant for everyone)

#     def __init__(self,name):
#         self.name = name
#         self.rasie = 0.02   # this is an instance variable (created at an instant and can be changed)

#     def showDetails(self):
#         print(f"The name of employee is {self.name} and raise is {self.rasie} in company {self.company}")

# emp1 = Employee("Sanidhya")
# emp1.rasie = 0.33
# emp1.company = "Samsung"      # this will only change for this object
# emp1.showDetails()                      # both are same it is converted into this from (Employee.showDetails(emp1))
# # Employee.showDetails(emp1)          

# Employee.company = "Google"     # this will set default/class variable to the assigned value 
# emp2 = Employee("Roshan")
# emp2.company = "tesla"  # can be changed here (only for this object)
# emp2.showDetails()

# ------------------------------->>>>> ((class Methods)) <<<<<------------------------------------------------------------------

# class Employee:
#     company = "Apple"

#     def show(self):
#         print(f"the name is {self.name} and company is {self.company}")

#     @classmethod  # using this the 1st variable will be taken as a class (default it is instance)
#     def changeCompany(cls, newCompany):
#         cls.company = newCompany    # this will take the variable as instance only (cls is just a replacement of self as a variable)

# e1 = Employee()
# e1.name = "Sanidhya"
# e1.show()
# e1.changeCompany("Samsoong")
# e1.show()

# print(Employee.company)

# -------------------------------->>>>> ((classmethod as alternative constructor)) <<<<<------------------------------------------

# class Employee:
#     def __init__(self,name,salary):
#         self.name = name
#         self.salary = salary

#     @classmethod
#     def fromStr(cls, string):
#         return cls(string.split("-")[0],int(string.split("-")[1]))

# obj = Employee("Sanidhya singh",150000)
# print(obj.name,obj.salary)

# string = "Dhramraj-80000"

# # obj2 = Employee(string.split("-")[0],string.split("-")[1])    # (.split will convert the string into list by the sepator)
# # print(obj2.name,obj2.salary)        # we use index to assigin the element as the arrguments 

# obj2 = Employee.fromStr(string)
# print(obj2.name,obj2.salary)

# --------------------------------------->>>>> ((dir,__dict__ and help etc Methods)) <<<<<----------------------------------------

# x = [1,2,3,4]
# print(dir(x))       # dir is used to know the methos that can be applied on the following
# print(x.__add__)

# class Person:
#     def __init__(self,name,age):
#         self.name = name
#         self.age = age

# x = Person("Sanidhya",20)
# print(x.__dict__)        # return the data of variable and its value ib dictionary(key-value) pair

# print(help(Person)) # used to get the overall view of an object

# {{SUPER KEYWORD}} :-

# class Parent:
#     def parent_method(self):
#         print("This is the parent class method")

# class Child(Parent):
#     def child_method(self):
#         print("This is the chid class method")
#         super().parent_method()         # use to call constructor/methods of the super class

# child = Child()
# child.child_method()

# [[Dunder Methods]] :-

# those methods which start from double underscore and ends with double underscore are dunder methods (for eg:= __init__())

# class Employee:
#     name = "Sanidhya"
#     age = 20

#     def __len__(self):
#         x = 0
#         for i in self.name:
#             x += 1
#         return x          # __str__,__repr__,__call__
                                                    
# x = Employee()
# print(x.name)
# print(len(x))

# --------------------------------------------->>>>> ((Method Overloading)) <<<<<--------------------------------------------------

# class Shape:
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y

#     def area(self):
#         return self.x * self.y
    
# # sq = Shape(3,5)
# # print(sq.area())
    
# class Circle(Shape):
#     def __init__(self, radius):
#         self.radius = radius

#     def area(self):     # overloading (using the same method again with diff. parameters)
#         return 3.14*2*self.radius*self.radius

# cr = Circle(5)
# print(cr.area())

# ------------------------------------>>>>>> ((operator overloading)) <<<<<<--------------------------------------------------------

# class vector:
#     def __init__(self,i,j,k):
#         self.i = i
#         self.j =j
#         self.k = k

#     def __str__(self):
#         return f"{self.i}i + {self.j}j + {self.k}k"
    
#     def __add__(self,x):
#         return vector(self.i + x.i, self.j + x.j, self.k + x.k)
    
# v = vector(3,5,6)
# print(v)

# v2 = vector(6,8,12)
# print(v2)

# print(v + v2)

# ------------------------------------->>>> ((Single inheritance)) <<<<-----------------------------------------------------------

# class Animal:

#     def __init__(self,name,species):
#         self.name = name
#         self.species = species

#     def make_sound(self):
#         print("Sound made by this animal")

# class Dog(Animal):
#     def __init__(self, name, breed):
#         Animal.__init__(self,name, species = "Dog")
#         self.breed = breed

#     def make_sound(self):
#         print("Bark!!")

# # d = Dog("Dog","Greman shepard")
# # d.make_sound()

# # a = Animal("dog","Mamal")
# # a.make_sound()
        
# # QUICK QUESTION

# class Cat(Animal):
#     def __init__(self, name, breed):
#         Animal.__init__(self,name, species = "cat")
#         self.breed = breed

#     def make_sound(self):
#         print("The cat says meowww !!")

#     def nature(self):
#         print("The cas is sleepy !!")

# c = Cat("jaggu","american")
# c.make_sound()
# c.nature()

# ------------------------------------->>>> ((Multiple inheritance)) <<<<-----------------------------------------------------------

# ------------>> multiple parents class in a single class

# class Employee:
#     def __init__(self,name):
#         self.name = name

#     def show(self):
#         print(f"The name is {self.name}")

# class Dancer:
#     def __init__(self,dance):
#         self.dance = dance

#     def show(self):
#         print(f"The dance is {self.dance}")

# class DanceEmployee(Employee,Dancer):
#     def __init__(self,dance,name):
#         self.dance = dance
#         self.name = name

# # x = DanceEmployee("Hip hop","sanidhya")
# # print(x.name,x.dance)
# # x.show()
# print(DanceEmployee.mro()) # to know the order of checking

# ------------------------------------->>>> ((Multilevel inheritance)) <<<<--------------------------------------------------------

# derive cls derives another cls (class1 -->> class2 -->> class3 and soo on)

# class Animal:
#     def __init__(self,name,species):
#         self.name = name
#         self.species = species

#     def show_details(self):
#         print(f"Name : {self.name}")
#         print(f"species : {self.species}")

# class Dog(Animal):
#     def __init__(self,name,breed):
#         Animal.__init__(self,name,species="Dog")
#         self.name = name
#         self.breed = breed

#     def show_details(self):
#         Animal.show_details(self)
#         print(f"Breed : {self.breed}")

# class GoldenRetriver(Dog):
#     def __init__(self,name,color):
#         Dog.__init__(self,name,breed="Golden Retriver")
#         self.color = color

#     def show_details(self):
#         Dog.show_details(self)
#         print(f"color : {self.color}")

# x = GoldenRetriver("Tommy","Brown")
# x.show_details()
# print(GoldenRetriver.mro())

# ---------------------------------->>>> ((Hybrid && Hierarchical inheritance)) <<<<--------------------------------------------

# two or more types of inheritance in a single is called Hybrid inheritance

# class Base:
#     pass

# class Dervie_1(Base):
#     pass

# class Derive_2(Base):
#     pass

# class Derive_3(Dervie_1,Derive_2):
#     pass

# herarchical is a type where more than one class is derived from the single parent class.

# class Base:
#     pass

# class D1(Base):
#     pass

# class D2(Base):
#     pass

# class D3(Base):
#     pass

# class D4(D1):
#     pass

# -------------------------------------------->>>> ((Time Module)) <<<<--------------------------------------------------------

# import time

# def usingwhile():
#     i = 0
#     while i <= 5000:
#         print(i)
#         i += 1

# def usingFor():
#     for i in range(0,5001):
#         print(i)

# init = time.time()  # shows time form the epoch time(time where the module was started (1st jan 1970)) im sec.
# usingFor()
# print(time.time() - init)

# init = time.time()
# usingwhile()
# print(time.time() - init)

# print(4)
# time.sleep(3) # makes the code wait to execute after this line (in sec.)
# print("this is printed after 3 secounds..")

# t = time.localtime()
# formatted_time = time.strftime("%Y-%m-%d %H:%M:%S",t) # format time according to you (shows current time)
# print(formatted_time)

# ---------------------------------------->>>> ((command line utililty)) <<<<-----------------------------------------------------

# import argparse
# import requests

# def download_file(url,Download):
#     # NOTE the stream=True parameter below

#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(Download, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192): 
#                 # If you have chunk encoded response uncomment if
#                 # and set chunk_size parameter to None.
#                 #if chunk: 
#                 f.write(chunk)
#     return Download

# parser = argparse.ArgumentParser()                        # '85' code-with-harry(incomplete h baad me dhek lena if in use)

# # add command line argumrnts

# parser.add_argument("url",help = "Url of the file to download")
# parser.add_argument("output", help = "By which name you want to save your file")

# # Parse the arguments
# args = parser.parse_args()

# # use the arguments in your code

# print(args.Url)
# print(args.Output)
# download_file(args.url,args.output)

# ---------------------------------------->>>> ((Walrus Operator)) <<<<------------------------------------------------------------

# a = True
# print(a := False) # walrus operator

# x = [1,2,3,4,5,6,7,8,9,10]

# while (n := len(x) > 0):
#     print(x.pop())

# walrus operator assigns a values to variable as a part of a larger expression

# food = list()

# while True:
#     x = input("what food do u like : ") -------->> normal code
#     if x == "quit" or x == "Quit":
#         break

#     food.append(x)

# print(food)

# food = list()

# while (x := input("What food do u like : ")) != "quit":  # walrus used code
#     food.append(x)

# print(food)

# # SHUTIL Module

# import shutil
# import os

# shutil.copy("Basic_Pratice.py","copy.py") # kisi bhi file ko kahi pe bhi copy karta h (file_to_be_copied,file_to_be_copied_in)

# shutil.copytree("Projects","copy") # to copy folder

# shutil.move("Data.txt","Projects\Student_challenge\Students_10.json") # move files and can also change data type

# shutil.rmtree("copy") # delete folder (only folders not capable of file)

# os.remove("copy.py") # delete can be done from os

# ------------------------------------------>> Request Module <<----------------------------------------------------------------

# import requests

# response = requests.get("https://www.codewithharry.com")
# print(response.text) 

# ---------------------------------------->>> Generators <<<-------------------------------------------------------

def my_gen():
    for i in range(5):
        yield i

gen = my_gen()

print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))