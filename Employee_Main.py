import Inheritance_Sub_Class
dev_1 = Inheritance_Sub_Class.Developer('Homer', 'Simpson', 40000, 'Fortan')
dev_2 = Inheritance_Sub_Class.Developer('Bart', 'Simpson', 32000, 'Pascal')

manager_1 = Inheritance_Sub_Class.Manager('Marge', 'Simpson', 30000, [dev_1, dev_2])
print(manager_1.email)

manager_1.print_employees()
print(isinstance(manager_1, Inheritance_Sub_Class.Developer))
print(issubclass(Inheritance_Sub_Class.Developer, Inheritance_Sub_Class.Employee))
#print(help(Inheritance_Sub_Class.Developer))