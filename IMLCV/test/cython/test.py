from cpickle import c2, p2
import inspect

myp2 = p2(1, 1)
myc2 = c2(1, 1)

print(inspect.signature(myc2.__init__).parameters)
print(inspect.signature(myp2.__init__).parameters)
