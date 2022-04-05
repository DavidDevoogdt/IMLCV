import cython

class p1:
    def __init__(self,*args,**kwargs) -> None:
        pass

class p2(p1):
    def __init__(self,a,b):
        self.a = a
        self.b = b

cdef class c1:
    def __init__(self,*args,**kwargs) -> None:
        pass

cdef class c2(c1):
    cdef int a,b
    def __init__(self,int a, int b):
        self.a = a
        self.b = b

