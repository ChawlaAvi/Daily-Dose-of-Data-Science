import pysnooper

@pysnooper.snoop()
def add_sub(a, b):
    
    add = a+b
    sub = a-b
    
    return (add, sub)
    
add_sub(18, 10)