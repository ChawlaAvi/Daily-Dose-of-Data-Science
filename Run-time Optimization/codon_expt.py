import time

def fib(n):
    if n<=1:
        return 1
    return fib(n-1) + fib(n-2)

def approximate_pi(num_terms):
    """
    Approximates pi using the series 4*(1 - 1/3 + 1/5 - 1/7...).
    
    Args:
        num_terms (int): The number of terms in the series to use.
        
    Returns:
        float: An approximation of pi.
    """
    pi_approx = 0.0
    sign = 1
    for i in range(num_terms):
        denom = 2 * i + 1
        pi_approx += sign * 4 / denom
        sign *= -1
    return pi_approx

start = time.time()
print("fib = ", fib(40))
print("fib(40) total time", time.time() - start)

start = time.time()
print("fib = ", fib(30))
print("fib(30) total time", time.time() - start)

start = time.time()
print("pi = ", approximate_pi(10**8))
print("pi total time", time.time() - start)