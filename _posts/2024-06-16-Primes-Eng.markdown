---
layout: post
title: Prime Numbers
date: 2024-06-16 00:00:00 +0300
description: Optimizing prime number calculations. # Add post description (optional)
img: key.jpg # Add image post (optional)
tags: [Primes, Cryptography] # add tag
---


## Exploring Different Ways to Calculate Prime Numbers with Python! 🐍

Did you know that there are several ways to find prime numbers, each with its own efficiency? I recently explored a few different approaches in Python and it was a fascinating experience. Shall we take a look at them?


### Tentative Division (O(n²)): 

This is the most basic approach, where we check each number up to 𝑛 n to see if it is divisible only by 1 and itself
#### Results obtained in 30s:
largest prime number: 105899

### Optimized Division (O(n√n)): 

We improved the first approach by checking divisors only up to the square root of 𝑛 n.
#### Results obtained in 30s:
largest prime number: 3857137


### One Line

Uses list comprehension to filter divisors, but is less efficient compared to the other methods.
#### Results obtained in 30s:
largest prime number: 44683


### Sieve of Eratosthenes (O(n log log n)): 

A much more efficient method that uses a list to mark multiples of each prime number found.
#### Results obtained in 30s:
largest prime number: 20597471


Here is an example of the code I used to compare these approaches, including a function to measure execution time:

Translated with DeepL.com (free version)


```python
from functools import wraps
import math
from time import perf_counter, time


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = (perf_counter() - s)
            if elapsed < 1:
                elapsed = elapsed * 1000
                msg = f"{elapsed:0.4f} ms."
            else:
                msg = f"{elapsed:0.4f} s."
            print(f"Method: {func.__name__} executed in {msg}.")

    return wrapper

def trial_division(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def optimized_division(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

@timeit
def primes_sieve(time_limit):
    prime, sieve = [], set()
    start_time = time()
    q = 2
    while (time() - start_time) < time_limit:
        if q not in sieve:
            prime.append(q)
            for multiple in range(q * q, int(q * q + time_limit * 10000), q):
                sieve.add(multiple)
        q += 1
    return prime

def one_line(time_limit):
    start_time = time()
    primes = []
    n = 2
    while (time() - start_time) < time_limit:
        divisors = [d for d in range(2, n // 2 + 1) if n % d == 0]
        if not divisors:
            primes.append(n)
        n += 1
    return primes

def find_primes_within_time_limit(time_limit, algorithm):
    primes = []
    start_time = time()
    n = 2
    while (time() - start_time) < time_limit:
        if algorithm(n):
            primes.append(n)
        n += 1
    return primes

def informs(primes):
    print(f'primes found: {len(primes)}')
    print(f'last prime found: {primes[-1]}')
    print(f'largest prime length found is: {len(str(primes[-1]))}')

@timeit
def run_algorithm(time_limit, algorithm, is_sieve=False):
    if is_sieve:
        primes = algorithm(time_limit)
    else:
        primes = find_primes_within_time_limit(time_limit, algorithm)
    informs(primes)
    return primes

if __name__ == '__main__':
    try:
        algorithms = [
            (one_line, False),
            (trial_division, False),
            (optimized_division, False),
            (primes_sieve, True)
        ]
        time_limit = 30  # seconds
        for algo, is_sieve in algorithms:
            run_algorithm(time_limit, algo, is_sieve)
    except Exception as e:
        print(e)
    finally:
        print('Done!')

```

## Optimization suggestion

This has been a short journey compared to the other posts, but do you have any other suggestions? 

I leave you with a challenge, to implement the segmented sieve method: An extension of Sieve of Eratosthenes, which divides the interval into segments to reduce memory usage, making it efficient for finding primes in large intervals as well of course.



To see other projects, visit my [project portfolio](https://github.com/mabittar/Portfolio).
