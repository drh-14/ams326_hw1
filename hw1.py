
import numpy as np
import time

answer = lambda value, iterations, flops: f'Approximate Solution: {value}, Iterations: {iterations}, Approximate Number of Floating Point Operations: {flops}'

def randomUniform():
    x1,x2 = time.time(), time.time()
    return 0.25 * ((x1 + x2) % 1) + 0.50

# Question 1
class Q1:
    def __init__(self):
        self.f = lambda x: np.exp(np.power(-x, 3)) - np.power(x, 4) - np.sin(x)
        self.TOL = 0.5 * np.power(10.0, -4)
        
    #Bisection Method   
    def method1(self):
        iterations = 0
        a,b = -1, 1
        while (b - a) / 2 > self.TOL:
            iterations += 1
            c = (a + b) / 2
            if self.f(c) == 0:
                break
            elif self.f(a) * self.f(c) < 0:
                b = c
            else:
                a = c
        return (a + b) / 2
    
    #Newton's Method
    def method2(self):
        deriv = lambda x: -3 * np.power(x, 2) * np.exp(np.pow(-x, 3)) - (4 * np.pow(x, 3)) - np.cos(x)
        x = 0
        iterations = 0 
        while abs(self.f(x)) >= self.TOL:
            x -= (self.f(x) / deriv(x))
            iterations += 1
        return x
    
    #Secant Method
    def method3(self):
        x1, x2 = -1, 1
        iterations = 0
        while abs(self.f(x2)) >= self.TOL:
            temp = x2
            x2 -= (x2 - x1) * (self.f((x2)) / (self.f(x2) - self.f(x1)))
            iterations += 1
            x1 = temp
        return x2
    
    #Monte Carlo Method
    def method4(self):
        iterations = 0
        while(True):
            x = randomUniform()
            iterations += 1
            if abs(self.f(x)) < self.TOL:
                return x
            
#Question 2
class Q2:
    def __init__(self):
        self.A = np.array([[1,1,1,1,1], [1,2,4,8,16], [1,3,9,27,81], [1,4,16,64,256], [1,5,25,125,625]])
        self.b = np.array([412, 407, 397, 398, 417])
     
    #Quartic Interpolation   
    def part1(self):
        coefficients = np.dot(np.linalg.inv(self.A), self.b)
        return sum([coefficients[i] * np.pow(6, i) for i in range(len(coefficients))])
    
    def part2(self):
        pass
            
if __name__ == "__main__":
    Q1 = Q1()
    print(Q1.method1())
    print(Q1.method2())
    print(Q1.method3())
    print(Q1.method4())
    Q2 = Q2()
    print(Q2.part1())