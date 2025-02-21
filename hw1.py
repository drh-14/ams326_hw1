
import numpy as np
import random

# Question 1
class Q1:
    def __init__(self):
        self.f = lambda x: np.exp(np.power(-x, 3)) - np.power(x, 4) - np.sin(x)
        self.r = 0.641583
        self.funcFLOPS = 43
        self.derivFLOPS = 45
        self.TOL = 0.5 * np.power(10.0, -4)
        
    def answer(self, value, iterations, flops):
      return f'Approximate Solution: {value}, Iterations: {iterations}, Approximate Number of Floating Point Operations: {flops}'    
        
    #Bisection Method   
    def method1(self):
        flops = 0
        iterations = 0
        a,b = -1, 1
        while (b - a) / 2 > self.TOL:
            flops += 2
            iterations += 1
            c = (a + b) / 2
            flops += 2
            if self.f(c) == 0:
                flops += self.funcFLOPS
                break
            elif self.f(a) * self.f(c) < 0:
                flops += 2 * self.funcFLOPS
                b = c
            else:
                a = c
        flops += 2
        return self.answer((a + b) / 2, iterations, flops)
    
    #Newton's Method
    def method2(self):
        flops = 0
        deriv = lambda x: -3 * np.power(x, 2) * np.exp(np.pow(-x, 3)) - (4 * np.pow(x, 3)) - np.cos(x)
        x = 0
        iterations = 0 
        while abs(self.f(x)) >= self.TOL:
            flops += self.funcFLOPS + 1
            x -= (self.f(x) / deriv(x))
            flops += self.funcFLOPS + self.derivFLOPS + 2
            iterations += 1
        return self.answer(x, iterations, flops)
    
    #Secant Method
    def method3(self):
        x1, x2 = -1, 1
        iterations = 0
        flops = 0
        while abs(self.f(x2)) >= self.TOL:
            flops += self.funcFLOPS + 1
            temp = x2
            x2 -= (x2 - x1) * (self.f((x2)) / (self.f(x2) - self.f(x1)))
            flops += 3 * self.funcFLOPS + 5
            iterations += 1
            x1 = temp
        return self.answer(x2, iterations, flops)
    
    #Monte Carlo Method
    def method4(self):
        iterations = 0
        flops = 0
        while True:
            x = random.uniform(0.50, 0.75)
            flops += 4
            iterations += 1
            if abs(x - self.r) < self.TOL:
                flops += self.funcFLOPS + 1
                return self.answer(x, iterations, flops)
            
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
        A = np.array([[1,1,1], [1,2,4], [1,3,9], [1,4,16],[1,5,25]])
        b = np.array([412, 407,397, 398, 417])
        transpose = np.linalg.matrix_transpose(A)
        pseudo = np.linalg.matmul(transpose, A)
        newB = np.linalg.matmul(transpose, b)
        coefficients = np.linalg.matmul(np.linalg.inv(pseudo), newB)
        return sum([coefficients[i] * np.pow(6, i) for i in range(len(coefficients))])
        
        
if __name__ == "__main__":
    Q1 = Q1()
    print(Q1.method1())
    print(Q1.method2())
    print(Q1.method3())
    print(Q1.method4())
    Q2 = Q2()
    print(Q2.part1())
    print(Q2.part2())