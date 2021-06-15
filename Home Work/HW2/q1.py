import math
# defining a class and all required functions as per the problem statement
class CGraph:
    def __init__(self, w1, x1, w2, x2):
        self.x1 = x1
        self.w1 = w1
        self.x2 = x2
        self.w2 = w2

    def multiply(self, op1, op2):
        return op1 * op2

    def sin(self, op):
        return math.sin(op)

    def cos(self, op):
        return math.cos(op)

    def square(self, op):
        return op ** 2

    def add(self, op1, op2):
        return op1 + op2

    def add_constant(self, constant, op):
        return constant + op

    def one_by_x(self, op):
        return 1 / op

    def diff_one_by_x(self, op):
        return -1 / (op ** 2)

    def diff_square(self, op):
        return 2 * op

    def diff_cos(self, op):
        return -math.sin(op)

    def diff_sin(self, op):
        return math.cos(op)

# function to calculate values for forward propogation
def forward_prop(graph, w1, x1, w2, x2):
    graph = CGraph(w1, x1, w2, x2)    
    res=graph.multiply(w1,x1)
    #graph.multiply(w2,x2)
    res1=graph.sin(res)
    res2=graph.square(res1)
    res3=graph.multiply(w2,x2)
    res4=graph.cos(res3)
    res5 = res2 + res4
    res6=graph.add_constant(2,res5)
    final = graph.one_by_x(res6)
    print("Forward propagation values are:\n", w1,x1,res,res1,res2,w2,x2,res3,res4,res5,res6,final)

    return res, res1, res3, res6
# function to calclulate backwards propogation
def back_propagation(graph, res, res1, res3, res6):
    back1=1
    back2 = (1*graph.diff_one_by_x(res6))
    #print back2 5 times
    back3 = back2*graph.diff_square(res1)
    back4 = back3*graph.diff_sin(res)

    back5 = back2 *graph.diff_cos(res3)
    back_w1 = -1 * back4
    back_x1 = 1 * back4
    back_w2 = -2 * back5
    back_x2 = 2 * back5
    print("Back propagation values are :\n", back_x2,back_w2,back_x1,back_w1,back5,back4,back3,back2,back2,back2,back2,back2,back1)

#calling the function. I have considered the values as 1,-1,2 and -2 for calculations
if __name__ == '__main__':
    graph = CGraph(1, -1, 2, -2)
    res, res1, res3, res6 = forward_prop(graph, 1, -1, 2, -2)
    print("\n")
    back_propagation(graph, res, res1, res3, res6)

