import numpy as np
#creating a class and defining all the required functions
class VGraph:
    def __init__(self, x, W):
        self.x = x
        self.W = W

    def multiply(self):
        return np.matmul(self.W, self.x)

    def sigmoid(self, op):
        return 1/(1+np.exp(-op))

    def l2_loss(self, op):
        return np.linalg.norm(op)**2

    def diff_l2(self, op):
        return np.multiply(2, op)

    def diff_sigmoid(self, op):
        return np.multiply((1 - self.sigmoid(op)), self.sigmoid(op))

    def diff_mult(self, op):
        return 2 * np.matmul(np.transpose(self.W), op), 2 * np.multiply(op, np.transpose(self.x))
 
 # function for forwards propogation    
def forward_propogation(x, W):
    graph = VGraph(x, W)
    fp = []
# used a list to store values
    fp.append(graph.multiply())
    fp.append(graph.sigmoid(fp[-1]))
    fp.append(graph.l2_loss(fp[-1]))

    return fp
#backward propogation
def backward_propogation(fp, x, W):
    graph = VGraph(x, W)
    bp = []

    bp.append(np.multiply(1, graph.diff_l2(fp[1])))
    bp.append(np.multiply(bp[-1], graph.diff_sigmoid(fp[0])))
    bp1, bp2 = graph.diff_mult(bp[-1]) 
    bp.append(bp1)
    bp.append(bp2)

    return bp
# calling functions
def main():
    W = np.array([[1, 2, 3], [3, 4, 5], [5, 4, 3]])
    x = np.array([[1], [0], [1]])

    fp = forward_propogation(x, W)
    print("FORWARD PROPOGATION\n", fp)
    bp = backward_propogation(fp, x, W)
    print("\nBACKWARD PROPOGATION\n")
    # print(bp)
    # using a functon to print values properly
    for i in range(len(bp)):
        for j in range(len(bp[i])):
            print(bp[i][j], '\t')
        print('\n')

if __name__ == '__main__':
    main()  
