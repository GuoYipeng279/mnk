import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from time import time

class Game:
    def __init__(self,abp=False):
        self.board = None
        self.abp = abp

    def play(self,m,n,k):
        self.initialize_game(m,n,k)
        self.drawboard()
        rec = time()
        if self.abp: self.max(-1000,1000)
        else: self.max()
        print("Time:",time()-rec)
        turn = 0
        while not self.is_terminal(self.chess_max if turn & 1 else self.chess_min):
            turn += 1
            if turn & 1:
                hint, sc = self.play_max()
                if hint is None: break
                mx,my = hint
                self.drawboard((mx,my))
                if True:
                    self.chess_max[mx,my] = 1
                    self.last.append((mx,my))
                    print("Max moved at {0},{1}".format(mx,my))
                    print("Score:",sc)
                    self.draw()
                # success = False
                # while not success:
                #     success = True
                #     try:
                #         mx, my = map(int,input().split(','))
                #         self.chess_max[mx,my] = 1
                #         self.last.append((mx,my))
                #     except:
                #         success = False
                self.drawboard()
            else:
                hint, sc = self.play_min()
                if hint is None: break
                mx,my = hint
                self.chess_min[mx,my] = 1
                self.last.append((mx,my))
                print("Min moved at {0},{1}".format(mx,my))
                print("Score:",sc)
                self.draw()
        if turn & 1: print("Win")
        elif len(self.last) == m*n: print("Draw")
        else: print("Lose")

    def play_max(self):
        m,n = self.board.shape
        best = None
        maxi = -9999
        for i in range(m):
            for j in range(n):
                if self.chess_max[i,j] == self.chess_min[i,j]:
                    self.chess_max[i,j] = 1
                    sign = self.sign()
                    score = self.strategy[sign]
                    if score > maxi:
                        best = (i,j)
                        maxi = score
                    self.chess_max[i,j] = 0
        return best, maxi

    def play_min(self):
        m,n = self.board.shape
        best = None
        mini = 9999
        for i in range(m):
            for j in range(n):
                if self.chess_max[i,j] == self.chess_min[i,j]:
                    self.chess_min[i,j] = 1
                    sign = self.sign()
                    score = self.strategy[sign]
                    if score < mini:
                        best = (i,j)
                        mini = score
                    self.chess_min[i,j] = 0
        return best, mini

    def initialize_game(self,m,n,k):
        self.board = np.zeros((m,n),dtype=np.int8)
        self.k = k
        for i in range(m):
            for j in range(n):
                self.board[i,j] = (i+j)&1
        self.chess_max = np.zeros_like(self.board,dtype=np.int8)
        self.chess_min = np.zeros_like(self.board,dtype=np.int8)
        self.last = []
        self.strategy = {}

    def drawboard(self,hint=None):
        plt.imshow(self.board.T)
        plt.scatter([m[0] for i,m in enumerate(self.last) if i&1==0],
                    [m[1] for i,m in enumerate(self.last) if i&1==0],
                    300,marker='X',c='black')
        plt.scatter([m[0] for i,m in enumerate(self.last) if i&1],
                    [m[1] for i,m in enumerate(self.last) if i&1],
                    300,marker='o',c='black')
        if self.last: plt.scatter(*self.last[-1],300,marker='X' if len(self.last)&1==1 else 'o',c='red')
        if hint is not None: plt.scatter(*hint,300,marker='X',c='blue')
        plt.show()

    def draw(self):
        print((self.chess_max-self.chess_min).T)
        print('('+','.join(map(str,(self.chess_max-self.chess_min).reshape(-1)))+')')

    def min(self,alpha,beta):
        self.at = self.chess_min
        sign = self.sign()
        if sign in self.strategy: return self.strategy[sign]
        if self.is_terminal(self.chess_max):
            ans = 999
            self.strategy[sign] = ans
            return ans
        m,n = self.board.shape
        ans = None
        for i,j in product(range(m),range(n)):
            if self.chess_max[i,j] == self.chess_min[i,j]:
                self.chess_min[i,j] = 1
                self.last.append((i,j))
                comp = self.max(alpha,beta)
                # comp = min(comp-1,comp+1,key=abs)
                ans = min(ans, comp) if ans is not None else comp
                self.chess_min[i,j] = 0
                self.last.pop()
                if alpha is not None and ans <= alpha: break
                if beta is not None: beta = min(beta,ans)
        ans1 = ans if ans is not None else 0
        self.strategy[sign] = ans1
        return ans1

    def max(self,alpha=None,beta=None):
        self.at = self.chess_max
        sign = self.sign()
        if sign in self.strategy: return self.strategy[sign]
        if self.is_terminal(self.chess_min):
            ans = -999
            self.strategy[sign] = ans
            return ans
        m,n = self.board.shape
        ans = None
        for i,j in product(range(m),range(n)):
            if self.chess_max[i,j] == self.chess_min[i,j]:
                self.chess_max[i,j] = 1
                self.last.append((i,j))
                comp = self.min(alpha,beta)
                # comp = min(comp-1,comp+1,key=abs)
                ans = max(ans, comp) if ans is not None else comp
                self.chess_max[i,j] = 0
                self.last.pop()
                if beta is not None and ans >= beta: break
                if alpha is not None: alpha = max(alpha,ans)
        ans1 = ans if ans is not None else 0
        self.strategy[sign] = ans1
        return ans1

    def sign(self):
        situ = self.chess_max-self.chess_min
        return tuple(*situ.reshape(1,-1))

    def is_terminal(self,chess):
        if not self.last: return False
        x,y = self.last[-1]
        ans = 0
        h,v,d,u = 0,0,0,0
        m,n = self.board.shape
        for i in range(-self.k+1,self.k):
            if 0<=x+i<m:
                h1 = h+chess[x+i,y]
                h = (h1-h)*h1
            if 0<=y+i<n:
                v1 = v+chess[x,y+i]
                v = (v1-v)*v1
            if 0<=x+i<m and 0<=y+i<n:
                d1 = d+chess[x+i,y+i]
                d = (d1-d)*d1
            if 0<=x-i<m and 0<=y+i<n:
                u1 = u+chess[x-i,y+i]
                u = (u1-u)*u1
            ans = max(ans,h,v,d,u)
        # print(self.chess_max-self.chess_min)
        # print(ans)
        return ans >= self.k

if __name__ == '__main__':
    g = Game(True)
    # g.initialize_game(3,3,3)
    # # (1,1,-1,-1,-1,1,1,0,0)
    # g.chess_max = np.array([[1,1,0],[0,0,1],[1,0,0]],dtype=np.int8)
    # g.chess_min = np.array([[0,0,1],[1,1,0],[0,0,0]],dtype=np.int8)
    # g.last = [(1,2)]
    # print(g.checkTerminal(g.chess_max))
    g.play(4,4,4)