import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from time import time

class Game:
    def __init__(self,abp=False):
        '''
        Initialize the game
        '''
        self.board = None
        self.abp = abp # use ab prune?

    def play(self,m,n,k,mode):
        '''
        simulate a mnk game, set mode=0 for timing(not playing),=1 for automatic play,=2 for manual play
        '''
        self.initialize_game(m,n,k)
        self.drawboard() # show the board
        rec = time()
        if self.abp: self.max(-1000,1000)
        else: self.max()
        print("Time:",time()-rec)
        print("States:",len(self.strategy))
        if mode == 0: return
        turn = 0
        # simulate the game until terminal
        while not self.is_terminal(self.chess_max if turn & 1 else self.chess_min):
            turn += 1
            if turn & 1: # MAX's turn
                hint, sc = self.play_max() # get hint from strategy set
                if hint is None: break
                mx,my = hint
                self.drawboard((mx,my)) # show the board with the hint
                if mode == 1: # play automatically, play manually by set this True to False
                    self.chess_max[mx,my] = 1
                    self.last.append((mx,my))
                    print("Max moved at {0},{1}".format(mx,my))
                    print("Score:",sc)
                    self.draw() # show board after the move (text)
                else: self.is_valid() # manual input
                self.drawboard() # show board after the move (imshow)
            else: # MIN's turn
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
        

    def is_valid(self):
        '''
        valid input getter, input format: x,y
        '''
        success = False
        m,n = self.board.shape
        while not success: # do until a valid input is gotten
            success = True
            try:
                mx, my = map(int,input().split(','))
                if not 0<=mx<m or not 0<=my<n or self.chess_max[mx,my] != 0:
                    raise NotImplementedError
                self.chess_max[mx,my] = 1
                self.last.append((mx,my))
            except:
                success = False
        return mx,my

    def play_max(self):
        '''
        MAX's strategy producer
        '''
        m,n = self.board.shape
        best = None
        maxi = -9999
        for i,j in product(range(m),range(n)):
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
        for i,j in product(range(m),range(n)):
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
                self.board[i,j] = (i+j)&1 # make the board 'chess-board like'
        self.chess_max = np.zeros_like(self.board,dtype=np.int8) # initialize MAX's play
        self.chess_min = np.zeros_like(self.board,dtype=np.int8) # initialize MIN's play
        self.last = [] # last play
        self.strategy = {} # visited state

    def drawboard(self,hint=None):
        '''print the board use imshow'''
        plt.imshow(self.board.T)
        plt.scatter([m[0] for i,m in enumerate(self.last) if i&1==0],
                    [m[1] for i,m in enumerate(self.last) if i&1==0],
                    300,marker='X',c='black') # marker for MAX
        plt.scatter([m[0] for i,m in enumerate(self.last) if i&1],
                    [m[1] for i,m in enumerate(self.last) if i&1],
                    300,marker='o',c='black') # marker for MIN
        if self.last: plt.scatter(*self.last[-1],300,marker='X' if len(self.last)&1==1 else 'o',c='red') # last move
        if hint is not None: plt.scatter(*hint,300,marker='X',c='blue') # hint move
        plt.show()

    def draw(self):
        print((self.chess_max-self.chess_min).T) # print the board as text
        # print('('+','.join(map(str,(self.chess_max-self.chess_min).reshape(-1)))+')')

    def min(self,alpha,beta):
        self.at = self.chess_min
        sign = self.sign()
        if sign in self.strategy: return self.strategy[sign] # If situation seen, use it. 
        if self.is_terminal(self.chess_max): # MIN check the board, if MAX had won, MIN report a lose
            ans = 999 # MIN says: I have lost
            self.strategy[sign] = ans
            return ans
        m,n = self.board.shape
        ans = None
        for i,j in product(range(m),range(n)):
            if self.chess_max[i,j] == self.chess_min[i,j]: # check if can play at i,j
                self.chess_min[i,j] = 1 # play
                self.last.append((i,j))
                comp = self.max(alpha,beta)
                comp = min(comp-1,comp+1,key=abs)
                ans = min(ans, comp) if ans is not None else comp
                self.chess_min[i,j] = 0 # rollback
                self.last.pop()
                if alpha is not None and ans <= alpha: break
                if beta is not None: beta = min(beta,ans)
        ans1 = ans if ans is not None else 0
        self.strategy[sign] = ans1 # save this as seen situation
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
                comp = min(comp-1,comp+1,key=abs)
                ans = max(ans, comp) if ans is not None else comp
                self.chess_max[i,j] = 0
                self.last.pop()
                if beta is not None and ans >= beta: break
                if alpha is not None: alpha = max(alpha,ans)
        ans1 = ans if ans is not None else 0
        self.strategy[sign] = ans1
        return ans1

    def sign(self):
        '''from a game situation to a dictionary key'''
        situ = self.chess_max-self.chess_min
        return tuple(*situ.reshape(1,-1))

    def is_terminal(self,chess):
        if not self.last: return False
        x,y = self.last[-1] # checker centered at last move
        ans = 0
        h,v,d,u = 0,0,0,0
        m,n = self.board.shape
        for i in range(-self.k+1,self.k):
            if 0<=x+i<m: # horizontal
                h1 = h+chess[x+i,y]
                h = (h1-h)*h1
            if 0<=y+i<n: # vertical
                v1 = v+chess[x,y+i]
                v = (v1-v)*v1
            if 0<=x+i<m and 0<=y+i<n: # diagonal
                d1 = d+chess[x+i,y+i]
                d = (d1-d)*d1
            if 0<=x-i<m and 0<=y+i<n: # diagonal
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
    g.play(4,4,4,no=True)
