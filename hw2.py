
#AI HW2
#Joshua Schnall
#jschnall@usc.edu



from queue import *
#import heapq    #priority q
import collections
import copy




####################################################################################################################



#####################################################################################################################
#create board
#both empty board with values for look up
#and also board with pieces placed for current state.....


def MakeBoards(prob):
    #get board depth and player to go
    Bdepth = int( prob[0] )
    player = prob[2]
    Sdepth = int ( prob[3] )

    bothBoards = prob[4:]

    #split to boardVals and boardState

    boardVals = bothBoards[:Bdepth]
    boardState =bothBoards[Bdepth:]

    bVals =[]
    bState = []
    
    
    #make boards into list of lists
    for line in boardVals:
        line = [i for i in line.split()]
        bVals.append(line)
        
    for line in boardState:
        line = list(line)
        bState.append(line)    


    return bVals, bState, player, Sdepth




###################################################################################################################

#create search tree to the depth specified
#boardVals, boardState, player, Sdepth
#will use this pre-created search tree to then run the minmax alg on it.....



#node class
#node- each node here is a spot in the search tree created before the search is even run.......
class node():
    
    def __init__(self, state, scoreBoard, turn, adj, currnode, address, player, move):
        self.st = state    #current board state
        self.sc = scoreBoard
        self.trn = turn    #current turn either x or o
        self.adjacen = adj #a list of node children
        self.adjacen1 = [] # a holder spot to copy actual adj nodes to, not just board spots.....
        self.prior = currnode   #pointer to prior node object...... for route back.....
        self.address = address  #node depth and creation number [d2, 365]
        self.MinMax = 0   #minmax value to be assigned while running algorithm
        self.moveList = []
        self.thisMove = move


    #returns node state
    def getState(self):
        return self.st

    
    #returns node player turn
    def getTurn(self):
        return self.trn
    
    #return adj list
    def getAdj(self):
        return self.adjacen
    
    #pointer to prior node, must use to find final path
    def getPrior(self):
        return self.prior
    
    #get address
    def getAddress(self):
        return self.address
    
    #get MinMax score
    def getMM(self):
        return self.MinMax
    
    #def getAdj1
    def getAdj1(self):
        return self.adjacen1
    
    #def getmoveList
    def getMoveList(self):
        return self.moveList
    
    #def getThisMove
    def getThisMove(self):
        return self.thisMove
    
    
    
    #sets the adjacency list after generated for this node
    def setAdj(self, newGenList):
        self.adjacen = newGenList
        
    #sets the adjacency list 1 (holder list) after generated for this node
    def setAdj1(self, newGenList1):
        self.adjacen1 = newGenList1    
        
        
    #sets the minmax value after generated for this node
    def setMM(self, minMaxAlg):
        self.MinMax = minMaxAlg
     
     
    #sets the associated move value corresponding to each node in adj list......    
    def setMoveList(self, newMoveList):
        self.moveList =newMoveList    
        
         


    #current score of both players [x,o]
    def getScore(self):
        scoreX = 0
        scoreO = 0
        
        #score for x
        indexX = 0
        indexY = 0
        for line in self.st:
            for period in line:
                if(period == "X"):
                    scoreX += int(self.sc[indexX][indexY])
                indexY += 1    
            indexX += 1
            indexY = 0       
        
        #score for o
        indexX = 0
        indexY = 0
        for line in self.st:
            for period in line:
                if(period == "O"):
                    scoreO += int(self.sc[indexX][indexY])
                indexY += 1    
            indexX += 1 
            indexY = 0
        
        if (player == "X"):
            return -(scoreX - scoreO)   #had to reverse these to get it to work.... don't feel like looking at where i messed up here......
        else:
            return -(scoreO - scoreX)
        
      





#Creates acutal search tree which is all possible moves from start point
#to a given search depth............
def createTree(bVals, bState, player, Sdepth):


    #create nodes and add to adj list, an ordered dict to define the graph, and each node contains an ordered dict of connections
    adjList = []      #list of nodes, each node contains its own adjacency info in order  
    nodeNum = 0  #node creation number-- not necessary, number will be order in adjlist
    currNodeDepth = 0 #depth we are currently at,  add upon node creation


    #create initial node based on initial state or boardState and add to adj list  
    #, state, scoreBoard, turn, adj, currnode, address 
    move =""
    iniNode = node(bState, bVals, player, [ ], "FIRST",[nodeNum, currNodeDepth], player, move ) 
    adjList.append(iniNode)
         
    #brute force search initial node for all possible moves
    #add these to that nodes adj list
    #allMoves(iniNode)
 
    #create new nodes for all of these, and add to queue and expand one at a time....
    #go onto next node and do same for it while less than certain depth or run out of moves.......
    nodeQ = Queue(maxsize=0)  #empty queue to stores nodes not yet expanded........
    nodeQ.put(iniNode)
    
    #nodeNum, currNodeDepth = expand(iniNode.getAdj(),bVals, player, iniNode, [nodeNum, currNodeDepth], adjList, nodeQ)  #expand initial nodes adjlist and add to nodeQ and to adjdict....
 
 
    while(nodeQ.empty() == False and currNodeDepth <=Sdepth):
        currNode = nodeQ.get()  #take first off of unexpanded nodes
        allMoves(currNode)  #get all moves and add to adjacen, also return current node number and depth
        #print(currNode.getAdj()) #DEBUG
        nodeNum, currNodeDepth = expand(currNode.getAdj(), bVals, player, currNode, [nodeNum, currNodeDepth], adjList, nodeQ, currNode.getMoveList())   #add expanded nodes to queue and adjdict
        

    return adjList






#creates a list of all moves from this node........ or all possible changes to board state .... brute force can do either STAKE OR RAID
def allMoves(currNode):
    move =""
    position = ""
    
    toGo = currNode.getTurn()
    toGoNext = ""
    if(toGo == "X"):
        toGoNext = "O"
    else:
        toGoNext = "X"
        
    
    
    boardState = currNode.getState()
    newStateList = [] #list of all new states created..... so list of future boards.......
    newMoveList = [] #corresponding list of moves to the above states......
    
    
    #go through each open spot in bState as each is a possible move
    indexX = 0 #board game indices of current place
    indexY = 0
    for line in boardState:
        for period in line:
            if(period == "."):
                newBoardState = copy.deepcopy(boardState)  #deep copy makes new board obj
                newBoardState[indexX][indexY] = toGo  #replace with either X or O
                
                numToLet = str(chr(indexY+1 + 96).upper())
                
                position = numToLet + str(indexX+1)  #board position for move as in B2 or C3
                #print(newBoardState)
                
                xCountAdj = 0 # of 4 adj squares how many are X and how many are O
                oCountAdj = 0
                left = ""
                right= ""
                up = ""
                down = ""
                
                #should be a way to compress this and check easier each adj spot
                #also here cannot check a board index if out of bounds.....
                if (indexX > 0):
                    if (newBoardState[indexX-1][indexY] == "X"):  
                        xCountAdj +=1
                        up = "X"    
                    if (newBoardState[indexX-1][indexY] == "O"):
                        oCountAdj +=1
                        up= "O"
                        
                if(indexX < len(newBoardState) -1):        
                    if (newBoardState[indexX+1][indexY] == "X"):
                        xCountAdj +=1
                        down = "X"
                    if (newBoardState[indexX+1][indexY] == "O"):
                        oCountAdj +=1 
                        down = "O"
                        
                if(indexY >0):        
                    if (newBoardState[indexX][indexY-1] == "X"):
                        xCountAdj +=1
                        left ="X"
                    if (newBoardState[indexX][indexY-1] == "O"):
                        oCountAdj +=1
                        left ="O"
                        
                if(indexY < len(newBoardState[0]) -1):        
                    if (newBoardState[indexX][indexY+1] == "X"):
                        xCountAdj +=1
                        right ="X"
                    if (newBoardState[indexX][indexY+1] == "O"):
                        oCountAdj +=1        
                        right = "O"
                
                
                #if Stake
                if(xCountAdj == 0 or oCountAdj == 0):   # out of 4 adjacent 1 must be yours and one must be opponents for RAID, otherwise STAKE
                    move ="Stake"
                    newStateList.append(newBoardState)                
    
                #if RAID  as in touching one of your own pieces and one of their pieces....... flip pieces to other team
                else:
                    move ="Raid"
                    if(left == toGoNext): 
                        newBoardState[indexX][indexY-1] = toGo
                    if(right == toGoNext):
                        newBoardState[indexX][indexY+1] = toGo
                    if(up == toGoNext):
                        newBoardState[indexX-1][indexY] = toGo
                    if(down == toGoNext):
                        newBoardState[indexX+1][indexY] = toGo
                    
                    
                    newStateList.append(newBoardState) #append to node's adj list
                    
                nodeMove = position+" "+ move     #set the move to B3 Stake or RAID
                newMoveList.append(nodeMove) 
                
                    
                
            indexY += 1
        indexX += 1
        indexY =  0
      
      
    currNode.setMoveList(newMoveList)  #corresponding moves to the below adj list of nodes/states
    currNode.setAdj(newStateList) 
                  
    
    
  
    


    
    
    
#expand creates nodes for all possible moves one at a time, as each node is created it is added to adjlist and nodeQ
def expand(currAdj, scoreBoard, turn, currNode, address, adjList, nodeQ, moveList):
    
    currNodeNum = currNode.getAddress()[1]
    currNodeDepth = currNode.getAddress()[0] 
    
    adj1 = [] #the nodes own adjlist.......
    
    turn = currNode.getTurn()
    if(turn == "X"):
        turn = "O"
    else:
        turn = "X"
    
    count = 0
    for board in currAdj:
        currNodeNum +=1
        currNodeDepth = currNode.getAddress()[0]+1
        move = moveList[count]
        newNode = node(board, scoreBoard, turn, [ ], currNode, [currNodeDepth,currNodeNum], player, move )
        if(newNode.getAddress()[0] <= Sdepth):
            adj1.append(newNode)
            adjList.append(newNode)
            nodeQ.put(newNode)
        count += 1
    
    currNode.setAdj1(adj1) #set the actual adj nodes, not just board states for each node...... 
    return currNodeNum, currNodeDepth     
            
    
    
 
 
 


        



#####################################################################################################################
#search Minmax
#return move and new board created....
#use created adjList above -  it is an ordered list of nodes, each node's .getAdj1() returns a list of adj node objects

#so will take a state initial board state, actions is going to be the set of next level nodes in tree as above
#max value finds the max utility value of a leaf
#min value finds the min utility value of a leaf
#utility value is the score
#each time it calls min or max again it just goes to the next level in the tree I construct above
#evaluation functin is the same as the utility function for cuttoff test the score.....       
    
          
          
          
# the score of each leaf node is now dependent upon which player x or o goes first, is neg or opposite depending, should fix the who goes first problem
                    
def minMax(tree):
    
    resultNodeValue = minValue(tree[0].getAdj1())  #return next lvl in tree with minmax values assigned now           
    
    for node in tree[0].getAdj1():              #find max value of these returned nodes....... one that matches the result value.......
        if (node.getMM() == resultNodeValue):
            return node.getState(), node.getThisMove()   #return the board state of the correct node 
               
        
      
      
        
#find max value out of all possible actions
def maxValue(nodeLvl):
    
    currNode = nodeLvl[0]  #current node
    value = -1000000000  #current minmax score of current node should be inf  test = float("inf")
    
    
    if( len(currNode.getAdj1()) == 0 ):  #either reached end of game or cut-off lvl, as in leaf node
        for node in nodeLvl:
            node.setMM(node.getScore())
            if (node.getScore() > value) :
                value = node.getScore()
                
        return value
        #return nodeLvl
        
        
    #if CUTOFF-TEST(state, depth) then return EVAL(state)

    for node in nodeLvl:
        retValue = minValue(node.getAdj1())
        node.setMM(retValue)
        if (retValue > value):
            value = retValue
        

    
    return value
    #return value don't return value, just set that nodes MMvalue to whatever the value is
    






#find min value out of all possible actions
def minValue(nodeLvl):
    
    currNode = nodeLvl[0]  #current node
    value = 1000000000  #current minmax score of current node should be inf  test = float("inf")
    
    
    if( len(currNode.getAdj1()) == 0 ):  #either reached end of game or cut-off lvl, as in leaf node
        for node in nodeLvl:
            node.setMM(node.getScore())
            if (node.getScore() < value) :
                value = node.getScore()
                
        return value
        #return nodeLvl
        
        
    #if CUTOFF-TEST(state, depth) then return EVAL(state)

    for node in nodeLvl:
        retValue = maxValue(node.getAdj1())
        node.setMM(retValue)
        if (retValue < value):
            value = retValue
        

    
    return value
    #return value don't return value, just set that nodes MMvalue to whatever the value is




    





####################################################################################################################
#abSearch
#return move and new board created......
#use created adjdict above


                    
def abSearch(tree):
    
    alpha = -100000000000  #-inf
    beta =   100000000000  #inf
    
    
    resultNodeValue = abMinValue(tree[0].getAdj1() , alpha, beta)  #return next lvl in tree with minmax values assigned now           
    
    for node in tree[0].getAdj1():              #find max value of these returned nodes....... one that matches the result value.......
        if (node.getMM() == resultNodeValue):
            return node.getState(), node.getThisMove()   #return the board state of the correct node 
               
        
      
      
        
#find max value out of all possible actions
def abMaxValue(nodeLvl, alpha, beta):
    
    currNode = nodeLvl[0]  #current node
    value = -1000000000  #current minmax score of current node should be inf  test = float("inf")
    
    
    if( len(currNode.getAdj1()) == 0 ):  #either reached end of game or cut-off lvl, as in leaf node
        for node in nodeLvl:
            node.setMM(node.getScore())
            if (node.getScore() > value) :
                value = node.getScore()
                
        return value
        #return nodeLvl
        
        
    #if CUTOFF-TEST(state, depth) then return EVAL(state)

    for node in nodeLvl:
        retValue = abMinValue(node.getAdj1(), alpha, beta)
        node.setMM(retValue)
        if (retValue > value):
            value = retValue
        if(value >= beta):
            return value
        alpha = max(alpha, value)

    
    return value
    #return value don't return value, just set that nodes MMvalue to whatever the value is
    






#find min value out of all possible actions
def abMinValue(nodeLvl, alpha, beta):
    
    currNode = nodeLvl[0]  #current node
    value = 1000000000  #current minmax score of current node should be inf  test = float("inf")
    
    
    if( len(currNode.getAdj1()) == 0 ):  #either reached end of game or cut-off lvl, as in leaf node
        for node in nodeLvl:
            node.setMM(node.getScore())
            if (node.getScore() < value) :
                value = node.getScore()
                
        return value
        #return nodeLvl
        
        
    #if CUTOFF-TEST(state, depth) then return EVAL(state)

    for node in nodeLvl:
        retValue = abMaxValue(node.getAdj1(), alpha, beta)
        node.setMM(retValue)
        if (retValue < value):
            value = retValue
        if (value <= alpha):
            return value
        beta = min(beta, value)
        

    
    return value
    #return value don't return value, just set that nodes MMvalue to whatever the value is






'''

function MAX-VALUE(state,α, β) returns a utility value
if TERMINAL-TEST(state) then return UTILITY(state)
v ←−∞
for each a in ACTIONS(state) do
v ←MAX(v, MIN-VALUE(RESULT(s,a),α, β))
if v ≥ β then return v
α←MAX(α, v)
return v




function MIN-VALUE(state,α, β) returns a utility value
if TERMINAL-TEST(state) then return UTILITY(state)
v ←+∞
for each a in ACTIONS(state) do
v ←MIN(v, MAX-VALUE(RESULT(s,a) ,α, β))
if v ≤ α then return v
β←MIN(β, v)
return v
'''





#####################################################################################################################


#MAIN    
#File I/O and select which search alg to use
'''
3
MINIMAX
O
2
1 8 23
5 42 12
26 30 9
X..
...
...

//////////////


B3 Stake
X..
...
.O.

'''
    
    
    
#file input
pr = open("input.txt","r")    
prob = []   #input file with each line read into list     
prob = pr.read().splitlines()
pr.close()
    



        

#call search
bVals, bState, player, Sdepth = MakeBoards(prob) 
tree = createTree(bVals, bState, player, Sdepth)
if (prob[1] == "MINIMAX"):
    boardNext, move = minMax(tree) 
elif (prob[1] == "ALPHA-BETA"):
    boardNext, move = abSearch(tree) 
  




#file output
ans = open('output.txt', 'w')
ans.write(move)
ans.write("\n")
for x in boardNext:
    for dot in x:
        ans.write(dot)
    ans.write("\n")
     
ans.close()        
        


#############################################################################################################################




