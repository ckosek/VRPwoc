# Traveling Salesman Problem - VRP
# For Artificial Intelligence
# Project 6 by Chase Kosek
import random
import copy
import os
import time
import math
import csv
import numpy as np
from tkinter import *

# global vars for holding values used throughout
list_of_cities = []
global_truck_list= []
best_route = []
best_route_length = 0
# probability that an individual Route will mutate
mut_rate = 0.4
# Number of generations to run for
gens = 100
# Population size of 1 generation (PathPopulation) 
gen_pop_size = 100
# Size of tournament pool
tournament_size = 7
# Whether WOC is in play or not
WOC = False
# Prioritizes packages
PriotrityPackages = False
# Percentage of WOC used for expert creation
WisdomPercent = 0.3
# Number of Trucks
trucks = 3


# Node class
class Node(object):
    # Stores city nodes and initializes
    def __init__(self, name, x, y, DistanceDict=None):
        self.name = name
        self.x = self.graph_x = x
        self.y = self.graph_y = y
        # Appends Node to list of cities
        list_of_cities.append(self)
        # Creates the distance dictionary
        self.DistanceDict = {self.name:0.0}
        if DistanceDict:
            self.DistanceDict = DistanceDict

    # Used to create Distance dict
    def calculate_distances(self): 
        for node in list_of_cities:
            tmp_dist = self.distance(self.x, self.y, node.x, node.y)
            self.DistanceDict[node.name] = tmp_dist

    def distance(self,x1,y1,x2,y2):
        return math.sqrt( ((x2-x1)**2)+((y2-y1)**2))


# Connection Class
class Connection(object):
    # Stores list of Node objs and path info
    def __init__(self):
        # Creates a path attribute equal to a randomly shuffled global_truck_list and gets its length
        self.path = sorted(global_truck_list, key=lambda *args: random.random())
        self.ConnectionLength()

    # Calculates connection length from nodes
    def ConnectionLength(self):
        self.length = 0
        for city in self.path:
            next_city = self.path[self.path.index(city)-len(self.path) + 1]
            # Uses the first city's DistanceDict attribute to find the distance to the next and adds to length
            dist_to_next = city.DistanceDict[next_city.name]
            self.length += dist_to_next

    def PrintPath(self):
        # Saves city in array and prints path
        PathDict = []
        for city in self.path:
            PathDict.append(city.name)
        print(PathDict)

    def GetPath(self):
        # Saves city in array and gets path
        PathDict = []
        for city in self.path:
            PathDict.append(city.name)
        return PathDict



# Contains a population of Connection() objects
class PathPopulation(object):
    def __init__(self, size, initialise):
        self.rt_pop = []
        self.size = size
        # If we want to initialise a population.rt_pop:
        if initialise:
            for n in range(0,size):
                new_rt = Connection()
                self.rt_pop.append(new_rt)
            self.MinInPop()

    def MinInPop(self):
        # sorts the list based on the path length
        sorted_list = sorted(self.rt_pop, key=lambda x: x.length, reverse=False)
        self.fittest = sorted_list[0]
        return self.fittest


# Class for bringing together all of the methods to do with the Genetic Algorithm
class GA(object):
    # Returns child path from crossing two parents
    def Crossover(self, parent1, parent2):
        child_rt = Connection()
        for x in range(0,len(child_rt.path)):
            child_rt.path[x] = None

        # Two random integer indices of the first parent
        start_pos = random.randint(0,len(parent1.path))
        end_pos = random.randint(0,len(parent1.path))


        # Takes the subpath from parent one and sticks it in itself
        if start_pos < end_pos:
            for x in range(start_pos,end_pos):
                child_rt.path[x] = parent1.path[x]
        # if the start position is after the end:
        elif start_pos > end_pos:
            for i in range(end_pos,start_pos):
                child_rt.path[i] = parent1.path[i]


        # Cycles through the parent2. And fills in the child path
        for i in range(len(parent2.path)):
            # If child doesn't have a node parent2 does it puts it in the first None spot
            if not parent2.path[i] in child_rt.path:
                for x in range(len(child_rt.path)):
                    if child_rt.path[x] == None:
                        child_rt.path[x] = parent2.path[i]
                        break
        # Repeats until all the cities are in the child path
        child_rt.ConnectionLength()
        return child_rt
    


    def Mutate(self, route_to_mut):
        # Swaps two random nodes in path to mutate
        if random.random() < mut_rate:

            # Two random nodes
            mut_pos1 = random.randint(0,len(route_to_mut.path)-1)
            mut_pos2 = random.randint(0,len(route_to_mut.path)-1)

            # Swap them
            city1 = route_to_mut.path[mut_pos1]
            city2 = route_to_mut.path[mut_pos2]

            route_to_mut.path[mut_pos2] = city1
            route_to_mut.path[mut_pos1] = city2

        # Recalculate the length of the path (updates it's .length)
        route_to_mut.ConnectionLength()
        return route_to_mut

    def Tournament(self, population):
        # Randomly selects fittest from smaller #
        tournament_pop = PathPopulation(size=tournament_size,initialise=False)

        # Fills it with random individuals but can have same one twice
        for n in range(tournament_size-1):
            tournament_pop.rt_pop.append(random.choice(population.rt_pop))
        
        # returns the fittest of the tournament
        return tournament_pop.MinInPop()

    def PopEvolve(self, init_pop):
        # Makes a new population and evolve it returning new pop
        descendant_pop = PathPopulation(size=init_pop.size, initialise=True)
        # amount of best paths which our expert is based on
        Experts = int(gen_pop_size*WisdomPercent)
        # Creates Adj table
        AdjTable = {}
        WOCOffset = 0
        # if we have WOC as true
        if WOC:
            # Sorted best to worst in a gen
            sorted_list = sorted(init_pop.rt_pop, key=lambda x: x.length, reverse=False)
            WOCOffset = 1
            # Removes paths not used for WOC
            del sorted_list[Experts:]
            for route in sorted_list:
                for city in route.path:
                    # Idenitifies connects in path accounting for x,y vs. y,x            
                    nextCity = route.path[route.path.index(city)-len(route.path)+1]
                    connect = str(city.name) + ',' + str(nextCity.name)
                    RevConnect = str(nextCity.name) + ',' + str(city.name)
                    # Counter for number of instances
                    if connect in AdjTable:
                        AdjTable[connect] += 1
                    elif RevConnect in AdjTable:
                        AdjTable[RevConnect] += 1
                    else:
                        AdjTable[connect] = 1
            # Creates Adjacency table
            AdjTable = sorted(AdjTable.items(), key = lambda kv:(kv[1], kv[0]))
            ArrayAdjTable = []
            # Empties most used edges into array to insert nodes
            while AdjTable:
                ArrayAdjTable.append((AdjTable.pop()[0]).split(','))
            WisdomPath = Connection()
            # Loop to insert a complete list from the WOC data
            while len(WisdomPath.path) != len(global_truck_list):
                for x in ArrayAdjTable:
                    cityA = global_truck_list[int(x[0])-1]
                    cityB = global_truck_list[int(x[1])-1]
                    # if and elifs for adj table insertion conditions
                    if(cityA not in WisdomPath.path and cityB == WisdomPath.path[0]):
                        WisdomPath.path.insert(0, cityA)
                    elif(cityA not in WisdomPath.path and cityB == WisdomPath.path[len(WisdomPath.path)-1]):
                        WisdomPath.path.insert(len(WisdomPath.path), cityA)
                    elif(cityB not in WisdomPath.path and cityA == WisdomPath.path[0]):
                        WisdomPath.path.insert(0, cityB)
                    elif(cityB not in WisdomPath.path and cityA == WisdomPath.path[len(WisdomPath.path)-1]):
                        WisdomPath.path.insert(len(WisdomPath.path), cityB)
                # Fills rest
                for city in global_truck_list:
                    if city not in WisdomPath.path:
                        # Greedy selection for remaining cities
                        x1 = city.x
                        y1 = city.y
                        x2 = WisdomPath.path[0].x
                        y2 = WisdomPath.path[0].y
                        x3 = WisdomPath.path[len(WisdomPath.path)-1].x
                        y3 = WisdomPath.path[len(WisdomPath.path)-1].x
                        beginingappend = math.sqrt( ((x2-x1)**2)+((y2-y1)**2))
                        endappend = math.sqrt( ((x3-x1)**2)+((y3-y1)**2))
                        # Chooses tail or head for city insertion
                        if endappend < beginingappend:
                            WisdomPath.path.append(city)
                        else:
                            WisdomPath.path.insert(0, city)
                        break
            #Inserts WOC child into path population for next gen
            for x in range(WOCOffset):
                descendant_pop.rt_pop[x] = WisdomPath
                
        # Goes through the new population and fills it with the child of two tournament winners from the previous population
        for x in range(WOCOffset,descendant_pop.size):
            # two parents:
            tournament_parent1 = self.Tournament(init_pop)
            tournament_parent2 = self.Tournament(init_pop)

            # A child:
            tournament_child = self.Crossover(tournament_parent1, tournament_parent2)

            # Fill the population up with children
            descendant_pop.rt_pop[x] = tournament_child
        # Mutates all the routes (mutation with happen with a prob p = mut_rate)
        for path in descendant_pop.rt_pop:
            if random.random() < 0.3:
                self.Mutate(path)

        # Update the fittest path:
        descendant_pop.MinInPop()

        return descendant_pop



class Run(object):
    # Runs the GA
    def __init__(self,generations,pop_size):
        # initializes GA Run for # of gens and pop size
        self.generations = generations
        self.pop_size = pop_size
        self.GraphingCoords()
        # Initiates a window object fo Tkinter
        self.window = Tk()

        # Makes two canvas one for current and one for best
        self.canvas_current = Canvas(self.window, height=410, width=810)
        self.canvas_best = Canvas(self.window, height=410, width=810)

        # Circle creation
        for city in global_truck_list:
            self.canvas_current.create_oval(city.graph_x*8-5, city.graph_y*4-5, city.graph_x*8 + 5, city.graph_y*4 + 5, fill='yellow')
            self.canvas_best.create_oval(city.graph_x*8-5, city.graph_y*4-5, city.graph_x*8 + 5, city.graph_y*4 + 5, fill='yellow')

        self.canvas_current.pack()
        self.canvas_best.pack()

        self.GA_loop(self.generations, self.pop_size)
        # Loops the window to keep it open
        #self.window.after(0,self.GA_loop(self.generations, self.pop_size))
        #self.window.mainloop()


    def GraphingCoords(self):
        # Very high and low numbers to be overruled
        min_x = 999999999999999
        max_x = -999999999999999
        min_y = 999999999999999
        max_y = -999999999999999

        # Finds max and mins
        for city in list_of_cities:
            if city.x < min_x:
                min_x = city.x
            if city.x > max_x:
                max_x = city.x
            if city.y < min_y:
                min_y = city.y
            if city.y > max_y:
                max_y = city.y

        # Shifts graph
        for city in list_of_cities:
            city.graph_x = (city.graph_x + (-1*min_x)) +3
            city.graph_y = (city.graph_y + (-1*min_y)) +3

        # Resets Vars
        min_x = 999999999999999
        max_x = -999999999999999
        min_y = 999999999999999
        max_y = -999999999999999

        # Finds max and mins
        for city in list_of_cities:
            if city.graph_x < min_x:
                min_x = city.graph_x
            if city.graph_x > max_x:
                max_x = city.graph_x
            if city.graph_y < min_y:
                min_y = city.graph_y
            if city.graph_y > max_y:
                max_y = city.graph_y


    def GraphUpdate(self,the_canvas,the_route,color):
        # Deletes path items to update the paths
        the_canvas.delete('path')

        # Loops through the path
        for n in range(len(the_route.path)):
            # similar to i+1 but will loop around at the end
            next = n-len(the_route.path)+1
            # creates the line from city to city
            the_canvas.create_line(the_route.path[n].graph_x*8,
                                the_route.path[n].graph_y*4,
                                the_route.path[next].graph_x*8,
                                the_route.path[next].graph_y*4,
                                tags=("path"),
                                fill=color)

            # Packs and updates the canvas
            the_canvas.pack()
            the_canvas.update_idletasks()

    def GA_loop(self,generations,pop_size):
        # Loop for GA. Manages populations and variables
        start_time = time.time()
        pop = PathPopulation(pop_size, True)
        global best_route
        global best_route_length
        # Gets the best length from the first pop
        initial_length = pop.fittest.length

        # Creates a random path called best_path. It will store our overall best path.
        best_path = Connection()

        # Update the two canvases with the just-created path on bottom graph
        self.GraphUpdate(self.canvas_current,pop.fittest,'red')
        self.GraphUpdate(self.canvas_best,best_path,'blue')


        for x in range(1,generations):
            # Updates the current canvas
            self.GraphUpdate(self.canvas_current,pop.fittest,'red')

            # Evolves the population:
            pop = GA().PopEvolve(pop)

            # Saves shorter path to best orute
            if pop.fittest.length < best_path.length:
                # set the path (copy.deepcopy because pop.fittest is persistent in this loop so will cause reference bugs)
                best_path = copy.deepcopy(pop.fittest)
                # Update the second canvas because we have a new best path:
                self.GraphUpdate(self.canvas_best,best_path,'blue')

            print('Generation',x,'of',generations,'\n')
            print('Overall best path length',best_path.length)
            print('Path (top graph):')
            best_path.PrintPath()
            print('\nCurrent path length',pop.fittest.length)
            print('Path (bottom graph):')
            pop.fittest.PrintPath()

        # Updates the best path canvas for the last time:
        self.GraphUpdate(self.canvas_best,best_path,'blue')
            
        end_time = time.time()
        run_time = end_time - start_time

        # Prints final output to terminal:
        print('\n=====Completed',generations,'generations=====')
        print('Runtime was',run_time,'seconds\n')
        print('Initial best path length:',initial_length)
        print('Final best path length:',best_path.length)
        print('The best path was:')
        best_path.PrintPath()
        best_route.append(pop.fittest.GetPath())
        best_route_length = best_route_length + best_path.length


def GeneticTSP():
    for city in list_of_cities:
        city.calculate_distances()
    Run(generations=gens,pop_size=gen_pop_size)


def distance(x1,x2,y1,y2):
    return math.sqrt( ((x2-x1)**2)+((y2-y1)**2))

def VRPGUI(TSPDict, Path, color,canvas):
    # Sets up Canvas for all vehicle routes
    canvas.pack() 
    # Draws lines between all node connections
    node = 1
    while node < len(Path):
        canvas.create_line(TSPDict.get(int(Path[node-1]))[0]*8, 
                    TSPDict.get(int(Path[node-1]))[1]*4, 
                    TSPDict.get(int(Path[node]))[0]*8, 
                    TSPDict.get(int(Path[node]))[1]*4,
                    width = 5,
                    fill = color)
        # Moves to next node
        node += 1

    # Creates return line
    canvas.create_line(TSPDict.get(int(Path[0]))[0]*8, 
                    TSPDict.get(int(Path[0]))[1]*4, 
                    TSPDict.get(int(Path[len(Path)-1]))[0]*8, 
                    TSPDict.get(int(Path[len(Path)-1]))[1]*4,
                    width = 5,
                    fill = color)

    # Creates Node Circles with Label n inside them
    for n in TSPDict:
        x = TSPDict.get(n)[0]
        y = TSPDict.get(n)[1]
        canvas.create_oval(x*8 - 5, y*4 - 5, x*8 + 5, y*4 + 5, outline='blue', fill='yellow')

if __name__ == '__main__':
    file = input("Enter your file path: ") 
    start_time = time.time()
    with open(file) as f:
        for _ in range(7):
                next(f)
        for line in f:
            (city, x, y) = line.split()
            Node(str(city), float(x), float(y))
    for city in list_of_cities:
        city.calculate_distances()
    if PriotrityPackages is True:
        random.shuffle(list_of_cities)
    else:
        # Sorts cities based on coordinates
        list_of_cities.sort(key=lambda x: (x.x,x.y), reverse=True)
    split = np.array_split(list_of_cities, trucks)
    truckpath_dict= {}
    # Declares depot mode and then inserts in split arrays
    depot = Node(str(9999), float(50), float(50))
    for city in list_of_cities:
        city.calculate_distances()
    x = 0
    for array in split:
        array = np.append(array,depot)
        truckpath_dict[x] = array
        x = x+1

    truck_routes = []
    # Runs algorithm for each truckpath set
    for x in truckpath_dict:
        global_truck_list = truckpath_dict[x]
        Run(generations=gens,pop_size=gen_pop_size)
    
    # Used to create Distance Dict for VRP GUI
    TSPfile = {}
    TSPDistanceDict = {}
    with open(file) as f:
        for _ in range(7):
            next(f)
        for line in f:
            (key, x, y) = line.split()
            TSPfile[int(key)] = [float(x),float(y)]
    TSPfile[int(9999)] = [float(50),float(50)]
    master = Tk() 
    canvas = Canvas(master, height=500, width=1000) 
    x = 1
    y = 1
    print('\n\n\n\n=====Complete Vehicle Paths=====\n')
    # Will uniquely color for up to 6 vehicle paths
    for path in best_route:
        print('Path',y,':',path)
        
        if x == 1:
            color = 'red'
        elif x == 2:
            color = 'blue'
        elif x == 3:
            color = 'green'
        elif x == 4:
            color = 'pink'
        elif x == 5:
            color = 'black'
        else:
            color = 'purple'
            x = 0
        VRPGUI(TSPfile,path,color,canvas)
        x = x+1
        y = y+1
    end_time = time.time()
    run_time = end_time - start_time
    print('Total length of vehicle paths:',best_route_length)
    print('Total Runtime:',run_time)
    mainloop()