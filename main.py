
"""Sat based solver for the RCPSP problems (see rcpsp.proto)."""

import collections



from ortools.sat.python import cp_model


import pandas as pd
from datetime import date
from datetime import datetime
import numpy as np



class prob:
    def __init__(self, tasks, resources):
        self.tasks = tasks
        self.resources = resources

    def display_components(self):
        print("Tasks:")
        for task in self.tasks:
            print(vars(task))
        print("Resources:")
        for resource in self.resources:
            print(vars(resource))

class task:
    def __init__(self, order, ESD, duration, demands, resources):
         self.order = order
         self.ESD = ESD
         self.duration = duration
         self.demands = demands
         self.resources = resources

         
class resources_object:
    def __init__(self, max_capacity, renewable, idd):
      self.max_capacity = max_capacity
      self.renewable = renewable   
      self.idd = idd
    



class VarArrayAndObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):
  """Print intermediate solutions."""

  def __init__(self, variables):
    cp_model.CpSolverSolutionCallback.__init__(self)
    self.__variables = variables
    self.__solution_count = 0

  def on_solution_callback(self):
    print('Solution %i' % self.__solution_count)
    print('  objective value = %i' % self.ObjectiveValue())
    print(self.__variables)
    for v in self.__variables:
      print('  %s = %i' % (v, self.Value(v)), end=' ')
    self.__solution_count += 1

  def solution_count(self):
    return self.__solution_count






def preproc():
    print("prepross start")
    df = pd.read_excel(r'C:\Users\L1061703\TotalEnergies\D-LAB - Documents\2_Notebooks\19_Scheduling_Optimizer\ortools adam\dataor\pre_ou\Order Data Gorm.xlsx')
    df2 = pd.read_csv(r'C:\Users\L1061703\TotalEnergies\D-LAB - Documents\2_Notebooks\19_Scheduling_Optimizer\ortools adam\dataor\pre_ou\Dan WorkCenters Capacity.csv', sep = ";")

    ### resssources
    
    WC = []
    CAPA = []
    dfcapa = df2[["Work Center","Capacity (Hrs)"]]
    WC = dfcapa["Work Center"] 
    CAPA = dfcapa["Capacity (Hrs)"] 
    list_ressources = []
    iidd = 0
    for i in range(len(WC)):
        if WC[i] == "MTN-MECH" or WC[i] == "MTN-RIGG" or WC[i] == "MTN-CRAN" or WC[i] == "MTN-INST" or WC[i] == "MTN-ELEC" or WC[i] == "MTN-SCAF" or WC[i] == "MTN-ROUS" or WC[i] == "MTN-TELE" or WC[i] == "MTN-TURB" or WC[i] == "MTN-ROPE" or WC[i] == "MTN-PAIN" or WC[i] == "MTN-SAT" or WC[i] == "MTN-LAGG" or WC[i] == "PRODTECH" or WC[i] == "QAQCELEC"  or WC[i] == "VEN-MECH"  or WC[i] == "VEN-HVAC"  or WC[i] == "VEN-CRAN"  :
            iidd += 1
            r = resources_object(CAPA[i]+5, True, iidd)
            list_ressources.append(r)
    
    
                
    ### date 
    ESD = []
    duration = [] #demands//max_capacity
    demands = [] # <max capa
    rounded_demands =[]
    resources = []
    order = [] # for task name
    rescap = 0  #ressouces cap
    dfdate = df[["Work center","Order","Work","Earliest start date"]]
    WC = dfdate["Work center"] 
    ESD = dfdate["Earliest start date"]
    order = dfdate["Order"]
    demands = dfdate["Work"]         
    demands =list(demands)
    import math
    for d in demands:
        if pd.isna(d):
            d = 0.0
        if type(d) == float:
            rounded_demands.append(math.ceil(d))  
        else:
            rounded_demands.append(math.ceil(float(d.replace(",",".")))) # for all data
        

    demands = rounded_demands
    for i in range(len(WC)):

        if WC[i] == "MTN-MECH":
            resources.append(1)
            rescap = 8
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-RIGG":
            resources.append(2)
            rescap = 8
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-CRAN":
            resources.append(3)
            rescap = 4
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap

            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap  

            else:
                duration.append(1)
        if WC[i] == "MTN-INST":
            resources.append(4)
            rescap = 8
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap

            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap

            else:
                duration.append(1)
        if WC[i] == "MTN-ELEC":
            resources.append(5)
            rescap = 8
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap

            else:
                duration.append(1)
        if WC[i] == "MTN-SCAF":
            resources.append(6)
            rescap = 8
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-ROUS":
            resources.append(7)
            rescap = 24
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-TELE":
            resources.append(8)
            rescap = 8
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-TURB":
            resources.append(9)
            rescap = 8
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-ROPE":
            resources.append(10)
            rescap = 20
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-PAIN":
            rescap = 4
            resources.append(11)
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-SAT":
            rescap = 4
            resources.append(12)
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "MTN-LAGG":
            rescap = 20
            resources.append(13)
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "PRODTECH":
            rescap = 16
            resources.append(14)
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "VEN-CRAN":
            rescap = 8
            resources.append(15)
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "VEN-HVAC":
            rescap = 8
            resources.append(16)
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "VEN-MECH":
            rescap = 8
            resources.append(17)
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
        if WC[i] == "QAQCELEC":
            rescap = 18
            resources.append(18)
            if demands[i] > rescap and demands[i] > rescap%2 == 0:
                duration.append(demands[i]//rescap)
                demands[i] = rescap
            elif demands[i] > rescap and demands[i] > rescap%2 == 1:
                duration.append(1+demands[i]//rescap)
                demands[i] = rescap
            else:
                duration.append(1)
                
                
    latest = []
    earliest = []
    for k in ESD:
        if type(k) == str:
            y = datetime.strptime(str(k), '%d-%m-%Y').year
            m = datetime.strptime(str(k), '%d-%m-%Y').month
            d = datetime.strptime(str(k), '%d-%m-%Y').day
            latest.append(date(y, m, d))
        else:
            latest.append(date(k.year, k.month, k.day)) 
            
    for k in ESD:
        if type(k) == str:
            y = datetime.strptime(str(k), '%d-%m-%Y').year
            m = datetime.strptime(k, '%d-%m-%Y').month
            d = datetime.strptime(k, '%d-%m-%Y').day
            earliest.append(date(y, m, d)) 
        else:
            earliest.append(date(k.year, k.month, k.day)) 
        
                
    ## normalize date
    most_late = date(1,1,1)
    most_early = date(9999,12,12)
    for k in latest:
        delta = most_late - k
        if int(delta.days) < 0:
            most_late  = k
            

    for k in earliest:
        delta =  k-most_early
        if int(delta.days) < 0:
            most_early  = k

    delta =  most_late-most_early

    
    norESD = []

    for st in earliest:
        
        delt = st - most_early 
        norESD.append(delt.days)

    list_tasks = []    
    
    for i in range(len(norESD)):
        
        demands[i] = str(demands[i]).replace(",",".")
        demands[i] = demands[i]
        t = task(order[i],norESD[i], duration[i],demands[i], resources[i])
        list_tasks.append(t)


    print("prepross end")
    problem = prob(list_tasks, list_ressources)          
    return problem

def SolveRcpsp():
    """Parse and solve a given RCPSP problem in proto format."""

    prob = preproc()
    # Create the model.
    model = cp_model.CpModel()
    num_tasks = len(prob.tasks)
    num_resources = len(prob.resources)

    all_active_tasks = range(1, num_tasks - 1)
    all_resources = range(num_resources)

    horizon = 900 

    # Containers.
    task_starts = {}
    task_ends = {}
    task_durations = {}
    task_intervals = {}
    task_to_resource_demands = collections.defaultdict(list)
    task_to_presence_literals = collections.defaultdict(list)
    task_to_recipe_durations = collections.defaultdict(list)
    task_resource_to_fixed_demands = collections.defaultdict(dict)
    resource_to_sum_of_demand_max = collections.defaultdict(int)


    list_start = []
    list_end = []

    print("# Create task variables")
    for t in all_active_tasks:
        
        task = prob.tasks[t]
        num_recipes = 1
        all_recipes = range(num_recipes)

        start_var = model.NewIntVar(task.ESD, horizon, f'start_of_task_{t}')
        end_var = model.NewIntVar(task.ESD+task.duration, horizon, f'end_of_task_{t}')
        list_start.append(start_var)
        list_end.append(end_var)
        literals = []
        if num_recipes > 1:
            # Create one literal per recipe.
            literals = [
                model.NewBoolVar(f'is_present_{t}_{r}') for r in all_recipes
            ]

            # Exactly one recipe must be performed.
            model.Add(cp_model.LinearExpr.Sum(literals) == 1)

        else:
            literals = [1]

        # Temporary data structure to fill in 0 demands.
        demand_matrix = collections.defaultdict(int)

        # Scan recipes and build the demand matrix and the vector of durations.

        task_to_recipe_durations[t].append(task.duration)
        for demand, resource in zip(str(task.demands), str(task.resources)):
            demand_matrix[(int(resource), 0)] = int(demand)
        
        # Create the duration variable from the accumulated durations.
        duration_var = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(task_to_recipe_durations[t]),
            f'duration_of_task_{t}')

        # Link the recipe literals and the duration_var.

        model.Add(duration_var == task_to_recipe_durations[t][0]).OnlyEnforceIf(literals)

        # Create the interval of the task.
        task_interval = model.NewIntervalVar(start_var, duration_var, end_var,
                                             f'task_interval_{t}')

        # Store task variables.
        task_starts[t] = start_var
        task_ends[t] = end_var
        task_durations[t] = duration_var
        task_intervals[t] = task_interval
        task_to_presence_literals[t] = literals

        # Create the demand variable of the task for each resource.
        

        for resource in all_resources:

            demands = [
                demand_matrix[(resource, recipe)] for recipe in all_recipes
            ]

            task_resource_to_fixed_demands[(t, resource)] = demands
            demand_var = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(demands), f'demand_{t}_{resource}')
            task_to_resource_demands[t].append(demand_var)

            # Link the recipe literals and the demand_var.
            for r in all_recipes:
                model.Add(demand_var == demand_matrix[(resource,
                                                       r)]).OnlyEnforceIf(
                                                           literals[r])

            resource_to_sum_of_demand_max[resource] += max(demands)
        print(round(t/len(all_active_tasks)*100,2)," %")
    # Create makespan variable
    makespan = model.NewIntVar(0, horizon, 'makespan')
    makespan_size = model.NewIntVar(1, horizon, 'interval_makespan_size')
    interval_makespan = model.NewIntervalVar(makespan, makespan_size,
                                             model.NewConstant(horizon + 1),
                                             'interval_makespan')

    print("# Create resources")
    # Create resources.
    for r in all_resources:
        resource = prob.resources[r]
        c = resource.max_capacity

        intervals = [task_intervals[t] for t in all_active_tasks]

        demands = [task_to_resource_demands[t][r] for t in all_active_tasks]

        intervals.append(interval_makespan)
        demands.append(c)

        model.AddCumulative(intervals, demands, c)
        print(round(r/len(all_resources)*100,2)," %")
    # Objective.
    
            
    for t in all_active_tasks:
        model.Add(task_ends[t] <= makespan)
        
        
    objective = makespan
    #objective = makespan

    model.Minimize(objective)


    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    #solver.parameters.max_time_in_seconds = 30.0
    #solver.parameters.max_memory_in_mb = 10000
    solver.parameters.absolute_gap_limit = 1
    solver.parameters.random_seed = 42
    
    #solver.parameters.optimize_with_max_hs    = True
    solver.parameters.enumerate_all_solutions  = True
    #solver.parameters.interleave_search  = True #run when alone
    #solver.parameters.use_lns_only = True


    sol = [None]*(len(list_start) + len(list_end))
    sol[0::2] = list_start
    sol[1::2] = list_end
    solution_printer  = VarArrayAndObjectiveSolutionPrinter(sol)
    print("# Solving ...")
    status = solver.Solve(model, solution_printer)
    
    
    
    if status == cp_model.OPTIMAL  or status == cp_model.FEASIBLE:
      # Prints out the makespan and the start times and ranks of all tasks.
      print('Optimal cost: %i' % solver.ObjectiveValue())
      print('Makespan: %i' % solver.Value(makespan))
      print(solver.ResponseStats())


      
    elif status == cp_model.INFEASIBLE:
      print('Solver exited with infeasible status')
    else:
      print('Solver exited with nonoptimal status: %i' % status)
      
    print('Statistics')
    print('  - status          : %s' % solver.StatusName(status))
    print('  - conflicts       : %i' % solver.NumConflicts())
    print('  - branches        : %i' % solver.NumBranches())
    print('  - wall time       : %f s' % solver.WallTime())



def main():
    startime = datetime.now()
    SolveRcpsp()
    print("Total time : ", datetime.now() - startime)

if __name__ == '__main__':
    main()

    



    
##### DOCU
#
#
#https://acrogenesis.com/or-tools/documentation/user_manual/manual/metaheuristics/jobshop_lns.html
#https://sofdem.github.io/gccat/gccat/Ccumulative.html#:~:text=Cumulative%20scheduling%20constraint%20or%20scheduling%20under%20resource%20constraints.,that%20point%2C%20does%20not%20exceed%20a%20given%20limit.
#https://github.com/google/or-tools/blob/stable/ortools/sat/doc/scheduling.md#cumulative-constraint
#https://github.com/google/or-tools/tree/b37d9c786b69128f3505f15beca09e89bf078a89/examples/python
#https://github.com/google/or-tools/tree/stable/ortools/sat/samples
#https://github.com/google/or-tools/tree/stable/ortools/sat
#https://www.projectmanagement.ugent.be/research/project_scheduling/rcpsp
#https://www.om-db.wi.tum.de/psplib/data.html
#
