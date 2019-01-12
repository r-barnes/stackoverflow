#!/usr/bin/env python3

import cvxpy as cp
import csv
from io import StringIO
import random

#CSV file for food data
#Drawn from: https://think.cs.vt.edu/corgis/csv/food/food.html
food_data = """1st Household Weight,1st Household Weight Description,2nd Household Weight,2nd Household Weight Description,Alpha Carotene,Ash,Beta Carotene,Beta Cryptoxanthin,Calcium,Carbohydrate,Category,Cholesterol,Choline,Copper,Description,Fiber,Iron,Kilocalories,Lutein and Zeaxanthin,Lycopene,Magnesium,Manganese,Monosaturated Fat,Niacin,Nutrient Data Bank Number,Pantothenic Acid,Phosphorus,Polysaturated Fat,Potassium,Protein,Refuse Percentage,Retinol,Riboflavin,Saturated Fat,Selenium,Sodium,Sugar Total,Thiamin,Total Lipid,Vitamin A - IU,Vitamin A - RAE,Vitamin B12,Vitamin B6,Vitamin C,Vitamin E,Vitamin K,Water,Zinc
28.35,1 oz,454,1 lb,0,0.75,0,0,13,0.0,LAMB,68,0,0.12,"LAMB,AUS,IMP,FRSH,RIB,LN&FAT,1/8""FAT,RAW",0.0,1.34,289,0,0,18,0.009,9.765,5.103,17314,0.501,156,0.985,254,16.46,26,0,0.232,11.925,6.9,68,0.0,0.145,24.2,0,0,1.62,0.347,0.0,0.0,0.0,59.01,2.51
0.0,,0,,0,0.9,27,0,141,7.0,SOUR CREAM,35,19,0.01,"SOUR CREAM,REDUCED FAT",0.0,0.06,181,0,0,11,0.0,4.1,0.07,1178,0.0,85,0.5,211,7.0,0,117,0.24,8.7,4.1,70,0.300000012,0.04,14.1,436,119,0.3,0.02,0.9,0.4,0.7,71.0,0.27
78.0,"1 bar, 2.8 oz",0,,0,0.0,0,0,192,46.15,CANDIES,13,0,0.0,"CANDIES,HERSHEY'S POT OF GOLD ALMOND BAR",3.799999952,1.85,577,0,0,0,0.0,0.0,0.0,19130,0.0,0,0.0,0,12.82,0,0,0.0,16.667,0.0,64,38.45999908,0.0,38.46,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
425.0,"1 package,  yields",228,1 serving,0,1.4,0,0,0,9.5,OLD EL PASO CHILI W/BNS,16,0,0.0,"OLD EL PASO CHILI W/BNS,CND ENTREE",4.300000191,1.18,109,0,0,0,0.0,1.87,0.0,22514,0.0,0,0.985,0,7.7,0,0,0.0,0.904,0.0,258,0.0,0.0,4.5,0,0,0.0,0.0,0.0,0.0,0.0,76.9,0.0
28.0,1 slice,0,,0,3.6,0,0,0,1.31,TURKEY,48,0,0.0,"TURKEY,BREAST,SMOKED,LEMON PEPPER FLAVOR,97% FAT-FREE",0.0,0.0,95,0,0,0,0.0,0.25,0.0,7943,0.0,0,0.19,0,20.9,0,0,0.0,0.22,0.0,1160,0.0,0.0,0.69,0,0,0.0,0.0,0.0,0.0,0.0,73.5,0.0
140.0,"1 cup, diced",113,"1 cup, shredded",0,5.85,63,0,772,2.1,CHEESE,85,36,0.027,"CHEESE,PAST PROCESS,SWISS,W/DI NA PO4",0.0,0.61,334,0,0,29,0.014,7.046,0.038,1044,0.26,762,0.622,216,24.73,0,192,0.276,16.045,15.9,1370,1.230000019,0.014,25.01,746,198,1.23,0.036,0.0,0.34,2.2,42.31,3.61
263.0,"1 piece, cooked, excluding refuse (yield from 1 lb raw meat with refuse)",85,3 oz,0,1.22,0,0,20,0.0,LAMB,91,0,0.108,"LAMB,DOM,SHLDR,WHL (ARM&BLD),LN&FAT,1/8""FAT,CHOIC,CKD,RSTD",0.0,1.97,269,0,0,23,0.022,7.79,6.04,17245,0.7,185,1.56,251,22.7,24,0,0.24,7.98,26.4,66,0.0,0.09,19.08,0,0,2.65,0.13,0.0,0.0,0.0,56.98,5.44
17.0,1 piece,0,,0,0.7,10,0,49,76.44,CANDIES,14,10,0.329,"CANDIES,FUDGE,CHOC,PREPARED-FROM-RECIPE",1.700000048,1.77,411,4,0,36,0.422,2.943,0.176,19100,0.14,71,0.373,134,2.39,0,43,0.085,6.448,2.5,45,73.12000275,0.026,10.41,159,44,0.09,0.012,0.0,0.18,1.4,9.81,1.11
124.0,"1 serving, 1/2 cup",0,,0,1.98,0,0,32,9.68,CAMPBELL SOUP,4,0,0.0,"CAMPBELL SOUP,CAMPBELL'S RED & WHITE,BROCCOLI CHS SOUP,COND",0.0,0.0,81,0,0,0,0.0,0.0,0.0,6014,0.0,0,0.0,0,1.61,0,0,0.0,1.613,0.0,661,1.610000014,0.0,3.63,806,0,0.0,0.0,1.0,0.0,0.0,83.1,0.0
142.0,"1 item, 5 oz",0,,0,3.23,0,0,109,22.26,MCDONALD'S,167,0,0.073,"MCDONALD'S,BACON EGG & CHS BISCUIT",0.899999976,2.13,304,0,0,12,0.137,5.546,1.959,21360,0.642,335,2.625,121,13.45,0,0,0.416,8.262,0.0,863,2.180000067,0.262,18.77,399,0,0.0,0.093,2.1,0.0,0.0,42.29,0.9"""

#Convert to dictionary
food_data = [dict(x) for x in csv.DictReader(StringIO(food_data))]
food_data = [x for x in food_data if float(x['1st Household Weight'])!=0]

#Values to track
quantities = ['Protein', 'Carbohydrate', 'Total Lipid', 'Kilocalories']

#Create random meals
meals = []
for mealnum in range(10):
  meal = {"name": "Meal #{0}".format(mealnum), "foods":[]}
  for x in range(random.randint(2,4)): #Choose a random number of foods to be in meal
    food = random.choice(food_data)
    food['min'] = 10                   #Use large bounds to ensure we have a feasible solution
    food['max'] = 1000
    weight               = float(food['1st Household Weight']) #Number of units in a standard serving
    for q in quantities:
      #Convert quantities to per-unit measures
      food[q] = float(food[q])/weight 
    meal['foods'].append(food)
  meals.append(meal)

#Create an optimization problem from the meals
total_daily_carbs_min   = 225
total_daily_carbs_max   = 325
total_daily_protein_min = 46
total_daily_protein_max = 56
total_daily_lipids_min  = 44
total_daily_lipids_max  = 78

#Construct variables, totals, and some constraints
constraints = []
for meal in meals:
  #Create a binary variable indicating whether we are using the variable
  meal['use_meal']     = cp.Variable(boolean=True) 
  for q in quantities:
    meal[q] = 0
  for food in meal['foods']:
    food['portion'] = cp.Variable(pos=True)
    #Ensure that we only use an appropriate amount of this food
    constraints.append( food['min']     <= food['portion'] )
    constraints.append( food['portion'] <= food['max']     )
    #Calculate this meal's contributions to the totals.
    #Each items contribution is the portion times the per-unit quantity times a
    #boolean (0, 1) variable indicating whether or not we use the meal
    for q in quantities:
      meal[q] += food['portion']*food[q]

#Dictionary with no sums of meals, yet
totals = {q:0 for q in quantities}

#See: "http://www.ie.boun.edu.tr/~taskin/pdf/IP_tutorial.pdf", "Nonlinear Product Terms"
#Since multiplying to variables produces a non-convex, nonlinear function, we
#have to use some trickery
#Let w have the value of `meal['use_meal']*meal['Protein']`
#Let x=meal['use_meal'] and y=meal['Protein']
#Let u be an upper bound on the value of meal['Protein']
#We will make constraints such that
#   w <= u*y                  
#   w >=0                     if we don't use the meal, `w` is zero
#   w <= y                    if we do use the meal, `w` must not be larger than the meal's value
#   w >= u*(x-1)+y            if we use the meal, then `w>=y` and `w<=y`, so `w==y`; otherwise, `w>=-u+y`

u = 9999 #An upper bound on the value of any meal quantity
for meal in meals:
  for q in quantities:
    w = cp.Variable()
    constraints.append( w<=u*meal['use_meal']             )
    constraints.append( w>=0                              )
    constraints.append( w<=meal[q]                        )
    constraints.append( w>=u*(meal['use_meal']-1)+meal[q] )
    totals[q] += w

#Construct constraints. The totals must be within the ranges given
constraints.append( total_daily_protein_min <= totals['Protein']      )
constraints.append( total_daily_carbs_min   <= totals['Carbohydrate'] )
constraints.append( total_daily_lipids_min  <= totals['Total Lipid']  )
constraints.append( totals['Protein']       <= total_daily_protein_max)
constraints.append( totals['Carbohydrate']  <= total_daily_carbs_max  )
constraints.append( totals['Total Lipid']   <= total_daily_lipids_max )



#Ensure that we're doing three meals because this is the number of meals people
#in some countries eat.
constraints.append( sum([meal['use_meal'] for meal in meals])==3 )

#With an objective value of 1 we are asking the solver to identify any feasible
#solution. We don't care which one.
objective = cp.Minimize(1)

#We could also use:
#objective = cp.Minimize(totals['Kilocalories'])
#to meet nutritional needs while minimizing calories intake

problem = cp.Problem(objective, constraints)

val = problem.solve(solver=cp.GLPK_MI, verbose=True)

if val==1:
  for q in quantities:
    print("{0} = {1}".format(q, totals[q].value))
  for m in meals:
    if m['use_meal'].value!=1:
      continue
    print(m['name'])
    for f in m['foods']:
      print("\t{0} units of {1} (min={2}, max={3})".format(f['portion'].value, f['Description'], f['min'], f['max']))