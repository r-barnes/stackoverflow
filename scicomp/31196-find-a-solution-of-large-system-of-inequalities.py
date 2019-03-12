The problem is that no optimization software implements strict inequalities. One reason for this is that the difference between say, 1e-999 and 0 is really, really small. So, operationally, a strict equality and an equality shakeout to the same answer. Let's explore this a bit.

[cvxpy](TODO) is a Python package that allows you to write a linear program in a generic way and then pass it to one of a large number solvers. Like most such programs, it does not allow for strict inequalitis.

I've written a representation of your problem below:

    #!/usr/bin/env python3
    import cvxpy as cp
    import numpy as np 

    def f(z):
      return np.array([
        z**2+z-3,
        z**3+1/z,
        0.1*z**4+1/np.sqrt(z),
        -4
      ])

    #Generate 100 random numbers in [-1,1)
    zs = np.random.uniform(low=-1, high=1, size=100)
    #Run the function on each number
    zs = [f(z) for z in zs]
    #Strip out results with a NaN
    zs = [z for z in zs if not any(np.isnan(z))]

    zs = [[-1,-1,-1,-1],[1,1,1,1]]

    #x vector of length four to match length of vector from `f`
    x = cp.Variable(4)

    #Multiply f(z)'s by x's and build constraints
    cons = [z*x>=0 for z in zs]

    #Constant objective implies a problem in which we only want to find a feasible
    #point
    obj = cp.Maximize(1) 

    #Create a problem with the objective and constraints
    prob = cp.Problem(obj, cons)

    #Solve problem, get optimal value
    val = prob.solve()

    if val==-np.inf:
      print("NO SOLUTION FOUND")
    else:
      print(x.value)

If we run the above code, we find that the solver indeed finds the 0-vector as an answer.

Now, you would like your answer to be strictly greater than zero. Since there is no solver that will do this (that I know of), one way to achieve it is to ask the solver to find a solution to a different problem. Rather than solving $Ax>0$, we can ask the solver to find $Ax\ge\epsilon$. Where $\epsilon$ is some small value. Since all you want is a feasible point, this is a reasonable approach since any solution to $Ax\ge\epsilon$ is also a solution to $Ax>0$. This is the simple solution you thought you were looking for.

We can implement this by changing the line:

    cons = [z*x>=0 for z in zs]

to

    cons = [z*x>=0.0001 for z in zs]

Running the code again, I get the answer:

    [ 1.34351053e-09 -9.62045447e-10  5.11738862e-09 -3.99990907e-05]

(You may get a different answer because the solvers leverage stochasticity.) This looks promising! But there's a caveat here... What if we use this program?

    #!/usr/bin/env python3
    import cvxpy as cp
    import numpy as np 

    #A system with no solution
    zs = [[-1,-1,-1,-1],[1,1,1,1]]

    #x vector of length four to match length of vector from `f`
    x = cp.Variable(4)

    #Multiply f(z)'s by x's and build constraints
    cons = [z*x>=0.0001 for z in zs]

    #Constant objective implies a problem in which we only want to find a feasible
    #point
    obj = cp.Maximize(1) 

    #Create a problem with the objective and constraints
    prob = cp.Problem(obj, cons)

    #Solve problem, get optimal value
    val = prob.solve()

    if val==-np.inf:
      print("NO SOLUTION FOUND")
    else:
      print(x.value)

It's obvious from inspection that there can be no solution satisfying this system. Interestingly, however, when you run the program (using the default solver) the 0-vector is returned as an answer! If we modify the program to read:

    cons = [z*x>=0.001 for z in zs]

the same thing happens. Only when we get to:

    cons = [z*x>=0.01 for z in zs]

do we finally get the correct response: that there is no solution.

There are a few reasons for this:

* Internally, the solver is using a floating-point representation whose limited precision results in it _thinking_ it's solved the problem when it really hasn't. (You could deal with this by using a rational-number solver that does its calculations using fractional representations.)
* More generally, the solver may not be numerically robust. The subfield of "robust optimization" can be leveraged to find disciplined ways of handling this.
* Philosophically, asking the solver to differentiate between small values of epsilon and 0 is silly. Say you're optimizing the floor plan of a house using numbers which represent metres and choose epsilon as 1e-10. You're asking for a solution that differs from zero by the width of an atom. Say you're calculating a solar trajectory with numbers representing astronomical units (1 AU is the distance from Earth to the sun - 93 million miles): the difference between 1e-10 and 0 is 50 feet (the width of a house).

Perhaps the simplest way of dealing with the problems above is to rescale your system so that small values _are_ unimportant. For instance, rather than measuring my floorplan in meters, I could measure it in millimeters.

It's worth noting that there is an entire class of problems for which your question has trivial and reliable solutions. If any column of your $A$ matrix contains only positive values greater than zero, then setting that column's corresponding $x$ value to 1 and all other $x$ values to 0 provides an answer. Similarly, if any column contains only negative values then choosing -1 for the corresponding $x$ value and 0 elsewhere provides an answer.