# F-Minimizer-Gradient-Descent
Finding the minimizer of F using Gradient Descent with Exact Line Search, Fixed Stepsize, and Backtracking Algortihm


# Minimizer of F using Exact Line Search

      def grad_f(x):
       x1, x2 = x
       grad_x1 = 4*x1 + 2*x2 + 1
       grad_x2 = 2*x2 + 2*x1 – 1
       return np.array([grad_x1, grad_x2])
       
This function calculates the gradient vector of the function f(x1,x2) by computing the
partial derivatives of both x1 and x2 and returns as array.

To find the minimizer of f using gradient descent with exact line search, we create a function
called gd. This function takes the arguments: start, f, gradient, hessian, maxiter, and the tolerance
(tol). Tolerance checks for the convergence and maxiter calculates how many iterations were
taken to reach convergence.

      def gd(start, f, gradient, hessian, maxiter, tol=0.01):
      step = [start] ## tracking history of x
      step_f = [f(start)]
      x = start
      k = 0
      for i in range(maxiter):
      h = hessian(x)
      step_size = (1)/(h)
      diff = step_size*gradient(x)
      if np.abs(diff)<tol:
      break
      x = x - diff
      k = k+1
      fc = f(x)
      step_f.append(fc)
      step.append(x) ## tracking
      return step, x
      
In this function we set k=0 to track the number of iterations needed to converge. We then
calculate the value of hessian and use the hessian to calculate the step size. We then find the
value of diff which is the step size multiplied by the gradient. Next we update x and k until the
value of diff is less than the value of diff is less than the value of the set tolerance

    hisory, solution = gd(np.array([1,-1]),f,grad_f,hess_f,23)
    print('solution =', solution)
    Iterations = 23
    solution = [-0.99938965 1.49902344]
    
We get that the minimizer of f is solution = [-0.99938965 1.49902344] and it took 23 iterations
to reach it.

# Minimizer of F using Fixed Stepsize

To find the minimizer of f using gradient descent with fixed stepsize we create a function called
gd. This function takes arguments: start, f, gradient, step_size, maxiter, and tolerance (tol).
Step_size is at a fixed value in this program it is equal to 0.1, maxiter is used to keep track of
how many iterations it took to reach convergence, and tol checks for when convergence is
reached.

    def gd(start, f, gradient, step_size, maxiter, tol=0.01):
     step = [start] ## tracking history of x
     x = start
     k=0
     for i in range(maxiter):
     diff = step_size*gradient(x)
     if np.all(np.abs(diff)<tol):
     break
     k=k+1
     x = x - diff
     fc = f(x)
     step.append(x) ## tracking
     print('iterations = ', k)
     return step, x

In this function we set k=0 to track the iteration count. The stepsize in this function is fixed at 0.1
and we multiply by the gradient to get our value for diff. We then update k and x until diff is less
than the value of the set tolerance.

    history, solution = gd(np.array([4,2]),f,grad_f,0.1,100)
    #print('history =', history)
    print('solution =', solution)
    iterations = 34
    solution = [-0.9223028 1.37428329]
    
We get the minimizer of f is solution = [-0.9223028 1.37428329] and it took 34 iterations to
reach convergence

# Minimizer of F using Backtracking Algorithm

To find the minimizer of f using gradient descent with backtracking algorithm we create a function called
steepestdescent. This function takes arguments: f, df, step_size, x0, maxiter, and tolerance (tol).
Step_size starts at 1 and is reduced by a factor β until the stopping condition, maxiter is used to
keep track of how many iterations it took to reach convergence, and tol checks for when
convergence is reached.

     def steepestdescent(f,df,step_size,x0,tol=1.e-3,maxit=100):
     x = x0
     r = df(x0)
     iters = 0
     while ( np.abs(npl.norm(r))>tol and iters<maxit ):
     lambda_k = step_size(x)
     x = x - lambda_k * r
     r = df(x)
     iters += 1
     return x, iters
 
In this function we start at the initial value x0 and set r =gradient at x0. We set iters = 0 to start
the count of how many iterations it takes to reach convergence. We then update the stepsize to
determine how far we go in the opposite direction of the gradient. X is then updated and used to
calculate the new value of the gradient r at the value of x. The iters counter is also updated and
this is done until the norm of gradient r is greater than tol.

      x0 = np.array([2.0,1.0])
      x, iters =steepestdescent(f, df, step_size,x0, tol = 1.e-8, maxit = 100)
      print('solution = ', x)
      print('iteration =', iters)
      solution = [-1. 1.5]
      iteration = 61

We find that the minimizer of the solution f = [-1, 1.5] and it took 61 iterations to reach
convergence
