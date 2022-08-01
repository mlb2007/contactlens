# Beta distribution

Ref: https://www.youtube.com/watch?v=UZjlBQbV1KU&list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo&index=23

* Beta distribution is a generalization of *uniform* distribution

* *uniform distribution*
  * continuous and bounded between 0-1
  * Flat line

* beta-distributions: A family of distributions determined by $\alpha, \beta$ (We will call it a, b for convenience)
  * values between 0-1
  * A generalization of "flat-line"uniform distribution
  * PDF: $ f(X) = C.X^{a-1} (1-X)^{b-1}$, $0 <= X <= 1$  
  * In order to make the PDF integrate and yield a value of 1, we integrate the PDF from [0-1] and determine $C$ (normalizing constant )
  * This famous integral is called the $\beta$ function
  
  * used for proportions and percentages distribution modeling ...

  * a, b can be *any* positive real number

  * a = 1, b = 1 is the uniform distribution

  * a = 1/2, b = 1/2 ==> U-shape

  * a = 2, b = 2 ==> inverted U-shape

  * Often used as a prior for a parameter (0-1)
    * i.e. we model the parameter's *distribution* using *beta* distribution

  * *conjugate-prior* to binomial distribution  

### Conjugate-prior

* Say we are completely ignorant of distribution of something
* Simply, we can use uniform distribution
  * But we can be sophisticated and use *beta* distribution

* Consider Binomial distribution $X|p ~ Bin(n, p)$ where $p$ is the prior distribution of probability, the probability distribution of our data X.
  * What is $p = beta(a,b)$ ? Here the prior itself is determined from another distribution. So before we observe X, we make a decision on the *prior* probability of X. i.e. we assume that our data is going to come from a binomial distribution with a probability given by the beat-distribution, instead of using say Bernoulli coin-flip probability (p = 1/2)

* After we observe the data, we want to update our posterior probability distribution, $p|X$. 
    * If we apply Bayes rule, we find that the posterior probability is *another* *beta* distribution, i.e. we get a *conjugate* distribution, i.e. another member of *family* of *beta* distribution! as: $p = beta(a+X, b+(n-X))$

* Intuitively, if we consider a, b to be proportions of stuff, the conjugate prior (fancy name for posterior) is simply adding the new observed data $X$ to the prior distribution
  * Note this is also the same Laplace trick we use for updating *reviews* given *reviews* we see ...

## Gamma distribution

* If we have a sequence of numbers, say error data, how do we *interpolate* that sequence between two numbers ?
  * We can have many ways to do this, but gamma function is a good way to do this ... Is this statement true ?

* $\Gamma(a) = \int X^a e^{-X}\frac{dX}{X}$, for $a > 0$ for any $X$

* Gamma function can be considered as approximation for factorial, In fact $\Gamma(X+1) =  X . \Gamma(X)$

* $\Gamma(\frac{1}{2}) = \sqrt(\pi)$. We see $\sqrt(\pi)$ in normal distribution .. [$\int \exp^{-\frac{X^2}{2}}dx$]

## Gamma distribution

* Going from Gamma function to Gamma distribution, we simply divide the gamma function defined above by $\Gamma(a)$ and we get gamma distribution
 
* Gamma distribution relates to Poisson, beta, normal distributions ...

* Given intervals, gamma distribution naturally arises as a summation of these intervals to get the real quantity
  * i.e. classic example is if the inter-arrival times are given (exponential distribution), the actual time of arrival is given by summation of inter-arrival times and this is shown to be a gamma function!!

 









