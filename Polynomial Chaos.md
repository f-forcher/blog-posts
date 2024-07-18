# Polynomial Chaos and its application to Hamiltonian mechanics

## Part 1: Intoduction to Polynomial Chaos
In this series of posts, I will introduce the tecnique known as Polynomial chaos (PC) or polynomial chaos expansion (PCE), 
and explore its application to numerical solutions of stochastic differential equations, especially ones derived from Hamiltonian mechanics.

> [!NOTE]
> A rigorous approach to this topic requires some advanced knowledge of math, physics and statistics,
> but I would like this tutorial to be useful to a wide audience of students, data wranglers and tinkerers from varied background,
> focused on practical applications of numerical computing.
> 
> So I aim to only assume basic knowledge of analysis and linear algebra, and to use images and code
> to introduce the concepts and formulas, if possible.

## Definitions and motivating example
Let's have a look at the equation of motion of an unidimensional oscillator, a basic ordinary differential equation from physics:

```math
x''(t)=-kx(t)
```

Here, $x(t)$ is a function that represents the position at time $t$ of a point mass that is constrained along a unidimensional axis $\hat x$. 
The mass is tied with an idealized spring pulling towards the origin of the axis (hence the negative sign, for a positive $k$). 
The mass of the particle has been absorbed inside the constant $k$, that represents the spring's "specific strength".
The notation $x''(t)$ represents the second derivative of the position with respect to time, also known as acceleration:

```math
x''(t)=\frac{d^2}{dt^2}(x)(t)
```


> [!TIP]
> It is conceptually useful to consider the derivative $\frac{d^2}{dt^2}(f)$ as a higher order function that takes in a first order function, such as our position
> $x(t)$, and transforms it into another function, in this case the acceleration $x''(t)$


> [!TIP]
> More in general, it will be useful to keep in mind the "type" of math expressions, analogous to types in computer science. I will use a pseudo-type-annotation
> syntax that should be fairly intuitive.
>
> In this example, both $x(t)$ and $x''(t)$ are simple `Real -> Real` scalar-valued functions of (scalar) time, while the "derivative operator"
> $\frac{d^2}{dt^2}$ would have type `(Real -> Real) -> (Real -> Real)`
