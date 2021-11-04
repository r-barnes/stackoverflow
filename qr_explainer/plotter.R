library(ggplot2)

### Equation: y=a*sin(b*t)+c.unif*amp
# variables
n <- 100 # number of data points
a <- 3
b <- 2
amplitude <- 0.5
c.unif <- runif(n)
c.norm <- rnorm(n)

# generate data and calculate "y"
set.seed(1)
y <- a*c.unif+b+amplitude*c.norm # uniform error

data = data.frame(x=c.unif, y=y)

ggplot(data, aes(x=x, y=y)) + geom_point()
ggsave("scatter.png", dpi=300)

ggplot(data, aes(x=x, y=y)) + geom_point() + geom_smooth(method='lm', formula= y~x, se=F, size=2)
ggsave("scatter_fit.png", dpi=300)

y2 <- 3.5*c.unif+b+amplitude*rnorm(n) # uniform error
y3 <- 2.5*c.unif+b+amplitude*rnorm(n) # uniform error

data=data.frame(x=c.unif, y=y, y2=y2, y3=y3)
ggplot(data) +
  geom_point(aes(x=x,y=y)) +
  geom_smooth(aes(x=x, y=y), method='lm', formula= y~x, se=F, size=2, color="#66c2a5") +
  geom_smooth(aes(x=x, y=y2), method='lm', formula= y~x, se=F, size=2, color="#fc8d62") +
  geom_smooth(aes(x=x, y=y3), method='lm', formula= y~x, se=F, size=2, color="#8da0cb")
ggsave("different_fits.png", dpi=300)



library(GGally)
library(ggplot2)
library(dplyr)

my_fn <- function(data, mapping, method="loess", ...){
  p <- ggplot(data = data, mapping = mapping) +
  geom_point() +
  geom_smooth(method=method, ...)
  p
}

temp = mtcars %>% select(MPG=mpg, Horsepower=hp, Weight=wt, Displacement=disp)

ggpairs(
  temp,
  lower = list(continuous = wrap(my_fn, method="lm")),
  diag=list(continuous="bar"),
  axisLabels='show')
ggsave("mtcars.png", dpi=300)


data=data.frame(x=c(-2,-1,1,2,3,4),y=c(4,1,2,1,5,6))
ggplot(data, aes(x=x,y=y)) + geom_point(size=3)
ggsave("quad_data.png", dpi=300)


fit_func <- function(x){
  1.1 + -0.35357143*x + 0.425*x^2
}

fx = seq(-5,5,0.01)
fy = fit_func(fx)
data=data.frame(x=c(-2,-1,1,2,3,4),y=c(4,1,2,1,5,6))
fdata=data.frame(x=fx, y=fy)
ggplot() + geom_point(data=data, aes(x=x,y=y), size=3) + geom_line(data=fdata,aes(x=fx,y=fy), size=3, color="blue")
ggsave("quad_fit.png", dpi=300)