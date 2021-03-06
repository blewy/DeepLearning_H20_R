rm(list = ls())
options(scipen=999)
options(width = 70, digits = 2)

library(RCurl)
library(h2o)
library(gridExtra)
library(mgcv)
library(RCurl)
library(jsonlite)
library(caret)
library(e1071)
library(data.table)

## basic stats packages
library(statmod)
library(MASS)

## neural networks
library(nnet)
library(neuralnet)
library(RSNNS)

## deep Learning
library(deepnet)
library(darch)
library(h2o)

#Packages for this chapter
library(glmnet)


set.seed(88)

train <- read.csv("./data/train.csv", stringsAsFactors = F)
train$id <- NULL
test <- read.csv("./data/test.csv", stringsAsFactors = F)
sub <- read.csv("./data/sample_submission.csv", stringsAsFactors = F)
all_species <- names(sub)[2:100]

train$species <- factor(train$species)
str(train$species)

cl <- h2o.init(
  max_mem_size = "4G",
  nthreads = 2)

h2oleafs <- as.h2o(
  train,
  destination_frame = "h2oleafs")

h2oleafs.train <- h2oleafs

xnames <- colnames(h2oleafs.train)[-1]

system.time(ex1 <- h2o.deeplearning(
  x = xnames,
  y = "species",
  training_frame= h2oleafs.train,
  validation_frame = h2oleafs.train,
  activation = "RectifierWithDropout",
  hidden = c(100),
  epochs = 100,
  adaptive_rate = FALSE,
  rate = .001,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(.2)
))

# Split dataset giving the training dataset 75% of the data
leafs.split <- h2o.splitFrame(data=h2oleafs, ratios=0.75)
# Create a training set from the 1st dataset in the split
leafs.train <- leafs.split[[1]]
# Create a testing set from the 2nd dataset in the split
leafs.test <- leafs.split[[2]]


run <- function(seed, name = paste0("m_", seed), run = TRUE) {
  set.seed(seed)
  
  p <- list(
    Name = name,
    seed = seed,
    depth = sample(1:5, 1),
    l1 = runif(1, 0, .01),
    l2 = runif(1, 0, .01),
    input_dropout = rbeta(1, 1, 12),
    rho = runif(1, .9, .999),
    epsilon = runif(1, 1e-10, 1e-4))
  
  p$neurons <- sample(20:600, p$depth, TRUE)
  p$hidden_dropout <- rbeta(p$depth, 1.5, 1)/2
  
  if (run) {
    model <- h2o.deeplearning(
      x = xnames,
      y = "species",
      training_frame = leafs.train,
      activation = "RectifierWithDropout",
      hidden = p$neurons,
      epochs = 100,
      loss = "CrossEntropy",
      input_dropout_ratio = p$input_dropout,
      hidden_dropout_ratios = p$hidden_dropout,
      l1 = p$l1,
      l2 = p$l2,
      rho = p$rho,
      epsilon = p$epsilon,
      export_weights_and_biases = TRUE,
      model_id = p$Name
    )
    
    ## performance on training data
    p$MSE <- h2o.mse(model)
    p$R2 <- h2o.r2(model)
    p$Logloss <- h2o.logloss(model)
    p$CM <- h2o.confusionMatrix(model)
    
    ## performance on testing data
    perf <- h2o.performance(model, leafs.test)
    p$T.MSE <- h2o.mse(perf)
    p$T.R2 <- h2o.r2(perf)
    p$T.Logloss <- h2o.logloss(perf)
    p$T.CM <- h2o.confusionMatrix(perf)
    
  } else {
    model <- NULL
  }
  
  return(list(
    Params = p,
    Model = model))
}

use.seeds <- c(403L, 10L, 329737957L, -753102721L, 1148078598L,
               -1945176688L,
               -1395587021L, -1662228527L, 367521152L, 217718878L, 1370247081L,
               571790939L, -2065569174L, 1584125708L, 1987682639L, 818264581L,
               1748945084L, 264331666L, 1408989837L, 2010310855L, 1080941998L,
               1107560456L, -1697965045L, 1540094185L, 1807685560L, 2015326310L,
               -1685044991L, 1348376467L, -1013192638L, -757809164L, 1815878135L,
               -1183855123L, -91578748L, -1942404950L, -846262763L, -497569105L,
               -1489909578L, 1992656608L, -778110429L, -313088703L, -758818768L,
               -696909234L, 673359545L, 1084007115L, -1140731014L, -877493636L,
               -1319881025L, 3030933L, -154241108L, -1831664254L)


model.res <- lapply(use.seeds, run)

save(model.res, file = "./RData/model.res.RData", compress = F)

library(reshape2)
library(ggplot2)
model.res.dat <- do.call(rbind, lapply(model.res, function(x)
  with(x$Params,
       data.frame(l1 = l1, l2 = l2,
                  depth = depth, input_dropout = input_dropout,
                  SumNeurons = sum(neurons),
                  MeanHiddenDropout = mean(hidden_dropout),
                  rho = rho, epsilon = epsilon, MSE = T.MSE))))

dev.off()
p.perf <- ggplot(melt(model.res.dat, id.vars = c("MSE")), aes(value, MSE))+ facet_wrap(~ variable, scales = "free_x", ncol = 2)  + stat_smooth(color = "black")+ geom_point()+theme_classic()
p.perf

str(model.res.dat)


library(glmnet)
library(gridExtra)
library(mgcv)
library(statmod)
require(mgcv)
summary(m.gam <- gam(MSE ~ s(l1, k = 4) +
                       s(l2, k = 4) +
                       s(input_dropout) +
                       s(rho, k = 4) +
                       s(epsilon, k = 4) +
                       s(MeanHiddenDropout, k = 4) +
                       s(depth, k = 4) +
                       s(SumNeurons, k = 4) ,
                       #te(depth, SumNeurons, k = 4),
                     data = model.res.dat))

m.gam$formula

par(mfrow = c(4, 2))
for (i in 1:8) {
  plot(m.gam, select = i)
}

save(m.gam, file = "./RData/m.gam.RData", compress = F)

#### -------- Bayesian Optimization --------------

library(rBayesianOptimization)
gam_optimizer <- function(l1_p,l2_p,input_dropout_p,rho_p,epsilon_p,MeanHiddenDropout_p,depth_p,SumNeurons_p) 
  {
    params <-data.frame(l1=l1_p,
                        l2=l2_p,
                        input_dropout=input_dropout_p,
                        rho=rho_p,
                        epsilon=epsilon_p,
                        MeanHiddenDropout=MeanHiddenDropout_p,
                        depth=depth_p,
                        SumNeurons=SumNeurons_p)
    
    gam.MSE <- predict(m.gam,params)*-1
 
    list(Score =gam.MSE,
         Pred = 0)
}


initial_grid <- model.res.dat
initial_grid$MSE<-NULL
 

OPT_Res <- BayesianOptimization(gam_optimizer,
                                bounds = list(l1 = c(0, 0.01),
                                              l2 = c(0.001, 0.01),
                                input_dropout  = c(0.001, 0.5),
                                rho  =  c(0.8, 1),
                                epsilon = c(0,0.001),
                                MeanHiddenDropout = c(0.15,0.5),
                                depth = c(1,5),
                                SumNeurons = c(990,1000)),
                                #init_grid_dt = initial_grid, 
                                init_points = 100, 
                                n_iter = 3,
                                acq = "ucb", 
                                kappa = 2.576, 
                                eps = 0.0,
                                verbose = TRUE)


best_params<- OPT_Res$Best_Par

best_params["l1"]


model.optimized <- h2o.deeplearning(
  x = xnames,
  y = "species",
  training_frame = leafs.train,
  activation = "RectifierWithDropout",
  hidden = best_params["SumNeurons"],
  epochs = 100,
  loss = "CrossEntropy",
  input_dropout_ratio = best_params["input_dropout"],
  hidden_dropout_ratios = best_params["MeanHiddenDropout"],
  l1 = best_params["l1"],
  l2 = best_params["l2"],
  rho = best_params["rho"],
  epsilon = best_params["epsilon"],
  export_weights_and_biases = TRUE,
  model_id = "optimized_model"
)


h2o.performance(model.optimized)



summary(model.res.dat$MSE )



























