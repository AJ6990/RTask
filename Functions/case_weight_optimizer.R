## Custom function to run glm in caret, with variable case weights
library(caret)
weighted_glm <- getModelInfo('glm', regex = F)[[1]]

weighted_glm$parameters <- data.frame(parameter = 'weight',
                                     class = 'numeric',
                                     label = 'Weight ratio (Minority/Majority')


weighted_glm$grid <- function(x, y, len = NULL, search = "grid") {
  p <- ncol(x)
  if(search == "grid") {
    grid <- expand.grid(weight = seq(len))
  } else {
    grid <- expand.grid(weight = runif(runif, min = 1, max = 10))
  }
  grid
}

weighted_glm$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  dat <- if(is.data.frame(x)) x else as.data.frame(x, stringsAsFactors = TRUE)
  dat$.outcome <- y
  if(length(levels(y)) > 2) stop("glm models can only use 2-class outcomes")
  
  theDots <- list(...)
  if(!any(names(theDots) == "family"))
  {
    theDots$family <- if(is.factor(y)) binomial() else gaussian()
  }
  
  ## pass a model weights based on the weight ratio parameter
  
  minority_class <- names(which.min(table(y)))
  majority_class <- names(which.max(table(y)))
  
  wts <- ifelse(y == minority_class,
                param$weight,
                1)
  
  wts <- wts / sum(wts) * length(wts)
  
  if(!is.null(wts)) theDots$weights <- wts
  
  modelArgs <- c(list(formula = as.formula(".outcome ~ ."), data = dat), theDots)
  
  out <- do.call("glm", modelArgs)
  ## When we use do.call(), the call information can contain a ton of
  ## information. Including the content of the data. We eliminate it.
  out$call <- NULL
  out
}


binarySummary <- function (data, lev = NULL, model = NULL) 
{
  out <- c(prSummary(data, lev = levels(data$obs), model = NULL))
  
  g_mean <- sqrt(out['Precision'] * out['Recall'])
  
  c(G = g_mean, out)
}