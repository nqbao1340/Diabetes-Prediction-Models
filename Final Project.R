#Importing data
library("Hmisc")
library(tree)
library(boot)
library(corrplot)
library(plotROC)
library(pROC)
library(glmnet)
library(e1071)
library(ROCR)
library(caret)

D <- read.csv("diabetes.csv")
str(D)
head(D)
dim(D)
table(D$Outcome)

#Data processing 
i = 1
while (i <= dim(D)[1]) {
  for (j in (2:8)) {
    if ((is.na(D[i,j])) || (D[i,j] == 0)) {
      D = D[-i,]
      i = i-1
      break
    }
  }
  i = i+1
} 

#Data exploring
dim(D)
head(D)
table(D$Outcome)
summary(D)
par(mfrow=c(2,4))
boxplot(D$Pregnancies~D$Outcome, data=D, col=c(4,2), xlab="Outcome", ylab="Pregnancies", main="Pregnancies ~ Outcome")
boxplot(D$Glucose~D$Outcome, data=D, col=c(4,2), xlab="Outcome", ylab="Glucose", main="Glucose ~ Outcome")
boxplot(D$BloodPressure~D$Outcome, data=D, col=c(4,2), xlab="Outcome", ylab="BloodPressure", main="BloodPressure ~ Outcome")
boxplot(D$SkinThickness~D$Outcome, data=D, col=c(4,2), xlab="Outcome", ylab="SkinThickness", main="SkinThickness ~ Outcome")
boxplot(D$Insulin~D$Outcome, data=D, col=c(4,2), xlab="Outcome", ylab="Insulin", main="Insulin ~ Outcome")
boxplot(D$BMI~D$Outcome, data=D, col=c(4,2), xlab="Outcome", ylab="BMI", main="BMI ~ Outcome")
boxplot(D$DiabetesPedigreeFunction~D$Outcome, data=D, col=c(4,2), xlab="Outcome", ylab="DiabetesPedigreeFunction", main="DiabetesPedigreeFunction ~ Outcome")
boxplot(D$Age~D$Outcome, data=D, col=c(4,2), xlab="Outcome", ylab="Age", main="Ages ~ Outcome")

par(mfrow=c(1,1))
r = cor(D[1:8])
round(r,2)
corrplot(r, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#Clustering
# K-means
X = D[,1:8]
m = c()
for (i in (20:50)) {
  set.seed(20289300)
  km = kmeans(X, centers = 2, nstart=i)
  m[i] = km$tot.withinss
}
m

km = kmeans(X, centers = 2, nstart=20)
km$size
km$betweenss
pp = prcomp(X)
plot(pp$x[,1:2], col=fitted(km, "classes")+1,
     xaxt="n", yaxt="n", pch=20, main = "K-means clustering (K = 2, nstart = 20")

fitted(km, "classes")
table(Outcome=D$Outcome, cluster=fitted(km, "classes"))
table(D$Outcome)

# Hierarchical
hc.complete = hclust(dist(X), method = "complete")
hc.complete
hc.single = hclust(dist(X), method = "single")
hc.single
hc.average = hclust(dist(X), method = "average")
hc.average
par(mfrow=c(1,3))
plot(hc.complete, main = "Complete Linkage", xlab = "", sub = "", cex = 0.9)
plot(hc.average, main = "Average Linkage", xlab = "", sub = "", cex = 0.9)
plot(hc.single, main = "Single Linkage", xlab = "", sub = "", cex = 0.9)
table(cutree(hc.complete, 2),D$Outcome)
table(cutree(hc.average, 2),D$Outcome)
table(cutree(hc.single, 2), D$Outcome)

#Purity
ClusterPurity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}
ClusterPurity(fitted(km, "classes"),D$Outcome)   #K-means
ClusterPurity(cutree(hc.complete, 2),D$Outcome)  #Complete
ClusterPurity(cutree(hc.average, 2),D$Outcome)   #Average  
ClusterPurity(cutree(hc.single, 2),D$Outcome)    #Single


#Split data into train and test dataset
set.seed(20289300)
train = sample(1:nrow(D), round(nrow(D)*0.65))

#Decision Tree model
D$Outcome = as.factor(D$Outcome)
str(D)
table(D$Outcome)

traintree = D[train,]
testtree = D[-train,]
treemodel1 = tree(Outcome~., traintree)
par(mfrow=c(1,1))
plot(treemodel1)
text(treemodel1,pretty=0)
summary(treemodel1)

tree_pred1 = predict(treemodel1, testtree, type="class")
mis = table(tree_pred1,testtree$Outcome)
table(tree_pred1,testtree$Outcome)
(mis[1,2]+mis[2,1])/sum(mis)
  
#cross validation
set.seed(20289300)
cv_regtree <- cv.tree(treemodel1, FUN=prune.misclass)
cv_regtree
plot(cv_regtree$size, cv_regtree$dev, type="b", main="Cross validation")
treemodel2 = prune.misclass(treemodel1, best=5)
plot(treemodel2, main = "Classification tree model with 5 terminal nodes")
text(treemodel2,pretty=0)

tree_pred2 = predict(treemodel2, testtree, type="class")
mis = table(tree_pred2,testtree$Outcome)
table(tree_pred2,testtree$Outcome)
(mis[1,2]+mis[2,1])/sum(mis)
summary(treemodel2)  

#Standardize data
str(D)
head(D)
mu=c()
s=c()
for (i in (1:8)){
  mu[i] = mean(D[,i])
  s[i] = sd(D[,i])
}
for (i in (1:dim(D)[1])){
  for (j in (1:8)) {
    D[i,j] = (D[i,j] - mu[j])/s[j]
  }
}

logtrain = D[train,]
logtest = D[-train,]
head(logtrain)
dim(logtrain)

#Backward selection
LogLoss <- function(pred, res){
  (-1/length(pred)) * sum (res * log(pred) + (1-res)*log(1-pred))
}
logmodel = glm(Outcome~., data=logtrain, family=binomial)
summary(logmodel)
pred = predict(logmodel, logtest, type="response")
LogLoss(pred,as.numeric(logtest$Outcome)-1)
confusionMatrix(table(predict(logmodel, logtest, type="response") >= 0.5, logtest$Outcome == 1))
g <- roc(logtest$Outcome ~ pred, data = logtest)
g$auc
plot(g, main = "ROC curve of Model 3")
anova(logmodel)


logmodel2 = glm(Outcome~Glucose + BMI, data=logtrain, family=binomial)
summary(logmodel2)
pred2 = predict(logmodel2, logtest, type="response")
LogLoss(pred2,as.numeric(logtest$Outcome)-1)
c = confusionMatrix(table(predict(logmodel2, logtest, type="response") >= 0.5, logtest$Outcome == 1))
g <- roc(logtest$Outcome ~ pred2, data = logtest)
plot(g, main = "ROC curve of Model 4")
c$table

#Polynomial
accuracy=c()
precision=c()
Aucc = c()
accuracy[1] = (c$table[1,1]+c$table[2,2])/sum(c$table)
precision[1] = (c$table[2,2])/(c$table[1,2]+c$table[2,2])
Aucc[1] = g$auc[1]

for (i in (2:6)) {
  polymodel = glm(Outcome~poly(Glucose,i) + poly(BMI,i), data=logtrain, family=binomial)
  c = confusionMatrix(table(predict(polymodel, logtest, type="response") >= 0.5, logtest$Outcome == 1))
  polypred = predict(polymodel, logtest, type="response")
  polyroc <- roc(logtest$Outcome ~ polypred, data = logtest)
  Aucc[i] = polyroc$auc[1]
  accuracy[i] = (c$table[1,1]+c$table[2,2])/sum(c$table)
  precision[i] = (c$table[2,2])/(c$table[1,2]+c$table[2,2])
  
}
accuracy
precision
Aucc

# Cut-off value
accuracy=c()
precision=c()
cutoff = seq(from=0.1, to=0.9, by=0.05)
j = 0

for (i in cutoff) {
  j = j+1
  c = confusionMatrix(table(predict(logmodel2, logtest, type="response") >= i, logtest$Outcome == 1))
  accuracy[j] = (c$table[1,1]+c$table[2,2])/sum(c$table)
  precision[j] = (c$table[2,2])/(c$table[1,2]+c$table[2,2])
}

plot(precision~cutoff, col=4, type="b", lwd=2,main = "Accuracy and Precision vs cutoff", ylab = "rate")
points(accuracy~cutoff, type="b", col=2, lwd=2)
abline(v=0.5, lty=3)
abline(v=0.3, lty=3)
legend("topright", c("Accuracy", "Precision"), col = c("red","blue"), lty = c(1, 1), pch=c(1,1))

logmodel2 = glm(Outcome~Glucose + BMI, data=logtrain, family=binomial)
summary(logmodel2)
pred2 = predict(logmodel2, logtest, type="response")
LogLoss(pred2,as.numeric(logtest$Outcome)-1)
c = confusionMatrix(table(predict(logmodel2, logtest, type="response") >= 0.3, logtest$Outcome == 1))
g <- roc(logtest$Outcome ~ pred2, data = logtest)
plot(g, main = "ROC curve of Model 4")
c$table
g$auc

# Supporting vector machine
set.seed(20289300)
svmtrain = D[train,]
svmtest = D[-train,]
head(svmtrain)
dim(svmtrain)
tune_out = tune(svm, Outcome ~ ., data = svmtrain, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)))
summary(tune_out)
bestmodel1 = tune_out$best.model
summary(bestmodel1)

# polynomial
set.seed(20289300)
tune_out2 = tune(svm, Outcome ~ ., data = svmtrain, kernel = "polynomial",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000), d = c(2:5)))
summary(tune_out2)
best_model2 = tune_out2$best.model
summary(best_model2)

#radial
set.seed(20289300)
tune_out3 = tune(svm, Outcome ~ ., data = svmtrain, kernel = "radial",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000), gamma = c(0.5, 1, 2, 3,4)))
summary(tune_out3)
best_model3 = tune_out3$best.model
summary(best_model3)

svmbest = svm(Outcome~., data=svmtrain, kernel="radial", cost=1, gamma=0.5)
svmpredict = predict(svmbest, svmtest)
table(svmpredict,svmtest$Outcome)

rocplot = function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance (predob, "tpr", "fpr")
  plot(perf, ...)
}
fitted = attributes(predict(svmbest, svmtest,
                            decision.values = TRUE))$decision.values
rocplot(fitted, svmtest$Outcome, main="ROC curve for radial kernel function with cost=1, gamma=0.5")

