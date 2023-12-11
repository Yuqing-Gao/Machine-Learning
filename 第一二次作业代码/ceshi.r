#异常值处理
price=read.csv("Boats.csv",fileEncoding = 'gbk',header=T)
price=price[c(1:961),c(2:11)]  #ID对建模无助，故删掉第一列
pic=boxplot(price$Price)
low=pic$stats[1]
up=pic$stats[5]
price=price[price$Price>=low&price$Price<=up,]
str(price)

#彩色相关矩阵
library(corrplot)
corr=cor(price)
corrplot(corr, method = 'shade', type = "full",
         addCoef.col = "black", number.cex = 0.75,
         tl.cex = 0.75, tl.col = "black", tl.srt = 45)#得出GDP.Capita和GDP两个变量与因变量关系小，后续可考虑删除处理
set.seed(20)
train=sample(1:926,648)#划分训练集与测试集比例7:3
train_price=price[train,]
test_price=price[-train,]
library(caret)
control <- trainControl(method = 'cv',number = 10)#选择10折交叉验证
k_values <- data.frame(k = 2:10)#选取不同K值，后续用于网格搜索
grid <- expand.grid(.k = k_values)

# knn模型训练——曼哈顿距离
#删除一个变量Draft
price12=price[,-c(5)]
train_price=price12[train,]
test_price=price12[-train,]
library(caret)
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
set.seed(20)
model12 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "manhattan")
model12

# knn模型训练——马氏距离
#删除一个变量Draft
price22=price[,-c(5)]
train_price=price22[train,]
test_price=price22[-train,]
library(caret)
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
mahal <- mahalanobis(train_price[, -1], center = colMeans(train_price[, -1]), cov = cov(train_price[, -1]))
set.seed(20)
model22 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "mahalanobis", mahal = mahal)
model22