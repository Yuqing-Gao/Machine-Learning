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
set.seed(20)
model11 <- train(Price~.,train_price,#用训练集进行knn算法模型训练
                 method = 'knn',
                 preProcess = c('center','scale'),#knn算法前提是对自变量进行中心化与标准化以消除量纲
                 trControl = control,
                 tuneGrid = grid, distance = "manhattan")#选取距离度量为曼哈顿距离
model11   #查看模型效果
predict11=predict(model11,newdata=test_price)#将模型应用于测试集进行预测

#删除一个变量Draft
price12=price[,-c(5)]
train_price=price12[train,]
test_price=price12[-train,]
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
predict12=predict(model12,newdata=test_price)

#删除变量Draft和GPD.Capita
price13=price[,-c(9,5)]
train_price=price13[train,]
test_price=price13[-train,]
library(caret)
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
set.seed(20)
model13 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "manhattan")
model13
predict13=predict(model13,newdata=test_price)

#删除变量Draft和GPD.Capita和Length
price14=price[,-c(1,9,5)]
train_price=price14[train,]
test_price=price14[-train,]
library(caret)
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
set.seed(20)
model14 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "manhattan")
model14
predict14=predict(model14,newdata=test_price)

x=cbind(1:278,predict11,predict12,predict13,predict14,test_price$Price)
x=x[order(x[,6]),]  #将曼哈顿距离的knn算法模型预测结果与真实值放入x，并将x按照真实值由小到大排序
plot(1:278,x[,6],type = "l",lwd=3,xlab='x',ylab = 'Price')#绘制横坐标为序数的真实值与预测值效果图进行效果查看
legend('topleft',legend = c('y_test','9-dimension','8-dimension','7-dimension','6-dimension'),
       col = c(1,6,4,3,7),lty = 1,lwd=c(3,2.8,2.6,2.4,2.2),
       cex=0.8, inset=0.02, x.intersp=0.5, y.intersp=0.5)
lines(1:278,x[,2],col=6,lwd=2.8)
lines(1:278,x[,3],col=4,lwd=2.6)
lines(1:278,x[,4],col=3,lwd=2.4)
lines(1:278,x[,5],col=7,lwd=2.2)

# knn模型训练——马氏距离
# 计算马氏距离矩阵
mahal <- mahalanobis(train_price[, -1], center = colMeans(train_price[, -1]), cov = cov(train_price[, -1]))
set.seed(20)
model21 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "mahalanobis", mahal = mahal)
model21 #查看模型效果（自变量全保留）
predict21=predict(model21,newdata=test_price)#将模型应用于测试集进行预测

#删除一个变量Beam
price22=price[,-c(4)]
train_price=price22[train,]
test_price=price22[-train,]
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
set.seed(20)
mahal <- mahalanobis(train_price[, -1], center = colMeans(train_price[, -1]), cov = cov(train_price[, -1]))
model22 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "mahalanobis", mahal = mahal)
model22
predict22=predict(model22,newdata=test_price)

#删除变量Beam和GDP.Capita
price23=price[,-c(4,9)]
train_price=price23[train,]
test_price=price23[-train,]
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
mahal <- mahalanobis(train_price[, -1], center = colMeans(train_price[, -1]), cov = cov(train_price[, -1]))
set.seed(20)
model23 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "mahalanobis", mahal = mahal)
model23
predict23=predict(model23,newdata=test_price)

#删除变量Beam和GDP.Capita和Draft
price24=price[,-c(4,5,9)]
train_price=price24[train,]
test_price=price24[-train,]
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
mahal <- mahalanobis(train_price[, -1], center = colMeans(train_price[, -1]), cov = cov(train_price[, -1]))
set.seed(20)
model24 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "mahalanobis", mahal = mahal)
model24
predict24=predict(model24,newdata=test_price)

y=cbind(1:278,predict21,predict22,predict23,predict24,test_price$Price)
y=y[order(y[,6]),]  #将马氏距离的knn算法模型预测结果与真实值放入y，并将y按照真实值由小到大排序
plot(1:278,y[,6],type = "l",lwd=3,xlab='x',ylab = 'Price')#绘制横坐标为序数的真实值与预测值效果图进行效果查看
legend('topleft',legend = c('y_test','9-dimension','8-dimension','7-dimension','6-dimension'),
       col = c(1,6,4,3,7),lty = 1,lwd=c(3,2.8,2.6,2.4,2.2),
       cex=0.8, inset=0.02, x.intersp=0.5, y.intersp=0.5)
lines(1:278,y[,2],col=6,lwd=2.8)
lines(1:278,y[,3],col=4,lwd=2.6)
lines(1:278,y[,4],col=3,lwd=2.4)
lines(1:278,y[,5],col=7,lwd=2.2)


# knn模型训练——欧氏距离
set.seed(20)
model31 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "euclidean")
model31 #查看模型效果（自变量全保留）
predict31=predict(model31,newdata=test_price)#将模型应用于测试集进行预测
#删除一个变量Beam
price32=price[,-c(4)]
train_price=price32[train,]
test_price=price32[-train,]
library(caret)
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
set.seed(20)
model32 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "euclidean")
model32
predict32=predict(model32,newdata=test_price)
#删除变量Beam和Draft
price33=price[,-c(4,5)]
train_price=price33[train,]
test_price=price33[-train,]
library(caret)
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
set.seed(20)
model33 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "euclidean")
model33
predict33=predict(model33,newdata=test_price)
#删除变量Beam和GDP.Capita和Draft
price34=price[,-c(4,5,9)]
train_price=price34[train,]
test_price=price34[-train,]
library(caret)
control <- trainControl(method = 'cv',number = 10)
k_values <- data.frame(k = 2:10)
grid <- expand.grid(.k = k_values)
set.seed(20)
model34 <- train(Price~.,train_price,
                 method = 'knn',
                 preProcess = c('center','scale'),
                 trControl = control,
                 tuneGrid = grid, distance = "euclidean")
model34
predict34=predict(model34,newdata=test_price)

z=cbind(1:278,predict31,predict32,predict33,predict34,test_price$Price)
z=z[order(z[,6]),]  #将欧式距离的knn算法模型预测结果与真实值放入z，并将z按照真实值由小到大排序
plot(1:278,z[,6],type = "l",lwd=3,xlab='x',ylab = 'Price')#绘制横坐标为序数的真实值与预测值效果图进行效果查看
legend('topleft',legend = c('y_test','9-dimension','8-dimension','7-dimension','6-dimension'),
       col = c(1,6,4,3,7),lty = 1,lwd=c(3,2.8,2.6,2.4,2.2),
       cex=0.8, inset=0.02, x.intersp=0.5, y.intersp=0.5)
lines(1:278,z[,2],col=6,lwd=2.8)
lines(1:278,z[,3],col=4,lwd=2.6)
lines(1:278,z[,4],col=3,lwd=2.4)
lines(1:278,z[,5],col=7,lwd=2.2)
#与第一次作业结果对比分析
#经对比发现模型11，31，34，24效果较好，现与第一次作业建立线性模型效果进行绘图对比
# 去除离群值



########## 与广义线性模型对比 ############(代码照搬了, 十分冗长笑死)
remove_outlier = function(col_name, data){
  for(cn in col_name){
    # 洗离群值
    Q1 <- quantile(data[,cn], 0.25)
    Q3 <- quantile(data[,cn], 0.75)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    data <- data[data[,cn]>=lower_bound & data[,cn]<=upper_bound, ]
  }
  return(data)
}

# 读取数据
load_data = function(){
  # 读
  boat_data <- read.csv("Boats.csv")[, c(2:11)]
  print(nrow(boat_data))
  
  # 缺失值
  boat_data <- na.omit(boat_data)
  cat('移除缺失值后', nrow(boat_data), '\n')
  
  # 非数字
  boat_data <- boat_data[is.numeric(boat_data$Price), ]
  cat('移除非数字后', nrow(boat_data), '\n')
  
  # 标准化
  boat_data[, 1:9] <- scale(boat_data[, 1:9])
  
  # 取对
  boat_data$Log_Price <- log(boat_data$Price)
  
  # 移除Log Price离群值
  boxplot(boat_data$Log_Price, main="Log_Price Boxplot", ylab="Log_Price")
  boat_data = remove_outlier(c('Log_Price'), boat_data)
  cat('移除Log_Price离群值后', nrow(boat_data), '\n')
  
  return(boat_data)
}
boat_data = load_data()
boat_data$Year2 <- boat_data$Year^2
boat_data$Draft2 <- boat_data$Draft^2
boat_data$LenLWL = 0.108500 * boat_data$Length + 0.038798 * boat_data$LWL
boat_data$Draft12 = -0.008648 * boat_data$Draft + 0.007299 * boat_data$Draft2
boat_data$Year12 = 0.218969 * boat_data$Year + 0.030731 * boat_data$Year2
boat_data$Length = NULL
boat_data$LWL = NULL
boat_data$Draft = NULL
boat_data$Draft2 = NULL
boat_data$Year = NULL
boat_data$Year2 = NULL
boat_data[, 8:10] <- scale(boat_data[, 8:10])

# 合并变量过后的
lr_model2 <- lm(Log_Price ~ 1 + 
                  LenLWL +
                  Year12+
                  Beam +
                  Draft12+
                  Displacement + 
                  SailArea + 
                  GDP + 
                  GDP.Capita, 
                data = boat_data)

summary(lr_model2)

price=read.csv("Boats.csv",fileEncoding = 'gbk',header=T)
price=price[c(1:961),c(2:11)]  #ID对建模无助，故删掉第一列
pic=boxplot(price$Price)
low=pic$stats[1]
up=pic$stats[5]
price=price[price$Price>=low&price$Price<=up,]

test_price <- na.omit(price)
test_price[, 1:9] <- scale(test_price[, 1:9])
test_data=test_price[-train,]
test_data[, 1:9] <- scale(test_data[, 1:9])
test_data$Log_Price <- log(test_data$Price)
test_data$Year2 <- test_data$Year^2
test_data$Draft2 <- test_data$Draft^2
test_data$LenLWL = 0.108500 * test_data$Length + 0.038798 * test_data$LWL
test_data$Draft12 = -0.008648 * test_data$Draft + 0.007299 * test_data$Draft2
test_data$Year12 = 0.218969 * test_data$Year + 0.030731 * test_data$Year2
test_data$Length = NULL
test_data$LWL = NULL
test_data$Draft = NULL
test_data$Draft2 = NULL
test_data$Year = NULL
test_data$Year2 = NULL
test_data[, 8:10] <- scale(test_data[, 8:10])

lr_test_pred = predict(lr_model2, newdata=test_data)
lr_test_pred
predict_lr2=exp(lr_test_pred)
predict_lr2

### 绘制 11, 31, 34, 24, 广义 的对比图 ### 

test_price=price[-train,]
z=cbind(1:278,predict11,predict31,predict34,predict24,predict_lr2,test_price$Price)
z=z[order(z[,7]),]  #将欧式距离的knn算法模型预测结果与真实值放入z，并将z按照真实值由小到大排序
plot(1:278,z[,7],type = "l",lwd=3,xlab='x',ylab = 'Price',main = '效果对比图')#绘制横坐标为序数的真实值与预测值效果图进行效果查看
legend('topleft',legend = c('y_test','model11','model31','model34','model24','lr_model2'),
       col = c(1,6,4,3,7,21),lty = 1,lwd=c(3,2.8,2.6,2.4,2.2,2.0),
       cex=0.8, inset=0.02, x.intersp=0.5, y.intersp=0.5)
lines(1:278,z[,2],col=6,lwd=2.8)
lines(1:278,z[,3],col=4,lwd=2.6)
lines(1:278,z[,4],col=3,lwd=2.4)
lines(1:278,z[,5],col=7,lwd=2.2)
lines(1:278,z[,6],col=21,lwd=1.9)

