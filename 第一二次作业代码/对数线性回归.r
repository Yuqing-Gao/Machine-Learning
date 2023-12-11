# version: R 4.2.3

data <- read.csv("Boats004.csv")

# 检查是否存在缺失值和非数字值，是的话删去所在行
sum(is.na(data$price))
sum(!is.numeric(data$price))
data <- na.omit(data)
data <- data[is.numeric(data$price), ]

# 取对
data$log_price <- log(data$price)

# 绘制价格的箱线图
boxplot(data$log_price, main="Log-Price Boxplot", ylab="Log-Price")

# 洗离群值
Q1 <- quantile(data$log_price, 0.25)
Q3 <- quantile(data$log_price, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
data_clean <- data[data$log_price >= lower_bound & data$log_price <= upper_bound, ]
# 查看洗完效果↓
# boxplot(data_clean$log_price, main="Log-Price Boxplot", ylab="Log-Price")

# 直接拟合获得model
model <- lm(log_price ~ 1 + length + year + lwl + beam + draft + displacement + sailarea + gdp + gdppc, data=data_clean)
summary(model)

# 把year全都减了2005获得model2，但好像差别不大，之后暂且只用model
# data_clean$offset_year <- data_clean$year - 2005
# model2 <- lm(log_price ~ 1 + length + offset_year + lwl + beam + draft + displacement + sailarea + gdp + gdppc, data=data_clean)
# summary(model2)

# 逐步回归
model.aic <- step(model, direction = "both")

# 多重共线性诊断，剔除length
library(car)
vif(model)

# 重新建模
model1 <-lm(log_price ~ 1 + year + lwl + beam + draft + displacement + sailarea + gdp + gdppc, data=data_clean)
# Multiple R-squared:  0.7813,	Adjusted R-squared:  0.7794 
summary(model1)

# 整体诊断
par(mfrow = c(2, 2))
plot(model1)

# 残差正态性检验
ks.test(scale(model1$residuals), "pnorm")

# 异方差检验
ncvTest(model1, method = "response")

# 多重共线性诊断
vif(model1)

# 错误率
y_true <- data_clean$price
y_pred <- predict(model1, newdata = data_clean)
y_pred_exp <- exp(y_pred)
r <- mean(abs((y_true - y_pred_exp) / y_true))


# ··························以下内容鉴定为：没有必要························

# 在model1基础上剔除draft和sailarea建立model3
# model3 <-lm(log_price ~ 1 + year + lwl + beam + displacement + gdp + gdppc, data=data_clean)
# Multiple R-squared:  0.7772,	Adjusted R-squared:  0.7758 
# summary(model3)

# 整体诊断
# par(mfrow = c(2, 2))
# plot(model3)

# 标准化残差正态性检验 通过
# ks.test(scale(model3$residuals), "pnorm")

# 异方差检验 通过
# ncvTest(model3, method = "response")

# 多重共线性诊断 通过
# vif(model3)

# 错误率(比model1低了百分之零点几就)
# y_true_3 <- data_clean$price
# y_pred_3 <- predict(model3, newdata = data_clean)
# y_pred_exp_3 <- exp(y_pred_3)
# r_3 <- mean(abs((y_true_3 - y_pred_exp_3) / y_true_3))

