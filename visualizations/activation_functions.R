library(ggplot2)
library(pracma)

xGrid = seq(-10,10, by = 0.005) ; xGrid

binary_step = function(x){
  if (x<0) 0 else 1
}

logistic = function(x){
  return(1/( 1+exp(-x) ))
}

tanh = function(x){
  num = exp(x) - exp(-x)
  den = exp(x) + exp(-x)
  return(num/den)
}

RELU = function(x){
  if (x<=0) 0 else x 
}

leaky_RELU = function(x){
  if(x<0) 0.01*x else x
}

PRELU = function(x){
  if(x<0) 0.2*x else x
}


GELU = function(x){
  0.5*x*(1+erf(x/sqrt(2)))
}


ELU = function(x){
  if (x<0) 0.2*(exp(x) - 1) else x
}


SELU = function(x){
  if (x<0) 1.0507*1.67326*(exp(x)-1) else 1.0507*x
}

# plot --------------------------------------------------------------------


par(mfrow=c(2,2),oma=c(0,0,3,0),bg="whitesmoke")
plot(xGrid, sapply(xGrid, binary_step), type="l",col="coral4",  panel.first = grid(25,25), lwd=3, ylab = "", xlab="")
plot(xGrid, sapply(xGrid, logistic), type="l",col="coral4",  panel.first = grid(25,25), lwd=3, ylab="", xlab= "")
plot(xGrid, sapply(xGrid, tanh), type="l",col="coral4",  panel.first = grid(25,25), lwd=3, ylab="", xlab ="")
plot(xGrid, sapply(xGrid, RELU), type="l",col="coral4",  panel.first = grid(25,25), lwd=3, ylab="", xlab= "")



# ggplot ------------------------------------------------------------------

# binary step 

df = data.frame(xGrid, sapply(xGrid, binary_step));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) + 
  geom_hline(yintercept = 0.5, color = "black", size=0.6, linetype="dashed") +
  labs(title="Binary step", x ="", y = "") +
  theme(plot.title = element_text(hjust = 0.5)) 

# logistic 

df = data.frame(xGrid, sapply(xGrid, logistic));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) + 
  geom_hline(yintercept=0, color = "black", size=0.6, linetype="dashed") +  
  geom_vline(xintercept = 0, color = "black", size=0.6, linetype="dashed") + 
  labs(title="Logistic", x ="", y = "") + 
  theme(plot.title = element_text(hjust = 0.5)) 

# tanh

df = data.frame(xGrid, sapply(xGrid, tanh));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) + 
  geom_hline(yintercept=0, color = "black", size=0.6, linetype="dashed") + 
  geom_vline(xintercept = 0, color = "black", size=0.6, linetype="dashed") + 
  labs(title="Tanh", x ="", y = "") + 
  theme(plot.title = element_text(hjust = 0.5)) 

# RELU 

df = data.frame(xGrid, sapply(xGrid, RELU));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) + 
  geom_vline(xintercept=0, color = "black", size=0.6, linetype="dashed") + 
  labs(title="RELU", x ="", y = "") +  
  theme(plot.title = element_text(hjust = 0.5)) 

# leaky RELU 

df = data.frame(xGrid, sapply(xGrid, leaky_RELU));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) +
  geom_vline(xintercept=0, color = "black", size=0.6, linetype="dashed") + 
  geom_hline(yintercept = 0, color = "black", size=0.6, linetype="dashed") + 
  labs(title="Leaky RELU", x ="", y = "") + 
  theme(plot.title = element_text(hjust = 0.5)) 


# PRELU 

df = data.frame(xGrid, sapply(xGrid, PRELU));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) + 
  geom_vline(xintercept=0, color = "black", size=0.6, linetype="dashed") + 
  geom_hline(yintercept = 0, color = "black", size=0.6, linetype="dashed") + 
  labs(title="PRELU", x ="", y = "") +
  theme(plot.title = element_text(hjust = 0.5)) 


# GELU 

df = data.frame(xGrid, sapply(xGrid, GELU));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) + 
  geom_vline(xintercept=0, color = "black", size=0.6, linetype="dashed") + 
  geom_hline(yintercept = 0, color = "black", size=0.6, linetype="dashed") + 
  labs(title="GELU", x ="", y = "") + 
  theme(plot.title = element_text(hjust = 0.5)) 

# ELU 

df = data.frame(xGrid, sapply(xGrid, ELU));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) + 
  geom_vline(xintercept=0, color = "black", size=0.6, linetype="dashed") +
  geom_hline(yintercept = 0, color = "black", size=0.6, linetype="dashed") + 
  labs(title="ELU", x ="", y = "") +  
  theme(plot.title = element_text(hjust = 0.5)) 

# SELU 

df = data.frame(xGrid, sapply(xGrid, SELU));colnames(df) = c("x", "y");  head(df)

ggplot(df, aes(x,y) ) + 
  geom_line(col="coral4", size=1) + 
  geom_vline(xintercept=0, color = "black", size=0.6, linetype="dashed") + 
  geom_hline(yintercept = 0, color = "black", size=0.6, linetype="dashed") + 
  labs(title="SELU", x ="", y = "") + 
  theme(plot.title = element_text(hjust = 0.5)) 
      






  
       