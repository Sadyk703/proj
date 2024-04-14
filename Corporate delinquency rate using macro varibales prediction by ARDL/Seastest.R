#install.packages("seastests")
library("readxl")
library("forecast")
library("seastests")

data <- read_excel("C:/Users/BelyaevMA/Documents/Macro_research/MacroDS/data/v3/macro_ds_m_f.xlsx")
data <- data[-c(0, 1, 169, 170, 171, 172), ]

head(data['OVR_DEBT'])
head(data)
tail(data)

length(data)

data[length(data)]
colnames(data)[6] 

df_key <- ts(data[6], start=c(2008, 2), end=c(2022, 3), frequency=12)
df_IMOEX <- ts(data[7], start=c(2008, 2), end=c(2022, 3), frequency=12)
df_RTSI <- ts(data[8], start=c(2008, 2), end=c(2022, 3), frequency=12)
df_USDRUB <- ts(data[9], start=c(2008, 2), end=c(2022, 3), frequency=12)
df_EURRUB <- ts(data[10], start=c(2008, 2), end=c(2022, 3), frequency=12)
df_BRENT <- ts(data[11], start=c(2008, 2), end=c(2022, 3), frequency=12)
df_RUSGAS <- ts(data[12], start=c(2008, 2), end=c(2022, 3), frequency=12)
df_PPI <- ts(data[13], start=c(2008, 2), end=c(2022, 2), frequency=12)
df_LIAB <- ts(data[14], start=c(2008, 2), end=c(2022, 1), frequency=12)
df_DBT <- ts(data[15], start=c(2008, 2), end=c(2022, 1), frequency=12)
df_EX <- ts(data[16], start=c(2008, 2), end=c(2022, 1), frequency=12)
df_IM <- ts(data[17], start=c(2008, 2), end=c(2022, 1), frequency=12)
df_OVR <- ts(data[18], start=c(2008, 2), end=c(2022, 2), frequency=12)


isSeasonal(df_key, test = "combined", freq = NA)
isSeasonal(df_key, test = "combined", freq = 12)


isSeasonal(data[6], test = "combined", freq = 12)

length(data)

########
for (val in 6: length(data))
{
  print(colnames(data)[val])
  print(isSeasonal(ts(data[val], start=c(2008, 2), end=c(2022, 3), frequency=12), test = "combined", freq = NA) )
  cat('\n')
}