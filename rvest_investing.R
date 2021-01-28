library(rvest)
library(stringr)
library(lubridate)
options(digits = 5, scipen = 100) # Setting max to five significant digits
rm(list = ls()) # Clears the environment (R objects)
graphics.off() # Clears Graphics/Plots
cat("\014") # Clears the console

V40A <-
  "https://www.investing.com/etfs/v20a-historical-data"
V40A_html <- read_html(V40A)

css_date <- paste0('tr:nth-child(3) .noWrap')
V40A_data_new <- V40A_html %>% 
  html_node(css_date) %>%
  html_text() %>%
  mdy() %>%
  toString()

css_value <- paste0('#curr_table tr:nth-child(3) :nth-child(2)')
V40A_valor_new <- V40A_html %>% 
  html_node(css_value) %>%
  html_text() %>%
  as.numeric()

values <- cbind(V40A_data_new, V40A_valor_new)

for (i in seq(4, 21)){
  css_date <- paste0('tr:nth-child(', toString(i), ') .noWrap')
  V40A_data_new <- V40A_html %>% 
    html_node(css_date) %>%
    html_text() %>%
    mdy() %>%
    toString()
  
  css_value <- paste0('#curr_table tr:nth-child(', toString(i), ') :nth-child(2)')
  V40A_valor_new <- V40A_html %>% 
    html_node(css_value) %>%
    html_text() %>%
    as.numeric()
  
  value <- cbind(V40A_data_new, V40A_valor_new)
  values <-  rbind(values, value)
  
}

URLs <- c(
  'https://www.investing.com/etfs/v40a-historical-data',
  'https://www.investing.com/etfs/v60a-historical-data',
  'https://www.investing.com/etfs/v80a-historical-data'
          )

Nomes <- c(
  'Dates',
  'VA20',
  'VA40',
  'VA60',
  'VA80'
)

for (URL in URLs){
  df <- data.frame(Date=as.Date(character()),
                   File=character(), 
                   User=character(), 
                   stringsAsFactors=FALSE) 
  
  for (i in seq(3, 21)){
    css_date <- paste0('tr:nth-child(', toString(i), ') .noWrap')
    
    URL_html <- read_html(URL)
    
    css_value <- paste0('#curr_table tr:nth-child(', toString(i), ') :nth-child(2)')
    URL_valor_new <- URL_html %>% 
      html_node(css_value) %>%
      html_text() %>%
      as.numeric()
    
    df <-  rbind(df, URL_valor_new)
  }

  values <- cbind(values, df)
}

values <- data.frame(values)

colnames(values) <- paste(Nomes)

write.csv(values,'D:/GDrive/_GitHub/Articles_and_studies/values.csv', row.names = FALSE)

