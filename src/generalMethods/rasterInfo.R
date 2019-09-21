getRasterRowsCols <- function(){
	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))

	con<-dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")
	
	df <- dbGetQuery(con, str_c("
		SELECT 
		(MAX(ST_UpperLeftX(rast) + (ST_Width(rast) * ST_PixelWidth(rast))) - 
		MIN(ST_UpperLeftX(rast))) / MAX(ST_PixelWidth(rast)) AS num_cols,
		(MAX(ST_UpperLeftY(rast)) - 
		MIN(ST_UpperLeftY(rast) - (ST_Height(rast) * ST_PixelHeight(rast)))) / 
		MAX(ST_PixelHeight(rast)) AS num_rows
		FROM eo.temperature;
	"))
	
	dbDisconnect(con)
	
	return(c(df$num_rows, df$num_cols))
}

getRasterRows <- function(){
	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))

	con<-dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")
	
	df <- dbGetQuery(con, str_c("
		SELECT 
		(MAX(ST_UpperLeftY(rast)) - 
		MIN(ST_UpperLeftY(rast) - (ST_Height(rast) * ST_PixelHeight(rast)))) / 
		MAX(ST_PixelHeight(rast)) AS num_rows
		FROM eo.temperature;
	"))
	
	dbDisconnect(con)
	
	return(df$num_rows)
}

getRasterCols <- function(){
	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))

	con<-dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")
	
	df <- dbGetQuery(con, str_c("
		SELECT 
		(MAX(ST_UpperLeftX(rast) + (ST_Width(rast) * ST_PixelWidth(rast))) - 
		MIN(ST_UpperLeftX(rast))) / MAX(ST_PixelWidth(rast)) AS num_cols
		FROM eo.temperature;
	"))
	
	dbDisconnect(con)
	
	return(df$num_cols)
}