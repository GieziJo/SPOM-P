getEOMatrix_Metapop <- function(){

	suppressMessages(library(rpostgis))
	suppressMessages(library(raster))
	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))
	source("generalMethods/rasterInfo.R")

	con<-dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")

	dim <- getRasterRowsCols()
	cols = dim[2]

	EOMatrix <- array(data = NA, dim = c(dim[1],dim[2],8,8))


	aspect <- matrix(getValues(pgGetRast(con, c("eo", "aspect"), rast = "rast", bands = 1, boundary = NULL)), ncol = cols, byrow = TRUE)
	eastness = cos(aspect)
	northness = sin(aspect)
	slope <- matrix(getValues(pgGetRast(con, c("eo",v"slope"), rast = "rast", bands = 1, boundary = NULL)), ncol = cols, byrow = TRUE)
	forest_presence <- matrix(2 - getValues(pgGetRast(con, c("eo", "forest_presence"), rast = "rast", bands = 1, boundary = NULL)), ncol = cols, byrow = TRUE)

	for (year_i in 2006:2013){
		temperature <- matrix(getValues(pgGetRast(con, c("eo", "temperature"), rast = "rast", bands = year_i-2005, boundary = NULL)), ncol = cols, byrow = TRUE)
		wetness <- matrix(getValues(pgGetRast(con, c("eo", "wetness"), rast = "rast", bands = year_i-2005, boundary = NULL)), ncol = cols, byrow = TRUE)
		brightness <- matrix(getValues(pgGetRast(con, c("eo", "brightness"), rast = "rast", bands = year_i-2005, boundary = NULL)), ncol = cols, byrow = TRUE)
		greenness <- matrix(getValues(pgGetRast(con, c("eo", "greenness"), rast = "rast", bands = year_i-2005, boundary = NULL)), ncol = cols, byrow = TRUE)
							
		EOMatrix[,,year_i - 2005,1] <- temperature
		EOMatrix[,,year_i - 2005,2] <- wetness
		EOMatrix[,,year_i - 2005,3] <- brightness
		EOMatrix[,,year_i - 2005,4] <- greenness
		EOMatrix[,,year_i - 2005,5] <- eastness
		EOMatrix[,,year_i - 2005,6] <- northness
		EOMatrix[,,year_i - 2005,7] <- slope
		EOMatrix[,,year_i - 2005,8] <- forest_presence
	}

	dbDisconnect(con)

	return(EOMatrix)
}