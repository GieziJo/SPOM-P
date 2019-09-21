getObservedSpeciesRaster <- function(speciesName, year){
	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))

	con<-dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")

	dbSendQuery(con,str_c("
		--- drop table if already existing
		drop view eo.tmp_species_raster;
		create view eo.tmp_species_raster as
		select
		--- set values
		ST_SetValues(
			--- add a band
			ST_AddBand(
				--- create empty raster with existing raster as reference
				ST_MakeEmptyRaster(w.rast),
				--- set band default values to 99
				1, '8BUI', 99, 99
			),
			1,
			(
				SELECT ARRAY(
					--- fill values where needed
					SELECT (c.geom,c.observed)::geomval
					FROM insitu.carabids_plots_observed_sampled c
					where c.species = '", speciesName, "'
					and c.year = ", year,"
				)
			)
		) as rast

		from eo.temperature
	"))

	suppressMessages(library(rpostgis))
	suppressMessages(library(raster))
	source("generalMethods/rasterInfo.R")

	cols <- getRasterCols()

	data <- matrix(getValues(pgGetRast(con, name = c("eo","tmp_species_raster"), rast = "rast", bands = 1, boundary = NULL)), ncol = cols, byrow = TRUE)
	data[data == 99] = NaN

	dbDisconnect(con)

	return(data)
}

getAllObservedSpeciesMatrix <- function(speciesName){
	source("generalMethods/rasterInfo.R")
	dim <- getRasterRowsCols()
	observedMatrix <- array(data = NA, dim = c(dim[1],dim[2],4))
	years = c(2006,2007,2012,2013)
	for (year_i in 1:4){
		observedMatrix[,,year_i] <- getObservedSpeciesRaster(speciesName, years[year_i])
	}
	return(observedMatrix)
}

getSampledSpeciesRaster <- function(speciesName, year){
	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))

	con<-dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")
	dbSendQuery(con,str_c("
		--- drop table if already existing
		drop view eo.tmp_species_raster;
		create view eo.tmp_species_raster as
		select
		--- set values
		ST_SetValues(
			--- add a band
			ST_AddBand(
				--- create empty raster with existing raster as reference
				ST_MakeEmptyRaster(w.rast),
				--- set default values to 99
				1, '8BUI', 99, 99
			),
			1,
			(
				SELECT ARRAY(
					--- fill values where needed
					SELECT (c.geom,c.sampled)::geomval
					FROM insitu.carabids_plots_observed_sampled c
					where c.species = '", speciesName, "'
					and c.year = ", year,"
				)
			)
		) as rast

		from eo.temperature
	"))

	suppressMessages(library(rpostgis))
	suppressMessages(library(raster))
	source("generalMethods/rasterInfo.R")

	cols <- getRasterCols()

	data <- matrix(getValues(pgGetRast(con, name = c("eo","tmp_species_raster"), rast = "rast", bands = 1, boundary = NULL)), ncol = cols, byrow = TRUE)
	data[data == 99] = NaN

	dbDisconnect(con)
	
	return(data)
}

getAllSampledSpeciesMatrix <- function(speciesName){
	source("generalMethods/rasterInfo.R")
	dim <- getRasterRowsCols()
	observedMatrix <- array(data = NA, dim = c(dim[1],dim[2],4))
	years = c(2006,2007,2012,2013)
	for (year_i in 1:4){
		observedMatrix[,,year_i] <- getSampledSpeciesRaster(speciesName, years[year_i])
	}
	return(observedMatrix)
}