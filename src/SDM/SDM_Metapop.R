getInitialPresenceGuessSDM <- function(speciesName){
	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))
	suppressMessages(library(rpostgis))
	suppressMessages(library(raster))

	con <- dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")

	df <- dbGetQuery(con, str_c("
		SELECT c.*
		FROM combined.carabids_plots_eo c
		where c.species = '", speciesName, "'
		and c.year = 2006
	")) %>%
	mutate(failure = sampled-observed) %>%
	mutate(present = observed>0) %>%
	mutate(forest_presence = factor(1 - (forest_presence - 1))) %>%
	mutate(valley = factor(valley,levels=c("gran_piano","lauson","orvieilles","san_besso","vaudalettaz","outside"))) %>%
	mutate(eastness = cos(aspect)) %>%
	mutate(northness = sin(aspect))

	glm_out <- glm(
		cbind(observed,failure) ~ 
		1+ 
		temperature + 
		wetness + 
		brightness + 
		greenness + 
		northness + 
		eastness + 
		slope + 
		forest_presence
		,family=binomial (logit), data=df)

	year = 2006

	valley <- getValues(pgGetRast(con, c("eo","valleys"), rast = "rast", bands = 1, boundary = NULL))


	temperature <- getValues(pgGetRast(con, c("eo", "temperature"), rast = "rast", bands = year-2005, boundary = NULL))
	wetness <- getValues(pgGetRast(con, c("eo", "wetness"), rast = "rast", bands = year-2005, boundary = NULL))
	brightness <- getValues(pgGetRast(con, c("eo", "brightness"), rast = "rast", bands = year-2005, boundary = NULL))
	greenness <- getValues(pgGetRast(con, c("eo", "greenness"), rast = "rast", bands = year-2005, boundary = NULL))
	aspect <- getValues(pgGetRast(con, c("eo", "aspect"), rast = "rast", bands = 1, boundary = NULL))
	eastness <- cos(aspect)
	northness <- sin(aspect)
	slope <- getValues(pgGetRast(con, c("eo", "slope"), rast = "rast", bands = 1, boundary = NULL))
	forest_presence <- factor(2 - getValues(pgGetRast(con, c("eo", "forest_presence"), rast = "rast", bands = 1, boundary = NULL)))

	
	dbDisconnect(con)

	ind_valley <- valley == 6
	valley[ind_valley] <- 1

	valley <- replace(valley, valley==1, "gran_piano")
	valley <- replace(valley, valley==2, "lauson")
	valley <- replace(valley, valley==3, "orvieilles")
	valley <- replace(valley, valley==4, "san_besso")
	valley <- replace(valley, valley==5, "vaudalettaz")
	valley <- replace(valley, valley==6, "outside")

	val <- setNames(aggregate(df$present, by=list(df$valley), FUN=sum), c("valley", "present"))
	val$present <- val$present > 0

	presenceMatrix <- rep(FALSE, length(valley))
	for(v_i in 1:nrow(val)){
		presenceMatrix[valley == val[v_i,]$valley] <- val[v_i,]$present
	}
	presenceMatrix[ind_valley] <- FALSE
	source("generalMethods/rasterInfo.R")
	presenceMatrix <- matrix(presenceMatrix,ncol = getRasterCols(), byrow = TRUE)

	valley <- factor(valley,levels=c("gran_piano","lauson","orvieilles","san_besso","vaudalettaz","outside"))

	df_EO <- data.frame(temperature, wetness, brightness, greenness, eastness, northness, slope, forest_presence, valley)

	p <- predict(glm_out, df_EO, type="response")

	P <- matrix(p, ncol = getRasterCols(), byrow = TRUE) * presenceMatrix

	return(P)
}