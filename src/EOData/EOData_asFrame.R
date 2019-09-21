getEOMatrix_Traps <- function(){

	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))

	con<-dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")

	df <- dbGetQuery(con, "
		SELECT c.*
		FROM combined.traps_eo c
	") %>%
	mutate(forest_presence = factor(1 - (forest_presence - 1))) %>%
	mutate(eastness = cos(aspect)) %>%
	mutate(northness = sin(aspect))

	dbDisconnect(con)

	return(df)
}

getEOMatrix_Plots <- function(){

	suppressMessages(library(RPostgreSQL))
	suppressMessages(library(tidyverse))

	con<-dbConnect(dbDriver("PostgreSQL"), dbname="GPNPdata", host="localhost", port=5432, user="giezi",password="obviousPassword")

	df <- dbGetQuery(con, "
		SELECT c.*
		FROM combined.plots_eo c
	") %>%
	mutate(forest_presence = factor(1 - (forest_presence - 1))) %>%
	mutate(eastness = cos(aspect)) %>%
	mutate(northness = sin(aspect))

	dbDisconnect(con)

	return(df)
}