install.packages("tidyverse")
options(repos = c(CRAN = "https://cloud.r-project.org"))

library(tidyverse)
library(data.table)

setwd("/Users/georgeflint/Desktop/keys/data/ratings/processed")
getwd()

process_ratings <- function(file_path, output_file) {
  dat <- read.csv(file_path)

  # Extract ratings
  dat.ratings <- dat %>%
    select(ResponseId, matches('^X', ignore.case = FALSE)) %>%
    filter(str_detect(ResponseId, 'R_'))

  # Extract adjective-ID matchings
  dat.adjective_ids <- dat %>%
    filter(ResponseId == "Response ID") %>%
    select(matches("Object")) %>%
    transpose(keep.names = "AdjectiveID") %>%
    rename(Adjective = V1) %>%
    separate(AdjectiveID, "AdjectiveID", sep = "_") %>%
    separate(Adjective, "Adjective", sep = " - ")

  # Reshape and cast to numeric
  dat.reshaped <- dat.ratings %>%
    gather(key = "RatingType", value = "Rating", -ResponseId) %>%
    mutate(Rating = as.numeric(Rating)) %>%
    separate(RatingType, into = c("AdjectiveID", "Question"), sep = "_") %>%
    spread(key = Question, value = Rating)

  # Calculate statistics
  dat.by_adjective <- dat.reshaped %>%
    group_by(AdjectiveID) %>%
    summarize(
      meanAssociation = mean(Gender.Association, na.rm = TRUE),
      medianAssociation = median(Gender.Association, na.rm = TRUE),
      # Add other boxplot stats for association
      meanObject = mean(Describe.an.Object, na.rm = TRUE),
      medianObject = median(Describe.an.Object, na.rm = TRUE),
      # Add other boxplot stats for object rating
      meanPerson = mean(Describe.a.Person, na.rm = TRUE),
      medianPerson = median(Describe.a.Person, na.rm = TRUE),
      # Add other boxplot stats for person rating
      percentKnown = sum(!is.na(Gender.Association)) / n() * 100
    ) %>%
    merge(dat.adjective_ids)

  # Save output
  write.csv(dat.by_adjective, output_file, row.names = FALSE)
}

# File paths
files <- c('en_adjective_ratings_unprocessed.csv', 'es_adjective_ratings_unprocessed.csv', 'de_adjective_ratings_unprocessed.csv')

# Process each file
for (file in files) {
  language <- str_extract(file, "^[a-z]{2}")
  output_file <- paste0(language, "_adjective_ratings.csv")
  process_ratings(file, output_file)
}
