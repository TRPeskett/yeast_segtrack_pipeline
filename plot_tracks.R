library("tidyverse")
library("cowplot")

# Point these at the run you want to plot, e.g. "output/2024-06-19_10-32-19",
# or set SEGTRACK_RUN in the environment before starting R.
run.dir <- Sys.getenv("SEGTRACK_RUN", unset = "output/<timestamp>")
data.dir <- run.dir
output.dir <- run.dir

pdir <- function(dirname, filename) {
  paste(dirname, filename, sep = "/")
}

# read the segmentation/tracking data
tracks <- read.csv(pdir(data.dir, "all_traps_summary.csv")) %>%
  
  # add some useful features to the df
  mutate(trap_gID = paste(trap_nb, Global_id, sep = '_')) %>%
  group_by(Global_id) %>%
  mutate(track_length = length(frame)) %>%
  ungroup()

# plot track lengths
tracks %>%
  ggplot(aes(x = track_length)) +
  geom_histogram()

# plot track lengths for different Types (mothers, trapped non-mothers, outsiders)
tracks %>%
  group_by(Global_id) %>%
  ggplot(aes(x = track_length)) +
  geom_histogram() +
  facet_wrap(~Type)

# Most outsiders have track lengths < 160 so seems to make sense to threshold here to increase chances of analysing well tracked mother cells.


# plot dispersion index for individual mother cells
tracks %>%
  filter(fluor_max >= 0, Type == 'mother', track_length > 160) %>%
  ggplot(aes(x = frame, y = (fluor_std^2 / fluor_avg))) +
  geom_line() +
  facet_wrap(~trap_gID)

# plot dispersion index for all mothers in one plot
tracks %>%
  filter(fluor_max >= 0, Type == 'mother', track_length > 160) %>%
  ggplot(aes(x = frame, y = (fluor_std^2 / fluor_avg), colour = trap_gID)) +
  geom_line() +
  theme_cowplot()

# plot average fluorescence for all mothers in one plot - here you can see we probably need background subtraction and flicker correction
# but it's also interesting that the population becomes more heterogeneous as it ages. Would be cool to test if this is general and whether
# it could lead to a competitive advantage.
tracks %>%
  filter(fluor_max >= 0, Type == 'mother', track_length > 160) %>%
  ggplot(aes(x = frame, y = (fluor_avg), colour = trap_gID)) +
  geom_line() +
  theme_cowplot()

# look at the centroids of the tracks
tracks %>%
  filter(Type == 'mother') %>%
  ggplot(aes(x = Centroid_x, y = Centroid_y, colour = frame)) +
  geom_point(alpha = .2) +
  scale_colour_viridis_c() +
  geom_density2d(colour = "black") +
  theme_cowplot()

tracks %>%
  filter(Type == 'mother') %>%
  ggplot(aes(x = Centroid_x, y = Centroid_y, colour = frame)) +
  geom_density2d()

tracks %>%
  ggplot(aes(x = Centroid_x, y = Centroid_y, colour = Type)) +
  geom_point(alpha = .2) +
  scale_colour_viridis_d() +
  theme_cowplot()

# Is there a way we can get an idea about age?
# Daughter cells are probably going to move in the x direction as they fly out of the traps
# Looking at overview it seems that the daughters are not tracked well e.g. often two different daughters are assigned the same ID

tracks %>%
  filter(trap_nb == 27) %>%
  ggplot(aes(x = frame, y = Centroid_x, colour = Type, group = Global_id)) +
  geom_line()

tracks %>%
  filter(trap_nb == 27) %>%
  ggplot(aes(x = Centroid_x, y = Centroid_y, colour = Type, group = Global_id)) +
  geom_line() +
  scale_colour_viridis_d()












