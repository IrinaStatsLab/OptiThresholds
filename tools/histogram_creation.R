# Make sure that the directory is set to the root of this project
if (!dir.exists("data")) {
  setwd("..")
}

library(dplyr)
library(ggplot2)
library(grid)
library(gridExtra)

# Load and process data
data <- read.csv("./data/brown2019_filtered.csv")
grouped_data <- data %>% group_by(id) %>% summarise(gl = list(gl))
data_list <- grouped_data$gl

# Pick a subject to draw a histogram of CGM values
x = data_list[[12]]

df <- data.frame(x = x)

# First histogram with frequent breaks
p1 <- ggplot(df, aes(x = x)) +
  geom_histogram(
    breaks = seq(39, 401, 1),
    aes(y = after_stat(density)), 
    alpha = 0.9, size = 0.1,
    color = "black", fill = "darkgray"
  ) +
  geom_vline(xintercept = c(54, 70, 181, 251), linetype = "dashed", 
             color = "darkgreen", size=0.4, alpha=0.9) +
  scale_y_continuous(expand = c(0, 0.00001)) + 
  ggtitle("Histograms for CGM Data") +
  xlab("Glucose level (mg/dL)") +
  ylab("Density") +
  theme_classic() + 
  theme(
    plot.title = element_text(hjust = 0.5, size = 15, face="bold", margin = margin(b=5.5)),
    panel.grid.major = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size=10),
    legend.position = "none" # or "right"
  )

# Second histogram with conventional thresholds
p2 <- ggplot(df, aes(x = x)) +
  geom_histogram(
    breaks = c(39, 54, 70, 181, 251, 401),
    aes(y = after_stat(density)),
    alpha = 0.9, size = 0.4,
    color = "black", fill = "darkgray"
  ) +
  scale_y_continuous(expand = c(0, 0.00001)) + 
  ggtitle("Summary with Consensus Thresholds") +
  xlab("Glucose level (mg/dL)") +
  ylab("Density") +
  theme_classic() +
  theme(
    # plot.title = element_text(hjust = 0.5, size = 16, face="bold"),
    plot.title = element_blank(),
    panel.grid.major = element_blank(),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size=10),
    axis.text.y = element_text(size=10),
    legend.position = "none" # or "right"
  )

# Arrange the two plots vertically
grid.arrange(p1, p2, ncol = 1)
grid_plot <- arrangeGrob(p1, p2, ncol=1)
grid.draw(grid_plot)

# ggsave("./images/histograms-t1d-summary3.pdf", grid_plot, 
#        width = 5.1, height = 6, dpi=500)


#######################################
### Histogram-quantile plot
#######################################

library(iglu)

# load the data
data <- example_data_5_subject

# Extract gl from the grouped data
grouped_data <- data %>% group_by(id) %>% summarise(gl = list(gl))
data_list <- grouped_data$gl

x1 = data_list[[5]]
dt <- data.frame(x = x1)
dt1 <- dt %>%
  mutate(
    bin_group = cut(x, breaks = c(40, 94.4, 150.5, 200.4, 245, 297.8, 400), 
                    labels = FALSE, include.lowest = TRUE),
    is_highlight = bin_group %in% 4:5  # Highlight bins 4 and 5
  )

# Plot the histogram with highlighted bins
p1 <- ggplot(dt1, aes(x = x, fill = is_highlight)) + 
  geom_histogram(breaks = c(40, 94.4, 150.5, 200.4, 245, 297.8, 400), 
                 color = "black", aes(y = ..density..),  
                 alpha = 0.9, size = 0.4) +     # color darkgreen also works
  scale_fill_manual(values = c("FALSE" = "darkgray", "TRUE" = "lightpink")) +
  ggtitle("Histogram") +
  scale_x_continuous(
    breaks = c(40, 94.4, 150.5, 200.4, 245, 297.8, 400),  # Set breakpoints
    labels = c(40, expression(s[1]), expression(s[2]), 
               expression(s[3]), expression(s[4]), expression(s[5]), 400),  # Custom labels
    limits = c(37, 400)  # Set the x-axis limits
  ) +
  scale_y_continuous(
    breaks = c(0, 0.005, 0.01, 0.015),  # Set breakpoints
    labels = c(expression(0), "", expression(0.01), ""),  # Custom labels
    expand = c(0, 0)
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face="bold", margin = margin(b=5.2)),
    panel.grid.major = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size=10),
    axis.text.y = element_text(size=10),
    legend.position = "none", # or "right"
    legend.title = element_blank()
  ) + 
  xlab("Glucose Level (mg/dL)") +
  coord_flip()  # Flip the x and y axes
# print(p1)

# Making piecewise linear quantile functions
F <- ecdf(x1)
knots <- unique(F(c(40, 94.4, 150.5, 200.4, 245, 297.8, 400)))
q_knots <- quantile(x1, probs = knots, type = 1)

gr <- seq(0, 1, length.out=202)
q_a <- approx(x = knots, y = q_knots, xout = gr[2:(length(gr)-1)], method = "linear", rule = 2)$y

# ggplot the quantile function
df <- data.frame(x = gr[2:(length(gr)-1)], y = q_a)
knots_df <- data.frame(x = knots[2:6], y = q_knots[2:6]) # Data for the knots (vertices) to emphasize
amalgam_line <- data.frame(x = c(knots[4], knots[6]), 
                           y = c(q_knots[4], q_knots[6]))  # Line segment for the amalgamated part
p2 <- ggplot(df, aes(x = x, y = y)) + 
  geom_line(size = 0.9) +
  geom_line(data = amalgam_line, aes(x = x, y = y), size = 0.9, color = "red") +  # Highlight the amalgamated part
  geom_point(data = knots_df, aes(x = x, y = y), color = "black", size = 3) +  # Highlight the knots
  geom_hline(yintercept = q_knots[2:6], linetype = "dashed", 
             color = "darkgreen", size=0.33, alpha=0.9) +  # Add horizontal lines
  scale_x_continuous(breaks = knots,
                     labels = c(0, expression(F(s[1])), expression(F(s[2])),
                                expression(F(s[3])), expression(F(s[4])), expression(F(s[5])), ""#, 1
                                )) +
  # # Add annotation for "1" at a slightly shifted position
  # annotate("text", x = 1.05, y = 39, label = "1", vjust = 4, hjust = 1) + 
  scale_y_continuous(breaks = c(40, 
                                # q_knots[2:6], 
                                400),
                     limits = c(39, 400)) +
  theme_classic() + 
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face="bold", margin = margin(b=5.2)),  # Centers the title and sets font size
    axis.title.x = element_blank(),
    axis.text.x = element_text(size=10),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    # panel.grid.minor = element_blank(),
    # axis.text.x = element_text(size = 10),
    axis.text.y = element_blank(), # remove yticks
  ) +
  ggtitle("Corresponding Quantile Function")  
  
# print(p2)

# Combine the two plots side by side with patchwork
library(patchwork)

combined_plot <- p1 + p2 + 
  plot_layout(ncol = 2, widths = c(0.8, 1)) +  
plot_annotation(theme = theme(plot.margin = margin(0, 0, 20, 0)))  # Add bottom margin (top, right, bottom, left)

# Draw the plot
combined_plot


#######################################
### Combine grid_plot and combined_plot horizontally
#######################################

# Wrap the grid_plot so it can be used with patchwork
wrapped_grid <- wrap_elements(grid_plot)
wrapped_grid <- wrapped_grid + 
  plot_annotation(theme = theme(plot.margin = margin(0, 0, 0, 0)))  # Add small space

# Combine horizontally with 1:2 width ratio
# Add labels (a) and (b) only - the combined_plot already has internal structure
final_plot <- wrapped_grid + combined_plot + 
  plot_layout(ncol = 2,
                widths = c(1, 2) 
                # heights = c(1.2, 1)
                ) +
  plot_annotation(tag_levels = list(c('(a)', '(b)')),
                  theme = theme(
                    plot.tag = element_text(size = 22, face = "bold")
                  ))

# Display the combined plot
# final_plot

# Save the combined plot
ggsave("./images/combined_histogram_quantile.pdf", 
       final_plot, 
       width = 12, height = 4.5, dpi = 500)

# Note about tag positioning:
# - plot.tag.position uses normalized coordinates (0 to 1)
# - c(x, y) where x=0 is left, x=1 is right, y=0 is bottom, y=1 is top
# - For bottom center: c(0.5, 0.02)
# - For upper-left: c(0.02, 0.98)
# - For upper-right: c(0.98, 0.98)
# - For center: c(0.5, 0.5)