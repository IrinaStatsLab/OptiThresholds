# Load required packages
library(tidyverse)
library(scales)
library(patchwork)

# Read data
nondiabetic_data = read_csv("data/shah2019_filtered.csv")
diabetic_data = read_csv("data/brown2019_filtered.csv")

# Ensure data falls within (39, 401)
diabetic_data = diabetic_data %>%
  filter(gl > 39, gl < 401)
nondiabetic_data = nondiabetic_data %>%
  filter(gl > 39, gl < 401)

# Create CDF data for diabetic and non-diabetic
make_cdf_data = function(df) {
  df %>%
    group_by(id) %>%
    arrange(gl, .by_group = TRUE) %>%
    mutate(
      id = as.numeric(id),
      cdf = cume_dist(gl)
    ) %>%
    ungroup()
}

cdf_data = make_cdf_data(diabetic_data)
nondiabetic_cdf_data = make_cdf_data(nondiabetic_data)

# Thresholds and lines (diabetic and trad)
k14_line = c(70, 181, 241, 306)
trad_line = c(54, 70, 181, 251)

# Filter data at closest glucose values for each set of thresholds
filtered_14 = cdf_data %>%
  group_by(id) %>%
  filter(gl %in% sapply(c(39, k14_line, 401), function(x) gl[which.min(abs(gl - x))])) %>%
  ungroup() %>%
  distinct(id, gl, cdf)

filtered_trad = cdf_data %>%
  group_by(id) %>%
  filter(gl %in% sapply(c(39, trad_line, 401), function(x) gl[which.min(abs(gl - x))])) %>%
  ungroup() %>%
  distinct(id, gl, cdf)


# Diabetic: full CDF + thresholded plots
# Empirical Quantiles
k14_full = ggplot(cdf_data, aes(x = cdf, y = gl)) +
  geom_line(aes(group = id, color = id), alpha = 0.2, size = 0.3) +
  scale_y_continuous(breaks = c(40, 100, 200, 300, 400), limits = c(39, 401)) +
  labs(title = "Empirical Quantiles", x = NULL, y = "Glucose Value (mg/dL)") +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 12, hjust = 0.5, face = "bold")
  )

# Discretized Quantiles with Semi-Supervised Thresholds
k14_thresh = ggplot(filtered_14, aes(x = cdf, y = gl)) +
  geom_line(aes(group = id, color = id), alpha = 0.2, size = 0.3) +
  scale_y_continuous(breaks = c(40, k14_line, 400), limits = c(39, 401)) +
  labs(
    title = "Discretized Quantiles with Semi-Supervised Thresholds",
    x = NULL, y = "Glucose Value (mg/dL)"
  ) +
  geom_hline(yintercept = k14_line, color = "black", linetype = "dashed", size = 0.6, alpha = 0.9) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 12, hjust = 0.5, face = "bold"),
    axis.title.y = element_blank()
  )

# Discretized Quantiles with Traditional Thresholds
trad_thresh = ggplot(filtered_trad, aes(x = cdf, y = gl)) +
  geom_line(aes(group = id, color = id), alpha = 0.2, size = 0.3) +
  scale_y_continuous(breaks = c(40, trad_line, 400), limits = c(39, 401)) +
  labs(
    title = "Discretized Quantiles with Traditional Thresholds",
    x = NULL, y = "Glucose Value (mg/dL)"
  ) +
  geom_hline(yintercept = trad_line, color = "black", linetype = "dashed", size = 0.6, alpha = 0.9) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 12, hjust = 0.5, face = "bold"),
    axis.title.y = element_blank()
  )

# Combine plots
diabetic_plots = k14_full + k14_thresh + trad_thresh

# Thresholds and lines (non-diabetic)
nondiabetic_k14_line = c(76, 101, 124, 155)

nondiabetic_filtered_14 = nondiabetic_cdf_data %>%
  group_by(id) %>%
  filter(gl %in% sapply(c(39, nondiabetic_k14_line, 401), function(x) gl[which.min(abs(gl - x))])) %>%
  ungroup() %>%
  distinct(id, gl, cdf)

nondiabetic_filtered_trad = nondiabetic_cdf_data %>%
  group_by(id) %>%
  filter(gl %in% sapply(c(39, trad_line, 301), function(x) gl[which.min(abs(gl - x))])) %>%
  ungroup() %>%
  distinct(id, gl, cdf)

# Empirical Quantiles
k14_full_nondiabetic = ggplot(nondiabetic_cdf_data, aes(x = cdf, y = gl)) +
  geom_line(aes(group = id, color = id), alpha = 0.2, size = 0.3) +
  scale_y_continuous(breaks = c(40, 100, 200, 300), limits = c(39, 301)) +
  labs(title = "Empirical Quantiles", x = NULL, y = "Glucose Value (mg/dL)") +
  theme_classic(base_size = 15) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 15, hjust = 0.5, face = "bold")
  )

# Discretization with Data-Driven Thresholds
k14_thresh_nondiabetic = ggplot(nondiabetic_filtered_14, aes(x = cdf, y = gl)) +
  geom_line(aes(group = id, color = id), alpha = 0.2, size = 0.3) +
  scale_y_continuous(breaks = c(40, nondiabetic_k14_line, 300), limits = c(39, 301)) +
  labs(title = "Discretization with Data-Driven Thresholds", x = NULL) +
  geom_hline(yintercept = nondiabetic_k14_line, color = "black", linetype = "dashed", size = 0.6, alpha = 0.9) +
  theme_classic(base_size = 15) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 15, hjust = 0.5, face = "bold"),
    axis.title.y = element_blank()
  )

# Discretization with Traditional Thresholds
trad_thresh_nondiabetic = ggplot(nondiabetic_filtered_trad, aes(x = cdf, y = gl)) +
  geom_line(aes(group = id, color = id), alpha = 0.2, size = 0.3) +
  scale_y_continuous(breaks = c(trad_line, 300), limits = c(39, 301)) +
  labs(title = "Discretization with Traditional Thresholds", x = NULL) +
  geom_hline(yintercept = trad_line, color = "black", linetype = "dashed", size = 0.6, alpha = 0.9) +
  theme_classic(base_size = 15) +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 15, hjust = 0.5, face = "bold"),
    axis.title.y = element_blank()
  )

# Combined plots
nondiabetic_plots = k14_full_nondiabetic + k14_thresh_nondiabetic + trad_thresh_nondiabetic

# Save out plots
ggsave("images/nondiabetic_quantiles.jpeg", nondiabetic_plots, width = 15, height = 4.5, dpi = 400)
ggsave("images/diabetic_quantiles.jpeg", diabetic_plots, width = 15, height = 4.5, dpi = 400)


## Combined data discrimination plots
nondiabetic_data <- nondiabetic_data %>% select(id, gl)
diabetic_data <- diabetic_data %>% select(id, gl)
combined_data = rbind(nondiabetic_data, diabetic_data) %>% 
  # Adjust ids to make diabetic vs. nondiabetic profiles clear
  mutate(id = ifelse(id > 5200, id - 9000, id))

# Reformat combined data and identify proportion of k = 2 trad TIR/TAR/TBR for each profile
trad_barplot_data = combined_data %>%
  mutate(trad = case_when(
    gl < 70 ~ "TBR",
    gl > 180 ~ "TAR",
    TRUE ~ "TIR"
  )) %>%
  group_by(id) %>%
  count(trad) %>%
  mutate(prop = n / sum(n)) %>%
  select(id, trad, prop) %>%
  pivot_wider(names_from = trad, values_from = prop, values_fill = 0) %>%
  mutate(
    diabetes = ifelse(id < 1000, "healthy", "t1d"),
    TBR = ifelse(is.na(TBR), 0, TBR),
    TIR = ifelse(is.na(TIR), 0, TIR),
    TAR = ifelse(is.na(TAR), 0, TAR)
  ) %>%
  group_by(diabetes) %>%
  arrange(diabetes, TBR) %>%
  ungroup() %>%
  arrange(factor(diabetes, levels = c("healthy", "t1d"))) %>%
  mutate(index = row_number()) %>%
  mutate(id = factor(id, levels = unique(id))) %>%
  pivot_longer(cols = c(TBR, TIR, TAR), names_to = "Range", values_to = "Proportion") %>%
  mutate(Range = factor(Range, levels = c("TAR", "TIR", "TBR")))

# Set breaks and labels for plots
n_healthy = sum(trad_barplot_data$diabetes == "healthy") / 3
custom_breaks = c(50, 100, 150, n_healthy + 50, n_healthy + 100, n_healthy + 150)
custom_labels = c("50", "100", "150","50", "100", "150")

# Plot disrimination using k = 2 trad thresholds
comb_trad = ggplot(trad_barplot_data, aes(x = index, y = Proportion, fill = Range)) +
  geom_bar(stat = "identity", position = "stack", width = 1) +
  scale_fill_manual(values = c("TBR" = "#5E3485", "TIR" = "#00796B", "TAR" = "#F1C90F"),
                    labels = c(">= 181 mg/dL", "70 - 180 mg/dL", "< 70 mg/dL")) +
  geom_vline(xintercept = n_healthy + 0.5, linetype = "dashed", color = "black", size = 1) +
  labs(
    title = "Traditional Thresholds [70, 181]",
    x = "Samples (Left: Non-Diabetic, Right: T1D)",
    fill = "Range"
  ) +
  scale_x_continuous(
    breaks = custom_breaks,
    labels = custom_labels,
    expand = c(0,0)
  ) +
  theme_classic() +
  theme(
    axis.title.y = element_blank(),
    axis.text.y = element_text(size = 14),
    axis.text.x = element_text(size = 14),
    axis.title.x = element_text(size = 16),
    plot.title = element_text(size = 20, hjust = 0.5, face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.position = c(0.17, 0.5),
    legend.background = element_rect(fill = "white", color = "black")
  ) +
  scale_y_continuous(expand = c(0, 0), labels = percent) 


# Reformat combined data and identify proportion of k = 2 data-driven TIR/TAR/TBR for each profile
custom_barplot_data = combined_data %>%
  mutate(trad = case_when(
    gl < 150 ~ "TBR", # 148
    gl > 257 ~ "TAR", # 256
    TRUE ~ "TIR"
  )) %>%
  group_by(id) %>%
  count(trad) %>%
  mutate(prop = n / sum(n)) %>%
  select(id, trad, prop) %>%
  pivot_wider(names_from = trad, values_from = prop, values_fill = 0) %>%
  mutate(
    diabetes = ifelse(id < 1000, "healthy", "t1d"),
    TBR = ifelse(is.na(TBR), 0, TBR),
    TIR = ifelse(is.na(TIR), 0, TIR),
    TAR = ifelse(is.na(TAR), 0, TAR)
  ) %>%
  group_by(diabetes) %>%
  arrange(diabetes, TBR) %>%
  ungroup() %>%
  arrange(factor(diabetes, levels = c("healthy", "t1d"))) %>%
  mutate(index = row_number()) %>%
  mutate(id = factor(id, levels = unique(id))) %>%
  pivot_longer(cols = c(TBR, TIR, TAR), names_to = "Range", values_to = "Proportion") %>%
  mutate(Range = factor(Range, levels = c("TAR", "TIR", "TBR")))

# Set breaks and labels for plots
n_healthy = sum(custom_barplot_data$diabetes == "healthy") / 3
custom_breaks = c(50, 100, 150, n_healthy + 50, n_healthy + 100, n_healthy + 150)
custom_labels = c("50", "100", "150","50", "100", "150")

# Plot disrimination using k = 2 data-driven thresholds
comb_l2k2 = ggplot(custom_barplot_data, aes(x = index, y = Proportion, fill = Range)) +
  geom_bar(stat = "identity", position = "stack", width = 1) +
  scale_fill_manual(values = c(
    "TBR" = "#5E3485",        
    "TIR" = "#00796B",        
    "TAR" = "#F1C40F"         
  ),
  labels = c(">= 258 mg/dL", "150 - 257 mg/dL", "< 150 mg/dL")) +
  geom_vline(xintercept = n_healthy + 0.5, linetype = "dashed", color = "black", size = 1) +
  labs(
    title = "Data-Driven Thresholds [150, 258]",
    x = "Samples (Left: Non-Diabetic, Right: T1D)",
    y = "Time-in-Range Proportions",
    fill = "Range"
  ) +
  scale_x_continuous(
    breaks = custom_breaks,
    labels = custom_labels,
    expand = c(0,0)
  ) +
  theme_classic() +
  theme(
    axis.title.y = element_text(size = 16),
    axis.text.y = element_text(size = 14),
    axis.text.x = element_text(size = 14),
    axis.title.x = element_text(size = 16),
    plot.title = element_text(size = 20, hjust = 0.5, face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.position = c(0.176, 0.5),
    legend.background = element_rect(fill = "white", color = "black")
  ) +
  scale_y_continuous(
    labels = percent,
    expand = c(0, 0)
  ) 

# Save out combined k = 2 plot
comb_plot = comb_l2k2 + comb_trad
ggsave("images/comb_plot.jpeg", comb_plot, width = 15, height = 7, dpi = 600)
print(comb_plot)



# K = 4
# Reformat combined data and identify proportion of k = 4 trad TIR/TAR/TBR for each profile
custom_barplot_data = combined_data %>%
  mutate(trad = case_when(
    gl < 54 ~ "TIR_1",
    gl >= 54 & gl < 70 ~ "TIR_2",
    gl >= 70 & gl < 181 ~ "TIR_3",
    gl >= 181 & gl < 251 ~ "TIR_4",
    TRUE ~ "TIR_5"
  )) %>%
  group_by(id) %>%
  count(trad) %>%
  mutate(prop = n / sum(n)) %>%
  select(id, trad, prop) %>%
  pivot_wider(names_from = trad, values_from = prop, values_fill = 0) %>%
  mutate(
    diabetes = ifelse(id < 1000, "healthy", "t1d"),
    TIR_1 = ifelse(is.na(TIR_1), 0, TIR_1),
    TIR_2 = ifelse(is.na(TIR_2), 0, TIR_2),
    TIR_3 = ifelse(is.na(TIR_3), 0, TIR_3),
    TIR_4 = ifelse(is.na(TIR_4), 0, TIR_4),
    TIR_5 = ifelse(is.na(TIR_5), 0, TIR_5),
  ) %>%
  group_by(diabetes) %>%
  arrange(diabetes, TIR_1) %>%
  ungroup() %>%
  arrange(factor(diabetes, levels = c("healthy", "t1d"))) %>%
  # Add a numerical index for x-axis
  mutate(index = row_number()) %>%
  mutate(id = factor(id, levels = unique(id))) %>%
  pivot_longer(cols = c(TIR_1, TIR_2, TIR_3, TIR_4, TIR_5), names_to = "Range", values_to = "Proportion") %>%
  mutate(Range = factor(Range, levels = c("TIR_5", "TIR_4", "TIR_3", "TIR_2", "TIR_1")))

# Set breaks and labels for plots
n_healthy = sum(custom_barplot_data$diabetes == "healthy") / 5
custom_breaks = c(50, 100, 150, n_healthy + 50, n_healthy + 100, n_healthy + 150)
custom_labels = c("50", "100", "150","50", "100", "150")

# Plot disrimination using k = 4 trad thresholds
comb_trad_4 = ggplot(custom_barplot_data, aes(x = index, y = Proportion, fill = Range)) +
  geom_bar(stat = "identity", position = "stack", width = 1) +
  scale_fill_manual(values = c(
    "TIR_1" = "#5E3485",        
    "TIR_2" = "#5A7BB5",        
    "TIR_3" = "#00796B",
    "TIR_4" = "#28A57B",
    "TIR_5" = "#F1C40F"
  ),
  labels = c(">= 251 mg/dL", "181 - 250 mg/dL", "70 - 180 mg/dL", "54 - 69 mg/dL", "< 54 mg/dL")) +
  geom_vline(xintercept = n_healthy + 0.5, linetype = "dashed", color = "black", size = 1) +
  labs(
    title = "Traditional Thresholds [54, 70, 181, 251]",
    x = "Samples (Left: Non-Diabetic, Right: T1D)",
    y = "Time-in-Range Proportions",
    fill = "Range"
  ) +
  scale_x_continuous(
    breaks = custom_breaks,
    labels = custom_labels,
    expand = c(0,0)
  ) +
  theme_classic() +
  theme(
    axis.title.y = element_text(size = 16),
    axis.text.y = element_text(size = 14),
    axis.text.x = element_text(size = 14),
    axis.title.x = element_text(size = 16),
    plot.title = element_text(size = 20, hjust = 0.5, face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    legend.position = c(0.15, 0.5),
    legend.background = element_rect(fill = "white", color = "black")
  ) +
  scale_y_continuous(
    labels = percent,
    expand = c(0, 0)
  ) 

# Reformat combined data and identify proportion of k = 4 data-driven TIR/TAR/TBR for each profile
custom_barplot_data = combined_data %>%
  # mutate(trad = case_when(
  #   gl < 128 ~ "TIR_1",
  #   gl >= 128 & gl < 193 ~ "TIR_2",
  #   gl >= 193 & gl < 247 ~ "TIR_3",
  #   gl >= 247 & gl < 312 ~ "TIR_4",
  #   TRUE ~ "TIR_5"
  # )) %>%
  mutate(trad = case_when( #82, 126, 193, 275
    gl < 82 ~ "TIR_1",
    gl >= 82 & gl < 126 ~ "TIR_2",
    gl >= 126 & gl < 193 ~ "TIR_3",
    gl >= 193 & gl < 275 ~ "TIR_4",
    TRUE ~ "TIR_5"
  )) %>%
  group_by(id) %>%
  count(trad) %>%
  mutate(prop = n / sum(n)) %>%
  select(id, trad, prop) %>%
  pivot_wider(names_from = trad, values_from = prop, values_fill = 0) %>%
  mutate(
    diabetes = ifelse(id < 1000, "healthy", "t1d"),
    TIR_1 = ifelse(is.na(TIR_1), 0, TIR_1),
    TIR_2 = ifelse(is.na(TIR_2), 0, TIR_2),
    TIR_3 = ifelse(is.na(TIR_3), 0, TIR_3),
    TIR_4 = ifelse(is.na(TIR_4), 0, TIR_4),
    TIR_5 = ifelse(is.na(TIR_5), 0, TIR_5),
  ) %>%
  group_by(diabetes) %>%
  arrange(diabetes, TIR_1) %>%
  ungroup() %>%
  arrange(factor(diabetes, levels = c("healthy", "t1d"))) %>%
  mutate(index = row_number()) %>%
  mutate(id = factor(id, levels = unique(id))) %>%
  pivot_longer(cols = c(TIR_1, TIR_2, TIR_3, TIR_4, TIR_5), names_to = "Range", values_to = "Proportion") %>%
  mutate(Range = factor(Range, levels = c("TIR_5", "TIR_4", "TIR_3", "TIR_2", "TIR_1")))

# Set breaks and labels for plots
n_healthy = sum(custom_barplot_data$diabetes == "healthy") / 5
custom_breaks = c(50, 100, 150, n_healthy + 50, n_healthy + 100, n_healthy + 150)
custom_labels = c("50", "100", "150","50", "100", "150")

# Plot disrimination using k = 4 data-driven thresholds
comb_l2k2_4 = ggplot(custom_barplot_data, aes(x = index, y = Proportion, fill = Range)) +
  geom_bar(stat = "identity", position = "stack", width = 1) +
  scale_fill_manual(values = c(
    "TIR_1" = "#5E3485",        
    "TIR_2" = "#5A7BB5",        
    "TIR_3" = "#00796B",
    "TIR_4" = "#28A57B",
    "TIR_5" = "#F1C40F"
  ),
  # labels = c(">= 312 mg/dL", "247 - 311 mg/dL", "193 - 246 mg/dL", "128 - 192 mg/dL", "< 128 mg/dL")) +
  labels = c(">= 275 mg/dL", "193 - 274 mg/dL", "126 - 192 mg/dL", "82 - 125 mg/dL", "< 82 mg/dL")) + #82, 126, 193, 275
  geom_vline(xintercept = n_healthy + 0.5, linetype = "dashed", color = "black", size = 1) +
  labs(
    title = "Data-Driven Thresholds [82, 126, 193, 275]",
    x = "Samples (Left: Non-Diabetic, Right: T1D)",
    y = "Time-in-Range Proportions",
    fill = "Range"
  ) +
  scale_x_continuous(
    breaks = custom_breaks,
    labels = custom_labels,
    expand = c(0,0)
  ) +
  theme_classic() +
  theme(
    axis.title.y = element_text(size = 16),
    axis.text.y = element_text(size = 14),
    axis.text.x = element_text(size = 14),
    axis.title.x = element_text(size = 16),
    plot.title = element_text(size = 20, hjust = 0.5, face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    legend.position = c(0.15, 0.5),
    legend.background = element_rect(fill = "white", color = "black")
  ) +
  scale_y_continuous(
    labels = percent,
    expand = c(0, 0)
  ) 

# Save out combined k = 4 plot
comb_plot_k4 = comb_l2k2_4 + comb_trad_4
ggsave("images/k4_comb_stacked_barplots.jpeg", comb_plot_k4, width = 15, height = 7, dpi = 600)
print(comb_plot_k4)



