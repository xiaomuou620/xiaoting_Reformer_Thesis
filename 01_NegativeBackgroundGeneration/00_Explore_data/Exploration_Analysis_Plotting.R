# =====================
# RBP Binding Site Exploration Script (Improved Visuals)
# =====================

.libPaths(new = c("/maps/projects/renlab/apps/R_packages/R_4.4.0/", 
                  "/opt/software/R/4.4.0/lib64/R/library"))

cat("▶ Start: Loading required packages...\n")
suppressPackageStartupMessages({
  library(GenomicFeatures)
  library(GenomicRanges)
  library(VariantAnnotation)
  library(tidyverse)
  library(BSgenome.Athaliana.TAIR.TAIR9)
  library(broom)
  library(gridExtra)
  library(rtracklayer)
  library(ChIPpeakAnno)
  library(Biostrings)
  library(ggplot2)
  library(reshape2)
  library(dplyr)
  library(ggsci)
})

# Save plot with a consistent theme
save_plot <- function(p, filename, outdir = ".", width = 5.5, height = 4.2, dpi = 300) {
  # Standard theme: centered titles, consistent font size, margins, and legend at bottom
  p <- p + theme(
    plot.title = element_text(hjust = 0.5),              
    axis.title.x = element_text(hjust = 0.5),            
    axis.title.y = element_text(size = 12),
    axis.text = element_text(size = 10),
    text = element_text(size = 12),                      
    plot.margin = margin(10, 10, 10, 10),                
    legend.position = "bottom",                          
    legend.title = element_blank()                       
  )

  ggsave(
    filename = file.path(outdir, filename),
    plot = p,
    width = width,
    height = height,
    dpi = dpi,
    limitsize = FALSE
  )
}

# === Input paths ===
gtf_path <- "../../00_FILES/Araport11_GFF3_genes_transposons.201606.gtf"
load("../../00_FILES/iCLIP_with_meta.Rdat")
load("../../00_FILES/ALBAGR_with_meta.Rdat")

# === Output directory ===
outdir <- "exploration_output_neg"
dir.create(outdir, showWarnings = FALSE)

# === Data preparation ===
gtf <- rtracklayer::import(gtf_path)
txdb <- txdbmaker::makeTxDbFromGFF(gtf_path, format = "gtf")
genome <- BSgenome.Athaliana.TAIR.TAIR9

seqlevelsStyle(gtf) <- "TAIR9"
seqlevelsStyle(genome) <- "TAIR9"

ECT2GR_check <- iCLIP[[3]]
seqlevelsStyle(ECT2GR_check) <- "TAIR9"

ALBAGR_check <- ALBAGR[ALBAGR$gene != ""]
seqlevelsStyle(ALBAGR_check) <- "TAIR9"

# === Colors ===
protein_colors <- c(
  "ECT2" = "#3B5BA4",
  "ALBA4" = "#C94040"
)
region_colors <- c(
  "CDS" = "#4DBBD5", 
  "5UTR" = "#E64B35", 
  "3UTR" = "#00A087", 
  "intron" = "#3C5488", 
  "intergenic" = "#F39B7F"
)

# === Load annotated binding site data ===
source("transcript_matching_functions.R")
load("../01_BackgroundGeneration/results/ect2/ECT2_Root_pos_annotated.RData")
load("../01_BackgroundGeneration/results/ect2/ECT2_Root_neg3.Rdat")
ECT2_annotated  <- filtered_random

load("../01_BackgroundGeneration/results/alba4/ALBA4_Root_pos_annotated.RData")
load("../01_BackgroundGeneration/results/alba4/ALBA4_Root_neg3.Rdat")
ALBA4_annotated  <- filtered_random

# === Plot region proportion distribution ===
plot_region_prop_distribution <- function(annotated, sample_name, outdir = ".") {
  df <- as.data.frame(annotated)
  
  if (!"prop" %in% colnames(df)) {
    warning("No 'prop' column found for: ", sample_name)
    return(NULL)
  }
  
  region_labels <- c("5'UTR", "CDS", "3'UTR")
  
  p <- ggplot(df, aes(x = prop)) +
    geom_density(fill = protein_colors[[sample_name]], alpha = 0.4, adjust = 1.2) +
    geom_vline(xintercept = c(1, 2), linetype = "dashed") +
    scale_x_continuous(
      name = "Transcript Region",
      breaks = c(0.5, 1.5, 2.5),
      labels = region_labels,
      limits = c(0, 3)
    ) +
    ylab("Density") +
    theme_minimal(base_size = 14) +
    labs(title = paste0(sample_name, " Binding Site Region Distribution")) +
    theme(
      axis.text = element_text(color = "black"),
      plot.title = element_text(hjust = 0.5)
    )
  
  save_plot(p, paste0(sample_name, "_region_prop_density.png"), outdir)
  return(p)
}

plot_region_prop_distribution(ALBA4_annotated, "ALBA4", outdir)
plot_region_prop_distribution(ECT2_annotated, "ECT2", outdir)

# === Plot chromosome distribution ===
plot_binding_site_chrom_distribution <- function(gr, sample_name) {
  df <- data.frame(Chrom = as.character(seqnames(gr)))

  # Rename "ChrC" (chloroplast) to "Pt"
  df$Chrom[df$Chrom == "ChrC"] <- "Pt"

  # Set chromosome order
  df$Chrom <- factor(df$Chrom, levels = c(paste0("Chr", 1:5), "Pt"))

  df_summary <- df %>%
    group_by(Chrom) %>%
    summarise(Count = n())

  write.csv(df_summary, file.path(outdir, paste0(sample_name, "_chrom_distribution.csv")), row.names = FALSE)

  fill_color <- protein_colors[[sample_name]]
  chrom_n <- length(levels(df$Chrom))

  p <- ggplot(df_summary, aes(x = Chrom, y = Count, fill = Chrom)) +
    geom_bar(stat = "identity", width = 0.7, color = "black", show.legend = FALSE) +
    geom_text(aes(label = Count), vjust = -0.3, size = 3.5) +
    scale_fill_manual(values = rep(fill_color, chrom_n)) +
    theme_minimal(base_size = 14) +
    labs(title = paste0(sample_name, " Binding Sites per Chromosome"),
         x = "Chromosome", y = "Binding Site Count") +
    ylim(0, max(df_summary$Count) * 1.15)

  save_plot(p, paste0(sample_name, "_chrom_distribution.png"), outdir)
  return(p)
}

plot_binding_site_chrom_distribution(ECT2GR_check, "ECT2")
plot_binding_site_chrom_distribution(ALBAGR_check, "ALBA4")

# === Plot transcript region structure distribution ===
plot_binding_site_structure_distribution <- function(annotated, outdir = ".", sample_name = "Protein") {
  df <- data.frame(Feature = annotated$LOCATIONS) |>
    dplyr::filter(!is.na(Feature)) |>
    dplyr::group_by(Feature) |>
    dplyr::summarise(Count = n()) |>
    dplyr::arrange(desc(Count)) |>
    dplyr::mutate(Feature = factor(Feature, levels = rev(Feature)))
  
  write.csv(df, file.path(outdir, paste0(sample_name, "_feature_distribution.csv")), row.names = FALSE)
  
  p <- ggplot(df, aes(x = Feature, y = Count, fill = Feature)) +
    geom_bar(stat = "identity", width = 0.7, show.legend = FALSE) +
    geom_text(aes(label = Count), hjust = -0.2, size = 4.2) +
    scale_fill_npg() +
    coord_flip() +
    theme_minimal(base_size = 14) +
    labs(title = paste0(sample_name, " Binding Site Distribution by Region"),
         y = "Binding Site Count", x = "Transcript Region") +
    theme(
      plot.title = element_text(hjust = 0.5),
      axis.text = element_text(color = "black"),
      panel.grid.major.y = element_blank()
    ) +
    ylim(0, max(df$Count) * 1.1)
  
  save_plot(p, paste0(sample_name, "_feature_distribution.png"), outdir)
  return(p)
}

plot_binding_site_structure_distribution(ECT2_annotated, outdir, sample_name = "ECT2")
plot_binding_site_structure_distribution(ALBA4_annotated, outdir, sample_name = "ALBA4")

# === Get binding site sequences (strand-aware) ===
get_binding_seq <- function(gr, genome) {
  # Remove records with '*' strand
  gr <- gr[strand(gr) %in% c("+", "-")]
  center <- resize(gr, width = 1, fix = "center")
  getSeq(genome, center)
}

# === Plot nucleotide composition ===
plot_nucleotide_composition <- function(seq, sample_name) {
  freq <- colSums(alphabetFrequency(seq, baseOnly = TRUE)[, c("A", "C", "G", "T")])
  df <- data.frame(Base = names(freq), Count = as.numeric(freq))
  write.csv(df, file.path(outdir, paste0(sample_name, "_base_composition.csv")), row.names = FALSE)
  
  fill_color <- protein_colors[[sample_name]]
  
  p <- ggplot(df, aes(x = Base, y = Count, fill = Base)) +
    geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
    geom_text(aes(label = Count), vjust = -0.3, size = 3) +
    scale_fill_manual(values = rep(fill_color, 4)) +
    theme_minimal(base_size = 14) +
    labs(title = paste0(sample_name, " Base Composition"), x = "Base", y = "Count") +
    ylim(0, max(df$Count) * 1.1)
  
  save_plot(p, paste0(sample_name, "_base_composition.png"), outdir)
  return(p)
}

ect2_seq <- get_binding_seq(ECT2GR_check, genome)
plot_nucleotide_composition(ect2_seq, "ECT2")

alba4_seq <- get_binding_seq(ALBAGR_check, genome)
plot_nucleotide_composition(alba4_seq, "ALBA4")

# === Plot binding site width distribution ===
plot_binding_width_distribution <- function(gr, sample_name) {
  df <- data.frame(Width = width(gr))
  write.csv(df, file.path(outdir, paste0(sample_name, "_width_distribution.csv")), row.names = FALSE)
  
  fill_color <- protein_colors[[sample_name]]
  
  p <- ggplot(df, aes(x = Width)) +
    geom_histogram(fill = fill_color, bins = 40, color = "black") +
    theme_minimal(base_size = 14) +
    labs(title = paste0(sample_name, " Binding Width Distribution"),
         x = "Width (nt)", y = "Frequency")
  
  save_plot(p, paste0(sample_name, "_width_distribution.png"), outdir)
  return(p)
}
