### === LOAD LIBRARIES & FUNCTIONS === ###
# Set library paths for R packages
.libPaths(new = c("/maps/projects/renlab/apps/R_packages/R_4.4.0/",
                  "/opt/software/R/4.4.0/lib64/R/library"))

# Load required libraries with suppressed messages
suppressMessages({
  library(GenomicFeatures)
  library(GenomicRanges)
  library(VariantAnnotation)
  library(tidyverse)
  library(rtracklayer)
  library(BSgenome.Athaliana.TAIR.TAIR9)
  library(broom)
  library(gridExtra)
})

### === ADD ANNOTATION TO GRanges OBJECT === ### 
# Function to annotate genomic sites with transcript features
# Parameters:
#   gr: GRanges object with genomic positions to annotate
#   gtf: GTF annotation file
#   txdb: TxDb object containing transcript database
#   genome: BSgenome object for sequence extraction
#   longest: Whether to use longest transcript isoform
#   fill_na_with_zeros: Whether to fill NA values with zeros
annotate_sites_with_tx_features <- function(gr, gtf, txdb, genome, longest = TRUE, fill_na_with_zeros = TRUE) {
  message("Loading GTF and building TxDb...")
  
  message("Extracting transcript structure...")
  # Extract coding sequences, UTRs by transcript
  cds_by_tx       <- cdsBy(txdb, by = "tx", use.names = TRUE)
  threeUTRs_by_tx <- threeUTRsByTranscript(txdb, use.names = TRUE)
  fiveUTRs_by_tx  <- fiveUTRsByTranscript(txdb, use.names = TRUE)
  
  message("Calculating feature region lengths...")
  # Calculate total lengths for each feature type
  cds_sums      <- unlist(lapply(cds_by_tx, width) %>% lapply(sum))
  threeUTR_sums <- unlist(lapply(threeUTRs_by_tx, width) %>% lapply(sum))
  fiveUTR_sums  <- unlist(lapply(fiveUTRs_by_tx, width) %>% lapply(sum))
  
  message("Mapping input GRanges to transcripts...")
  # Source external functions for transcript matching
  source("transcript_matching_functions.R")
  locGR <- prepareTX(txdb, myGR = gr, seqstyle = "TAIR9", longest = TRUE)
  if (length(locGR) == 0) stop("No GRanges matched to transcripts.")
  
  message("Mapping features...")
  features_list <- list()
  
  # Process 5' UTR features
  if ("fiveUTR" %in% locGR$LOCATIONS) {
    features_list$fiveUTRGR <- getFeatureSet("fiveUTR", locGR, fiveUTRs_by_tx, fiveUTR_sums)
  }
  
  # Process coding sequence features
  if ("coding" %in% locGR$LOCATIONS) {
    features_list$cdsGR <- getFeatureSet("coding", locGR, cds_by_tx, cds_sums)
    features_list$cdsGR$prop <- features_list$cdsGR$prop + 1
  }
  
  # Process 3' UTR features
  if ("threeUTR" %in% locGR$LOCATIONS) {
    features_list$threeUTRGR <- getFeatureSet("threeUTR", locGR, threeUTRs_by_tx, threeUTR_sums)
    features_list$threeUTRGR$prop <- features_list$threeUTRGR$prop + 2
  }
  
  message("Combining annotated features...")
  annotated <- unlist(GRangesList(features_list))
  if (length(annotated) == 0) stop("No annotated GRanges returned.")
  
  message("Adding start/stop codon distances from GTF...")
  # Note: Modified to avoid warning problems and transcript loss during GTF to TxDb transformation
  annotated$stop_dist <- NA_integer_
  annotated$start_dist <- NA_integer_
  
  # Calculate distances to nearest stop and start codons
  stop_hits <- distanceToNearest(annotated, gtf[gtf$type == "stop_codon"])
  start_hits <- distanceToNearest(annotated, gtf[gtf$type == "start_codon"])
  
  annotated$stop_dist[queryHits(stop_hits)] <- mcols(stop_hits)$distance
  annotated$start_dist[queryHits(start_hits)] <- mcols(start_hits)$distance
  
  # Remove entries with missing distance information
  annotated <- annotated[complete.cases(mcols(annotated)[, c("start_dist", "stop_dist")])]
  
  message("Adding transcript structure-derived features...")
  tx_lengths <- computTxLengths(txdb)
  
  # Join with transcript length information
  feature_df <- as_tibble(mcols(annotated)) %>%
    left_join(tx_lengths, by = c("TXID", "tx_name"))
  
  if (fill_na_with_zeros) {
    # Add missing columns if they don't exist
    if (!"cds_start" %in% colnames(feature_df)) {
      feature_df$cds_start <- NA_integer_
    }
    if (!"cds_end" %in% colnames(feature_df)) {
      feature_df$cds_end <- NA_integer_
    }
    
    # Fill NA values with appropriate defaults
    feature_df <- feature_df %>%
      mutate(
        tx_len     = coalesce(tx_len, 0),
        utr3_len   = coalesce(utr3_len, 0),
        utr5_len   = coalesce(utr5_len, 0),
        nexon      = coalesce(nexon, 0),
        cds_start  = coalesce(cds_start, NA_integer_),
        cds_end    = coalesce(cds_end, NA_integer_)
      )
  }
  
  mcols(annotated) <- feature_df
  
  message("Calculating position-related features...")
  # Calculate distance from transcript start
  annotated$dist_from_start <- ifelse(strand(annotated) == "+",
                                      start(annotated) - annotated$tx_start,
                                      annotated$tx_end - end(annotated))
  
  # Calculate distance from transcript end
  annotated$dist_from_end <- ifelse(strand(annotated) == "+",
                                    annotated$tx_end - end(annotated),
                                    start(annotated) - annotated$tx_start)
  
  # Calculate distance from CDS start
  annotated$start_dist <- ifelse(!is.na(annotated$cds_start),
                                 abs(start(annotated) - annotated$cds_start), 0)
  
  # Calculate distance from CDS end
  annotated$stop_dist <- ifelse(!is.na(annotated$cds_end),
                                abs(end(annotated) - annotated$cds_end), 0)
  
  message("Adding final attributes and returning...")
  # Add additional feature annotations
  annotated$feature_width <- width(annotated)
  annotated$coding        <- as.integer(annotated$LOCATIONS == "coding")
  annotated$fiveUTR       <- as.integer(annotated$LOCATIONS == "fiveUTR")
  annotated$threeUTR      <- as.integer(annotated$LOCATIONS == "threeUTR")
  annotated$id            <- paste0(seqnames(annotated), "_", start(annotated), ",", strand(annotated))
  
  return(annotated)
}

# Function to generate multiple negative samples for each positive sample
# Parameters:
#   transcripts: GRanges object containing transcript annotations
#   pos_samples: GRanges object representing positive sample positions (midpoint, width = 1)
#   num_neg_per_pos: Number of negative samples to generate per positive sample
#   buffer_factor: Multiplier for extra candidate generation before filtering
#   region_size: Length of each negative sample region (default: 511 bp)
#   remove_window: Window size for removing overlaps with positive samples
generate_multiple_negative_samples <- function(transcripts,
                                               pos_samples,
                                               num_neg_per_pos,
                                               buffer_factor = 5,
                                               region_size = 511,
                                               remove_window = 10) {
  
  message("Generating negative samples...")
  half_window <- floor((region_size - 1) / 2)  # Half region size for boundary checking

  # Get chromosome lengths from BSgenome reference
  chr_lengths <- seqlengths(BSgenome.Athaliana.TAIR.TAIR9)

  # Find transcripts that contain positive samples
  overlap_idx <- subjectHits(findOverlaps(pos_samples, transcripts))
  tx_used <- transcripts[unique(overlap_idx)]

  # Merge transcripts into non-overlapping regions for sampling
  tx_merged <- GenomicRanges::reduce(tx_used, ignore.strand = FALSE)

  # Initialize list to store negative candidates for each positive sample
  neg_list <- vector("list", length(pos_samples))
  message("Processing positive samples...")
  
  # Process each positive sample independently
  for (i in seq_along(pos_samples)) {
    pos <- pos_samples[i]
    ps_chr <- as.character(seqnames(pos))  # Chromosome name
    ps_str <- as.character(strand(pos))    # Strand information

    # Extract transcript regions for same chromosome and strand
    tx_chr <- as.character(seqnames(tx_merged))
    tx_str <- as.character(strand(tx_merged))
    sub_tx <- tx_merged[tx_chr == ps_chr & tx_str == ps_str ]

    if (length(sub_tx) == 0) {
      warning(sprintf("Positive sample %d (%s:%s) has no matching transcript regions", i, ps_chr, ps_str))
      neg_list[[i]] <- GRanges()
      next
    }

    # Calculate total transcript width for weighted sampling
    widths <- width(sub_tx)
    total_width <- sum(widths)
    if (total_width < region_size) {
      warning(sprintf("Positive sample %d (%s:%s) has insufficient transcript length (%d bp required)",
                      i, ps_chr, ps_str, region_size))
      neg_list[[i]] <- GRanges()
      next
    }
    w_cumsum <- cumsum(as.numeric(widths))

    # Generate extra candidates with buffer factor
    candidates_to_draw <- num_neg_per_pos * buffer_factor
    rand_vals <- runif(candidates_to_draw, min = 1, max = total_width)
    intervals_idx <- findInterval(rand_vals, w_cumsum)
    left_bound <- c(0, w_cumsum)[intervals_idx]
    offset <- rand_vals - left_bound
    candidate_starts <- floor(start(sub_tx)[intervals_idx] + offset)

    # Create candidate negative samples with fixed length
    candidates <- GRanges(
      seqnames = ps_chr,
      ranges   = IRanges(start = candidate_starts, width = region_size),
      strand   = ps_str
    )

    # Adjust boundaries to stay within chromosome limits
    chr_len <- chr_lengths[as.character(ps_chr)]
    candidate_ends <- end(candidates)
    new_end <- pmin(candidate_ends, chr_len)

    # Ensure valid region lengths
    invalid_idx <- which((new_end - start(candidates) + 1) < region_size)
    if (length(invalid_idx) > 0) {
      start(candidates)[invalid_idx] <- pmax(new_end[invalid_idx] - region_size + 1, 1)
    }
    end(candidates) <- new_end

    # Validate candidate midpoints are within chromosome boundaries
    centers <- round((start(candidates) + end(candidates)) / 2)
    valid_idx <- which(centers - half_window >= 1 & centers + half_window <= chr_len)

    if (length(valid_idx) == 0) {
      warning(sprintf("Positive sample %d: No valid candidate negative samples after boundary check", i))
      neg_list[[i]] <- GRanges()
      next
    }
    candidates <- candidates[valid_idx]

    # Remove candidates overlapping with positive samples
    if (remove_window > 0) {
      pos_expanded <- pos
      start(pos_expanded) <- start(pos_expanded) - remove_window
      end(pos_expanded)   <- end(pos_expanded) + remove_window
      ov <- findOverlaps(candidates, pos_expanded)
      if (length(ov) > 0) {
        candidates <- candidates[-unique(queryHits(ov))]
      }
    } else {
      ov <- findOverlaps(candidates, pos)
      if (length(ov) > 0) {
        candidates <- candidates[-unique(queryHits(ov))]
      }
    }

    # Remove duplicates based on midpoints
    candidate_centers <- round((start(candidates) + end(candidates)) / 2)
    unique_idx <- !duplicated(candidate_centers)
    candidates <- candidates[unique_idx]

    # Sample required number of negatives
    if (length(candidates) < num_neg_per_pos) {
      warning(sprintf("Positive sample %d: Only %d unique candidate negatives found (needed %d)",
                      i, length(candidates), num_neg_per_pos))
      neg_list[[i]] <- candidates
    } else {
      neg_list[[i]] <- sample(candidates, num_neg_per_pos)
    }
  }

  # Combine all negative samples
  final_negatives <- unlist(GRangesList(neg_list))

  # Convert regions to single base midpoints
  mid <- round((start(final_negatives) + end(final_negatives)) / 2)
  ranges(final_negatives) <- IRanges(start = mid, width = 1)
  
  # Print strand distribution
  table(strand(final_negatives))

  return(final_negatives)
}

# Function to process negative samples using Method 3/4 approach
# Parameters:
#   class: Classification category (e.g., "ect2", "alba4")
#   rbp: RNA-binding protein name
#   cell: Cell type
#   txdb: TxDb object
#   gtf: GTF annotation
#   genome: BSgenome object
#   output_dir: Output directory for results
process_one_neg34 <- function(class, rbp, cell, txdb, gtf, genome, output_dir = "results") {
  message("Generating Method 3/4 negatives for: ", rbp, " | ", cell, " | ", class)
  
  # === Load positive data === #
  pos_rdata_file <- file.path(output_dir, class, paste0(rbp, "_", cell, "_pos_annotated.RData"))
  if (!file.exists(pos_rdata_file)) {
    stop("Positive RData file not found: ", pos_rdata_file)
  }
  load(pos_rdata_file)  # loads `pos_annotated`
  
  message("Extracting transcript structure...")
  # === Extract TxDb-derived annotations === #
  fiveUTRs_by_tx <- fiveUTRsByTranscript(txdb, use.names = TRUE)
  cds_by_tx <- cdsBy(txdb, by = "tx", use.names = TRUE)
  threeUTRs_by_tx <- threeUTRsByTranscript(txdb, use.names = TRUE)
  
  # Calculate feature lengths
  fiveUTR_sums <- sum(width(fiveUTRs_by_tx))
  cds_sums <- sum(width(cds_by_tx))
  threeUTR_sums <- sum(width(threeUTRs_by_tx))
  
  message("Preparing transcript subset...")
  # === Generate negative candidates === #
  # Use only transcripts that contain positive samples
  tx_names_pos <- unique(pos_annotated$tx_name)
  transcripts_all <- transcripts(txdb)
  target_transcripts <- transcripts_all[transcripts_all$tx_name %in% tx_names_pos]

  # Generate negative sample candidates
  num_neg_per_pos <- 20
  neg_candidates <- generate_multiple_negative_samples(
    transcripts = target_transcripts,
    pos_samples = pos_annotated,
    num_neg_per_pos = num_neg_per_pos,
    buffer_factor = 5,
    region_size = 511,
    remove_window = 0
  )
  
  # Map negative candidates to transcripts
  loc_neg <- prepareTX(txdb, myGR = neg_candidates, seqstyle = "TAIR9", longest = TRUE)
  
  message("Assigning genomic regions...")
  # === Assign regions to feature types === #
  fiveUTRGR_neg <- getFeatureSet("fiveUTR", loc_neg, fiveUTRs_by_tx, fiveUTR_sums)
  cdsGR_neg <- getFeatureSet("coding", loc_neg, cds_by_tx, cds_sums)
  threeUTRGR_neg <- getFeatureSet("threeUTR", loc_neg, threeUTRs_by_tx, threeUTR_sums)
  
  # Adjust proportional values for different regions
  cdsGR_neg$prop <- cdsGR_neg$prop + 1
  threeUTRGR_neg$prop <- threeUTRGR_neg$prop + 2

  # Combine all negative candidates
  neg_candidates <- unlist(GRangesList(cdsGR_neg, threeUTRGR_neg, fiveUTRGR_neg))
  message("Annotated region breakdown:")
  print(table(neg_candidates$LOCATIONS))
  
  message("Adding features and sequences...")
  # === Add features and sequence information === #
  neg_candidates <- addStopStart(neg_candidates, gtf)
  neg_candidates$LOCATIONS <- as.factor(as.vector(neg_candidates$LOCATIONS))
  neg_candidates <- oneHotFeatures(neg_candidates, y = neg_candidates$LOCATIONS)
  
  # Add transcript length information
  tx_lengths <- computTxLengths(txdb)
  mcols(neg_candidates) <- as_tibble(mcols(neg_candidates)) %>%
    left_join(tx_lengths, by = c("TXID", "tx_name"))
  
  # Create unique identifiers
  neg_candidates$id <- paste0(seqnames(neg_candidates), "_", start(neg_candidates), ",", strand(neg_candidates))
  names(neg_candidates) <- neg_candidates$id
  
  # Extract sequences
  neg_candidates$motif <- as.vector(getSeq(genome, neg_candidates + 255))
  pos_annotated$motif <- as.vector(getSeq(genome, pos_annotated + 255))
  
  message("Assigning labels...")
  # === Assign labels === #
  pos_annotated$label <- 1
  neg_candidates$label <- 0
  posneg <- c(pos_annotated, neg_candidates)

  message("Implementing Method 3: Random sampling...")
  # === Method 3: Same transcript random sampling === #
  tosampleGR <- posneg[posneg$label == 0]
  
  # Random sampling with fixed seed for reproducibility
  set.seed(42)
  random_bg_ids <- sort(sample(tosampleGR$id, length(pos_annotated), replace = FALSE))
  posneg$random_bg <- as.numeric(posneg$id %in% random_bg_ids)
  
  message("Exporting negative samples...")
  # === Export Method 3 results === #
  random_bg <- posneg[posneg$random_bg == 1]
  
  # Calculate N content ratio for quality filtering
  n_ratio2 <- vapply(as.character(random_bg$motif), function(seq) {
    sum(str_count(seq, "N")) / nchar(seq)
  }, numeric(1))
  
  # Filter sequences with low N content (< 5%)
  valid_indices2 <- which(n_ratio2 <= 0.05)
  set.seed(42)
  filtered_random <- sample(random_bg[valid_indices2], length(pos_annotated))
  
  # Save as RData file
  output_file_neg3_rdat <- file.path(output_dir, class, paste0(rbp, "_", cell, "_neg3.Rdat"))
  save(filtered_random, file = output_file_neg3_rdat)
  message("Saved RData file: ", output_file_neg3_rdat)
  
}

### === LOAD FUNCTIONS & SHARED DATA === ###
# Load external functions for transcript matching
source("transcript_matching_functions.R")

####################################################################################
####################################################################################
# === Setup annotation files and genome === #
gtf_path <- "/maps/projects/renlab/people/nkz325/Thesis_RBP/00_FILES/Araport11_GFF3_genes_transposons.201606.gtf"
gtf <- rtracklayer::import(gtf_path)
txdb <- txdbmaker::makeTxDbFromGFF(gtf_path, format = "gtf")
genome <- BSgenome.Athaliana.TAIR.TAIR9
transcripts_gr <- transcripts(txdb, columns = "tx_name")

# Align sequence styles to TAIR9
seqlevelsStyle(transcripts_gr) <- "TAIR9"
seqlevelsStyle(gtf) <- "TAIR9"
seqlevelsStyle(genome) <- "TAIR9"

# Load experimental data
load("/maps/projects/renlab/people/nkz325/Thesis_RBP/00_FILES/iCLIP_with_meta.Rdat")
load("/maps/projects/renlab/people/nkz325/Thesis_RBP/00_FILES/ALBAGR_with_meta.Rdat")

################# Data Processing ##############

################# Process ECT2 data #################
message("Processing ECT2 data...")
ECT2GR <- iCLIP[[3]]
ECT2GR_check <- ECT2GR[ECT2GR$gene != ""]  # Filter out entries without gene annotation

seqlevelsStyle(ECT2GR_check) <- "TAIR9"

# Annotate positive samples with transcript features
pos_annotated <- annotate_sites_with_tx_features(
  gr = ECT2GR_check, 
  gtf = gtf, 
  txdb = txdb, 
  genome = BSgenome.Athaliana.TAIR.TAIR9
)

# Create output directory and save annotated positive samples
dir.create("results/ect2", recursive = TRUE, showWarnings = FALSE)
save(pos_annotated, file = "results/ect2/ECT2_Root_pos_annotated.RData")

# Process negative samples using Method 3/4
process_one_neg34(
  class = "ect2",
  rbp = "ECT2",
  cell = "Root",
  txdb = txdb,
  gtf = gtf,
  genome = BSgenome.Athaliana.TAIR.TAIR9
)

################# Process ALBA4 data #################
message("Processing ALBA4 data...")
ALBAGR_check <- ALBAGR[ALBAGR$gene != ""]  # Filter out entries without gene annotation
seqlevelsStyle(ALBAGR_check) <- "TAIR9"

# Annotate positive samples with transcript features
pos_annotated <- annotate_sites_with_tx_features(
  gr = ALBAGR_check, 
  gtf = gtf, 
  txdb = txdb, 
  genome = BSgenome.Athaliana.TAIR.TAIR9
)

# Create output directory and save annotated positive samples
dir.create("results/alba4", recursive = TRUE, showWarnings = FALSE)
save(pos_annotated, file = "results/alba4/ALBA4_Root_pos_annotated.RData")

# Process negative samples using Method 3/4
process_one_neg34(
  class = "alba4",
  rbp = "ALBA4",
  cell = "Root",
  txdb = txdb,
  gtf = gtf,
  genome = BSgenome.Athaliana.TAIR.TAIR9
)