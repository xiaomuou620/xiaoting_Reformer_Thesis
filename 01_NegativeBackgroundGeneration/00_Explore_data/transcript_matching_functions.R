computTxLengths <- function(txdb){
  tl <- dplyr::as_tibble(GenomicFeatures::transcriptLengths(txdb, 
                                                            with.cds_len = TRUE, 
                                                            with.utr5_len = TRUE, 
                                                            with.utr3_len = TRUE))
  
  introns_by_tx <- GenomicFeatures::intronsByTranscript(txdb, use.names = TRUE)
  intron_sums <- lapply(introns_by_tx, function(x) sum(width(x)))
  intron_sums <- dplyr::tibble(tx_name = names(intron_sums), intron_len = as.numeric(unlist(intron_sums)))
  tl <- tl %>% dplyr::left_join(intron_sums, by = "tx_name")
  tl <- tl %>% dplyr::mutate(tx_id = as.character(tx_id)) %>% dplyr::rename(TXID=tx_id) 
  return(tl)
}


handle <- function(tvec){
  if("threeUTR" %in% tvec){
    return("threeUTR")
  }
  if("fiveUTR" %in% tvec){
    return("fiveUTR")
  }
  if("coding" %in% tvec){
    return("coding")
  }
  if("intron" %in% tvec){
    return("intron")
  }
  if("intergenic" %in% tvec){
    return("intergenic")
  }
  return("")
}

prepareTX <- function(txdb, myGR, chrs = NULL, seqstyle = NULL, longest=TRUE) {
  if (!is.null(seqstyle)) {
    seqlevelsStyle(myGR) <- seqstyle
    seqlevelsStyle(txdb) <- seqstyle
  }
  
  if (is.null(chrs)) {
    chrs <- intersect(seqlevels(myGR), seqlevels(txdb))
  }
  
  myGR <- keepSeqlevels(myGR, chrs, pruning.mode = "coarse")
  txdb <- keepSeqlevels(txdb, chrs, pruning.mode = "coarse")
  
  seqinfo(myGR) <- seqinfo(txdb)[chrs]
  myGR <- trim(myGR)
  
  locVar <- suppressWarnings({locateVariants(myGR, txdb, AllVariants())})
  myGR$index <- seq_along(myGR)
  
  tl <- dplyr::as_tibble(GenomicFeatures::transcriptLengths(txdb))
  tl <- as_tibble(tl) %>% dplyr::rename(TXID=tx_id) %>% mutate(TXID = as.character(TXID))
  
  # modified
  if(longest){
    variant_annots <- as_tibble(locVar) %>% mutate(TXID = as.character(TXID)) %>% left_join(tl,by="TXID") %>% 
      group_by(QUERYID) %>%  
      summarise(tx_name = tx_name[which(tx_len==max(tx_len))[1]],
                LOCATIONS = LOCATION[which(tx_len==max(tx_len))[1]],
                TXID = TXID[which(tx_len==max(tx_len))[1]]) %>% ungroup()
  }else{
    variant_annots <- as_tibble(locVar) %>%
      group_by(QUERYID) %>%
      summarise(LOCATIONS = handle(LOCATION), 
                GENEID = GENEID[which(LOCATION == LOCATIONS)[1]],
                TXID   = TXID[which(LOCATION == LOCATIONS)[1]]) %>%
      unnest(cols = c("GENEID", "TXID"))
  }
  
  tx_names <- GenomicFeatures::id2name(txdb)
  tx_names <- dplyr::as_tibble(tx_names,rownames="TXID")
  colnames(tx_names)[2] <- "tx_name"
  
  m <- match(myGR$index, variant_annots$QUERYID)
  mcols(myGR) <- cbind(as_tibble(mcols(myGR)) , variant_annots[m,]) 
  
  return(myGR)
}



# getFeatureSet <- function(feature = "coding", myGR, tx_object, tx_sums) {
#   tmp <- myGR[which(myGR$LOCATIONS == feature)]
#  transcript_obj <- tx_object[which(names(tx_object) %in% tmp$tx_name)]
# 
#   ol <- findOverlaps(tmp, unlist(transcript_obj))
# 
#   subHits <- unlist(transcript_obj)[subjectHits(ol)]
#   qHits <- tmp[queryHits(ol)]
# 
#   subHits_trim <- subHits[which(qHits$tx_name == names(subHits))]
#   qHits_trim <- qHits[which(qHits$tx_name == names(subHits))]
# 
#   qHits_trim$feature_width <- width(subHits_trim)
#   qHits_trim$exon_rank <- subHits_trim$exon_rank
# 
#   dist_from_end <- abs(end(subHits_trim) - start(qHits_trim))
#   dist_from_start <- abs(start(qHits_trim) - start(subHits_trim))
# 
#   mcols(qHits_trim)$dist_from_end <- dist_from_end
#   mcols(qHits_trim)$dist_from_start <- dist_from_start
# 
#   neg_strand <- which(strand(qHits_trim) == "-")
#   mcols(qHits_trim)$dist_from_end[neg_strand] <- dist_from_start[neg_strand]
#   mcols(qHits_trim)$dist_from_start[neg_strand] <- dist_from_end[neg_strand]
# 
#   qHits_trim_prop <- getFeatureProp(feature = feature, myGR = qHits_trim, tx_object = tx_object, tx_sums = tx_sums)
# 
#   unique(qHits_trim_prop)
# }

getFeatureSet <- function(feature = "coding", myGR, tx_object, tx_sums) {
  # === Use all myGR, don't filter by LOCATIONS === #
  transcript_obj <- tx_object[which(names(tx_object) %in% myGR$tx_name)]

  ol <- findOverlaps(myGR, unlist(transcript_obj))

  if (length(ol) == 0) return(GRanges())  # early exit

  subHits <- unlist(transcript_obj)[subjectHits(ol)]
  qHits <- myGR[queryHits(ol)]

  # Match only where tx_name matches
  matched_idx <- which(qHits$tx_name == names(subHits))
  subHits_trim <- subHits[matched_idx]
  qHits_trim <- qHits[matched_idx]

  # Annotate feature info
  qHits_trim$feature_width <- width(subHits_trim)
  qHits_trim$exon_rank <- subHits_trim$exon_rank

  dist_from_end <- abs(end(subHits_trim) - start(qHits_trim))
  dist_from_start <- abs(start(qHits_trim) - start(subHits_trim))

  neg_strand <- which(strand(qHits_trim) == "-")
  qHits_trim$dist_from_end <- dist_from_end
  qHits_trim$dist_from_start <- dist_from_start
  qHits_trim$dist_from_end[neg_strand] <- dist_from_start[neg_strand]
  qHits_trim$dist_from_start[neg_strand] <- dist_from_end[neg_strand]

  # Calculate proportion along transcript
  qHits_trim_prop <- getFeatureProp(feature, qHits_trim, tx_object, tx_sums)

  return(unique(qHits_trim_prop))
}


getFeatureProp <- function(feature = "coding", myGR , tx_object, tx_sums){
  tmp <- myGR
  mcols(tmp)$prop <- NA
  res <- mapToTranscripts(tmp,transcripts = tx_object[names(tx_object) %in% tmp$tx_name],ignore.strand=FALSE)
  res$flen <- tx_sums[as.vector(seqnames(res))]
  res$prop <- start(res)/res$flen
  result <- tapply(res$prop,res$xHits,function(x) mean(x[which(x<=1)],na.rm=T))
  mcols(tmp[as.numeric(names(result))])$prop <- result
  return(tmp)
}


bin_by_quantile <- function(x, missing_code = 9) {
  if (all(is.na(x))) return(rep(missing_code, length(x)))
  
  qs <- quantile(x, probs = c(0.05, 0.1, 0.25, 0.50, 0.75, 0.9, 0.95,1), na.rm = TRUE)
  
  case_when(
    is.na(x)       ~ missing_code,
    x <= qs[1]     ~ 1,
    x <= qs[2]     ~ 2,
    x <= qs[3]     ~ 3,
    x <= qs[4]     ~ 4,
    x <= qs[5]     ~ 5,
    x <= qs[6]     ~ 6,
    x <= qs[7]     ~ 7,
    x <= qs[8]     ~ 8,
    TRUE           ~ 9
  )
}

addStopStart <- function(myGR,gtf){
  myGR$stop_dist <- mcols(distanceToNearest(myGR,gtf[gtf$type=="stop_codon"]))$distance
  myGR$start_dist <- mcols(distanceToNearest(myGR,gtf[gtf$type=="start_codon"]))$distance
  return(myGR)
}

oneHotFeatures <- function(myGR,y){
  loc_factor <- factor(y)

  onehot_mat <- model.matrix(~ loc_factor - 1)
  
  onehot_df <- as.data.frame(onehot_mat)
  colnames(onehot_df) <- levels(loc_factor)
  
  mcols(myGR) <- cbind(mcols(myGR),onehot_df)
  return(myGR)
}

match_background_unique <- function(X,y,ids, strict = FALSE, k = 10, seed = 123) {
  set.seed(seed)
  X <- apply(X,2,scale)
  cases <- X[y==1,]
  cases_ids <- ids[y==1]
  
  controls <- X[y==0,]
  controls_ids <- ids[y==0]
  
  knn_res <- FNN::get.knnx(data = controls, query = cases, k = k)
  ids_return <- (unique(as.vector(knn_res$nn.index)))
  return(controls_ids[ids_return])
}
