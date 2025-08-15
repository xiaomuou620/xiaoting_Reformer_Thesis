### LOAD LIBRARIES (install if necessary) ###
.libPaths(new = c("/maps/projects/renlab/apps/R_packages/R_4.4.0/",
                  "/opt/software/R/4.4.0/lib64/R/library"))

library(GenomicFeatures)
library(GenomicRanges)
library(VariantAnnotation)
library(tidyverse)
library(rtracklayer)
library(BSgenome.Athaliana.TAIR.TAIR9)
library(broom)
library(gridExtra)

####################################################################################
####################################################################################

##I'll put this on the github
gtf <- rtracklayer::import("/maps/projects/renlab/people/nkz325/Thesis_RBP/00_FILES/Araport11_GFF3_genes_transposons.201606.gtf")

source("transcript_matching_functions.R")
####################################################################################
####################################################################################
txdb <- makeTxDbFromGFF(gtf, format = "gtf")


# Obtain transcript parts (5UTR, CDS, 3UTR and Introns)
cds_by_tx     <- GenomicFeatures::cdsBy(txdb, by="tx", use.names=TRUE)
threeUTRs_by_tx <- GenomicFeatures::threeUTRsByTranscript(txdb, use.names=TRUE)
fiveUTRs_by_tx  <- GenomicFeatures::fiveUTRsByTranscript(txdb, use.names=TRUE)
introns_by_tx <- intronsByTranscript(txdb, use.names=TRUE)

# Here I calculate the feature coverage per transcript, for each part. This is because I need this information
# later (for calculating the proportion of the way through the feature a position falls)
intron_sums <- lapply(introns_by_tx,function(x) sum(width(x)))
intron_sums <- unlist(intron_sums)

cds_sums <- lapply(cds_by_tx,function(x) sum(width(x)))
cds_sums <- unlist(cds_sums)

threeUTR_sums <- lapply(threeUTRs_by_tx,function(x) sum(width(x)))
threeUTR_sums <- unlist(threeUTR_sums)

fiveUTR_sums <- lapply(fiveUTRs_by_tx,function(x) sum(width(x)))
fiveUTR_sums <- unlist(fiveUTR_sums)
####################################################################################
####################################################################################

#just some example positions, those labelled "1" are m6A sites, those labeled "0" are random positions (in my case with the DRACH motif which is the m6A-specific context)
load("~/Dropbox/Projects/m6A_DL_python/example_GRange.Rdat")
testGR$id <- paste0(seqnames(testGR),":",start(testGR),"_",strand(testGR))
names(testGR) <- testGR$id

locGR <- prepareTX(txdb,myGR = testGR,longest = TRUE)
table(locGR$LOCATIONS)


fiveUTRGR <- getFeatureSet("fiveUTR",locGR,fiveUTRs_by_tx,fiveUTR_sums)
cdsGR <- getFeatureSet(feature = "coding",myGR = locGR,tx_object = cds_by_tx,tx_sums = cds_sums)
cdsGR$prop <- cdsGR$prop+1 
threeUTRGR <- getFeatureSet("threeUTR",locGR,threeUTRs_by_tx,threeUTR_sums)
threeUTRGR$prop <- threeUTRGR$prop+2

# NOTE: even though there are 4511 intronic sites in locGR they don't make it to intronGR below
# seems only a problem for introns, and something to look into in terms of improving the code!
intronGR <- getFeatureSet(feature = "intron",myGR = locGR,tx_object = introns_by_tx,tx_sums = intron_sums)

testGR_annotated <- unlist(GRangesList(cdsGR,threeUTRGR,fiveUTRGR,intronGR))

#most positions in the UTRs and coding make it through the annotations stage:
table(testGR_annotated$LOCATIONS)
###################################################
########### ANNOTATE FURTHER ####################
###################################################

testGR_annotated <- addStopStart(testGR_annotated,gtf)
testGR_annotated$LOCATIONS <- as.factor(as.vector(testGR_annotated$LOCATIONS))
testGR_annotated <- oneHotFeatures(testGR_annotated,y = testGR_annotated$LOCATIONS)

tx_lengths <- computTxLengths(txdb)
mcols(testGR_annotated) <- as_tibble(mcols(testGR_annotated)) %>% left_join(tx_lengths,by=c("TXID","tx_name"))

#### this adds an id
testGR_annotated$id <- paste0(seqnames(testGR_annotated),"_",start(testGR_annotated),",",strand(testGR_annotated))

#### this adds the 5 bp context of the position -- very useful for the m6A project, not sure about the RBP project
testGR_annotated$motif <- as.vector(getSeq(BSgenome.Hsapiens.UCSC.hg38,testGR_annotated+2))

###################################################
########### COMPUTE BG ############################
###################################################

#Prepares data for PCA and runs PCA, then extracts the first 5 principle components (5 looked a good fit for me, but could be different for other applications)

#select features and bin some of them into quantiles (the quantile definitions can be changed, I just played around with the current version of the function "bin_by_quantile")
features_use <- c("tx_len","utr3_len","utr5_len","nexon","feature_width","stop_dist","start_dist","dist_from_end","dist_from_start","coding","fiveUTR","threeUTR","prop")
feature_annos <- mcols(testGR_annotated)[,features_use[features_use %in% colnames(mcols(testGR_annotated))]]
features_bin <- c("tx_len","utr3_len","utr5_len","nexon","exon_rank","feature_width","stop_dist","start_dist","dist_from_end","dist_from_start")
feature_annos[,intersect(colnames(feature_annos),features_bin)] <- apply(feature_annos[,intersect(colnames(feature_annos),features_bin)],2,bin_by_quantile)
head(feature_annos)

table(feature_annos$tx_len)

PCA_data <- as_tibble(feature_annos)
PCA <- PCA_data %>% prcomp(scale=T) %>% augment(mcols(testGR_annotated))

PCA <- PCA %>% mutate(.fittedPC1=scale(.fittedPC1),
                      .fittedPC2=scale(.fittedPC2),
                      .fittedPC3=scale(.fittedPC3),
                      .fittedPC4=scale(.fittedPC4),
                      .fittedPC5=scale(.fittedPC5))

TP1 <- PCA %>%
  #sample() %>%
  arrange(label) %>%
  ggplot(aes(.fittedPC1,.fittedPC2,colour=as.factor(LOCATIONS))) + geom_point(size=.1)

TP2 <- PCA %>%
  #sample() %>%
  arrange(label) %>%
  ggplot(aes(.fittedPC1,.fittedPC2,colour=as.factor(label))) + geom_point(size=.1)

grid.arrange(TP1,TP2)
##############################################################################################

testGR_annotated_with_backgrounds <- testGR_annotated

####################MATCHED LOCATION##########################################################

X <- data.frame(PCA$.fittedPC1,PCA$.fittedPC2,PCA$.fittedPC3,PCA$.fittedPC4,PCA$.fittedPC5)

background_ids <- match_background_unique(X = X,y=PCA$label,ids=PCA$id, strict = FALSE, k = 2)
#asking for 2 nearest neighbours (k=2) usually gives too many, so let's downsample:
background_ids <- sort(sample(background_ids,sum(PCA$label),replace=FALSE))

# positives
testGR_annotated_with_backgrounds$m6a_matched_fg <- testGR_annotated_with_backgrounds$label

# match background set
testGR_annotated_with_backgrounds$m6a_matched_bg <- as.numeric(testGR_annotated_with_backgrounds$id %in% background_ids)

# to create a random background I can sample from the same genes as the positives
transcripts_use <- testGR_annotated_with_backgrounds[testGR_annotated_with_backgrounds$label==1]$tx_name
tosampleGR <- testGR_annotated_with_backgrounds[testGR_annotated_with_backgrounds$label==0 & (testGR_annotated_with_backgrounds$tx_name %in% transcripts_use)]
background_random_ids <- sort(sample(tosampleGR$id, sum(PCA$label),replace=FALSE))

testGR_annotated_with_backgrounds$m6a_random_bg <- as.numeric(testGR_annotated_with_backgrounds$id %in% background_random_ids)

######### I can have a quick look at the distribution in the 5UTR/CDS/3UTR
par(mfrow=c(1,3))
tp <- testGR_annotated_with_backgrounds[which(testGR_annotated_with_backgrounds$m6a_matched_fg==1)]$prop
plot(density(tp))
abline(v=c(1,2))

#matched background should have a similar shape
tp <- testGR_annotated_with_backgrounds[which(testGR_annotated_with_backgrounds$m6a_matched_bg==1)]$prop
plot(density(tp))
abline(v=c(1,2))

#random positions 
tp <- testGR_annotated_with_backgrounds[which(testGR_annotated_with_backgrounds$m6a_random_bg==1)]$prop
plot(density(tp))
abline(v=c(1,2))


#####check new background set on PCA
TP3 <- PCA %>% mutate(in_bg = id %in% background_ids) %>%
  #sample() %>%
  arrange(in_bg) %>%
  ggplot(aes(.fittedPC1,.fittedPC2,colour=as.factor(in_bg))) + geom_point(size=.1)

grid.arrange(TP1,TP2,TP3,ncol=3)

