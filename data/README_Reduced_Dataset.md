# Wildfire Dataset - Reduced Version

## Overview
This is a reduced version of the original Wildfire Dataset, optimized for project delivery while maintaining representative data quality.

## Dataset Details

### Original Dataset
- **Source:** https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset
- **Size**: ~1 GB
- **Rows**: 9,509,925
- **Date Range**: 2013-12-31 to 2025-04-13
- **Overall Fire Rate**: 5.28%

### Reduced Dataset (Wildfire_Dataset_Reduced.csv)
- **Size**: ~29 MB
- **Rows**: 249,504
- **Date Range**: 2024-04-13 to 2025-04-13 (last 12 months)
- **Fire Rate**: 19.39%
- **Reduction**: 97.4% fewer rows
- **Unique Locations**: 3,653
- **Total Fires**: 48,379

## Features
The reduced dataset maintains all 19 original columns:
- **Location**: `latitude`, `longitude`
- **Time**: `datetime`
- **Target**: `Wildfire` (Yes/No)
- **Weather Features**: `pr`, `rmax`, `rmin`, `sph`, `srad`, `tmmn`, `tmmx`, `vs`
- **Fire Indices**: `bi`, `fm100`, `fm1000`, `erc`, `etr`, `pet`, `vpd`

## Why This Time Window?
The last 12 months (April 2024 - April 2025) was selected because:
1. **Most Recent Data**: Closest to present as requested
2. **All Locations Available**: Includes 3,653 unique geographic locations
3. **Rich in Positive Cases**: Contains 48,379 wildfire events (19.39% fire rate)
4. **Good Balance**: Sufficient data for training/testing while maintaining manageable file size
5. **Realistic Distribution**: Represents actual wildfire patterns in recent years

## Usage
This reduced dataset can be used as a drop-in replacement for the full dataset in most scenarios:
- Training and testing machine learning models
- Data analysis and visualization
- Model evaluation and validation
- Demonstrations and presentations

## Note
The higher fire rate (19.39% vs 5.28% overall) reflects the recent increase in wildfire events, particularly in 2024-2025, making this dataset especially relevant for current wildfire prediction models.

## Creation Details
- **Created**: December 14, 2025
- **Script**: `create_reduced_dataset.py`
- **Method**: Filtered full dataset to include only data from the last 12 months
