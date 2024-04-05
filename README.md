# Protein_Sequence_Generation

**Data Handling and Analysis**

**Dataset Overview: (Protein Sequence Dataset.csv)**
The dataset consists of 90,000 rows containing protein sequences along with associated metadata such as EC number, UniProt IDs, names, functions and sequences.

**Preprocessing Steps: (spliting_encoding_vectorization.ipynb)**

1. **Splitting and Encoding:**
   - The 'name' column was split into 'gene' and 'species' using '_' as a delimiter.
   - Label encoding was performed on 'EC_number' and 'species' columns.
   - Sequence lengths were calculated and added as a new feature.

2. **Outlier Detection:**
   - A box plot was generated to visualize the distribution of sequence lengths.
   - Outliers were identified and addressed by filtering out sequences with length 0.

3. **Data Visualization:**
   - Plots were created to visualize sequence length distribution for specific EC numbers and overall.
   - A box plot was generated to illustrate the distribution of sequence lengths across different EC number encodings.

**Key Observations:**
- The dataset comprises sequences with varying lengths, with outliers detected and handled accordingly.
- Sequence lengths exhibit variation across different EC number encodings, indicating potential differences in protein families or functions.

**Data Cleaning and Vectorization: (cleaning_vectorization.ipynb)**

1. **Data Cleaning:**
   - Invalid sequences containing non-standard amino acid codes were filtered out.
   
2. **Sequence Vectorization:**
   - One-hot encoding was applied to represent each amino acid in the sequences numerically.
   - The resulting encoded sequences were added as a new feature ('sequence_encoded').

**Conclusion:**
The dataset underwent preprocessing to handle outliers, encode categorical variables, clean invalid sequences, and convert protein sequences into numerical representations suitable for machine learning models. These steps ensure data quality and prepare the dataset for subsequent modeling and analysis.

For detailed code implementation, please refer to the provided Jupyter Notebook files: 'splitting_encoding_outlierdetection.ipynb' and 'cleaning_vectorization.ipynb'.
