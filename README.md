# gep-cfd-stability-predictor
This project develops a machine learning framework to predict the stability of CFD simulations directly from symbolic turbulence model equations generated using Gene Expression Programming (GEP).

The approach avoids expensive and time-consuming CFD runs by learning patterns between algebraic model structures and solver behavior.

Each turbulence model is represented as a combination of four symbolic expressions corresponding to tensor basis functions. These expressions are tokenized and processed using a Transformer Encoder, which captures relationships between operators, invariants (I1, I2), and constants.

The model performs:
• Binary classification: stable vs unstable simulations  
• Regression: prediction of pressure initial residual (p_initial_residual)

Key Features:
• Symbolic expression learning using Transformers  
• Log-scale regression for highly varying residual values  
• Automated dataset generation from CFD logs  
• Early instability detection for GEP-based RANS models  

Applications:
• Accelerating turbulence model discovery  
• Reducing computational cost of CFD simulations  
• Data-driven analysis of RANS model stability  

Results:
The model achieves ~70–75% accuracy in predicting solver stability, demonstrating the feasibility of learning CFD behavior directly from symbolic equations.
