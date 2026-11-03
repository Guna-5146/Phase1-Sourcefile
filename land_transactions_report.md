# Generator report

Records: 80,000  Columns: 34
Villages: 1078  Taluks: 27
Fraudulent (is_fraud = 1): 5,483 (6.9%)
Label noise: 1%  Hard negatives: 3% of legitimate

## Table 2 shares (% of generated fraud)

fraud_type
undervaluation       45.0
duplicate_sale       15.0
misclassification    14.0
forged_poa           14.0
flipping             12.0

## Table 2 rules

- Undervaluation: declared price = U(0.4, 0.8) x guideline value
- Misclassification: paid sale (bank/cash consideration, non-relative buyer) registered as settlement or gift, duty 1% instead of 7%
- Duplicate sale: same survey number and seller sold to a new buyer within 90 days
- Forged PoA: PoA sale by a non-relative holder; seller absent from revenue records in 75% of cases
- Flipping: resold within 180 days at (1.8, 3.5) x previous price
- Legitimate price ratio: 1 + |N(0.05, 0.15)|, with 4% of legitimate sales at U(0.85, 1.0)
