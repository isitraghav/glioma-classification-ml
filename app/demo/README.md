# Mock expression files for the demo app

Synthetic tumours, not real patients. Each row averages 3 random **training** patients of one class (log2 scale), adds small Gaussian noise, and is converted back to raw RSEM values, so test patients are never reused. Upload either file in the app's *Upload expression file* tab.

| File | Sample | Class it was built from |
|---|---|---|
| mock_expression_1.csv | MOCK-1-01 | IDHwt |
| mock_expression_1.csv | MOCK-1-02 | IDHmut-non-codel |
| mock_expression_1.csv | MOCK-1-03 | IDHmut-codel |
| mock_expression_2.csv | MOCK-2-01 | IDHmut-codel |
| mock_expression_2.csv | MOCK-2-02 | IDHwt |
| mock_expression_2.csv | MOCK-2-03 | IDHmut-non-codel |
