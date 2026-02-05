t-test assignment
Each of the three datasets below should be analyzed with a t-test in R. 
- Use a two-tailed test (the default) for all three tests. 
- Provide a brief write-up, besides the R output, of the analysis you did for each dataset that answers the following questions:
• What hypotheses was tested (written in words and with “H0: “ notation),
• What was the specific method used to analyze the data (e.g. two sample t-test variance equal)?
• What was the conclusion?
• What is the p-value?
• What is(are) the sample mean(s)?
Include your full R program.
1. The file “ttest newproduct.csv” contains the results of two separate focus groups where two
different groups of random customers were asked to evaluate two products, product A and
product B. The customer ratings were summarized in a composite “Rating” score. Compare
product A and product B to determine if there is a significant difference in their average rating.
2. The file “ttest bp.csv” contains the results of a clinical trial to determine the impact of a new
blood pressure medicine in reducing diastolic blood pressure. Each patient’s blood pressure was
taken before the medicine was administered and six hours after the medicine was administered.
Determine if the medicine significantly impacted blood pressure.
3. The file “ttest act.csv” contains a random sample of ACT scores for students currently attending
a large university. The university claims their average ACT score for students is 24. Determine if
the university’s claim is accurate.
4. A farm and food company is conducting a study to test the impact of a new fertilizer on a certain
variety of kale. In a controlled greenhouse environment, two groups of kale plants will be
randomly selected. One group will be given the currently used fertilizer; the other group the
new fertilizer. The company wants to detect a difference of at least 2-inches in mean plant
height at harvest age. The standard deviation of mature kale plant height is historically known
to be 2.43 inches and can be assumed to be equal between the two groups. What sample size
should be used for each group in this experiment?


#----------------------------
# %% files
ttest_act<-read.csv("C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Classes Spring 2026/BDA Spring 26/Datasets/ttest act.csv")
ttest_bp<-read.csv("C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Classes Spring 2026/BDA Spring 26/Datasets/ttest bp.csv")
ttest_newprod<-read.csv("C:/Users/tyler/OneDrive/Documents/GitHub/work_to_show/Classes Spring 2026/BDA Spring 26/Datasets/ttest newproduct.csv")

# %% View Data Sets
View(ttest_act)
colnames(ttest_act) # "student"  "ACTScore"
View(ttest_bp)
colnames(ttest_bp) # "patient" "before"  "after"
View(ttest_newprod)
colnames(ttest_newprod) # "CustomerID" "Product"    "Rating"

# %% Use a two-tailed test (the default) for all three tests
t.test(ttest_act$ACTScore,mu=6)
t.test(ttest_bp$before,mu=6)
t.test(ttest_bp$after,mu=6)
t.test(ttest_newprod$Rating,mu=6)

# %% Hypothesis test for every data set
