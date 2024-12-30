import statsmodels.formula.api as smf
from scipy.stats import norm
import pandas as pd

# Load your data
file_path = 'data/processed/long_format_results_with_split.csv'  # Replace with actual path
data = pd.read_csv(file_path)

# Define hypotheses and subsets
hypotheses = {
    "Hypothesis 1": {"condition": (data["Sentiment"] == "NG") & (data["Pitch"] == "high"), "description": "High pitch and negative sentiment conveys happiness", "direction": "greater"},
    "Hypothesis 2": {"condition": (data["Sentiment"] == "P") & (data["Pitch"] == "low"), "description": "Low pitch and positive sentiment conveys sadness", "direction": "less"},
    "Hypothesis 3": {"condition": (data["Sentiment"] == "NT") & (data["Pitch"] == "high"), "description": "High pitch and neutral sentiment conveys happiness", "direction": "greater"},
    "Hypothesis 4": {"condition": (data["Sentiment"] == "NT") & (data["Pitch"] == "low"), "description": "Low pitch and neutral sentiment conveys sadness", "direction": "less"},
}

# Initialize results dictionary
results = {}

# Iterate through hypotheses and fit models
for hypothesis, details in hypotheses.items():
    subset_data = data[details["condition"]]
    if not subset_data.empty:
        # Fit the linear mixed model
        model = smf.mixedlm("Rating ~ 1", data=subset_data, groups=subset_data["Username"])
        result = model.fit()
        
        # Extract necessary values
        intercept = result.params["Intercept"]
        stderr = result.bse["Intercept"]
        test_value = 4  # Hypothesized test value (can be parameterized)
        direction = details["direction"]
        
        # Calculate z-score and p-value
        z_score = (intercept - test_value) / stderr
        if direction == "greater":
            p_value = 1 - norm.cdf(z_score)  # One-tailed test for greater
        else:  # "less"
            p_value = norm.cdf(z_score)  # One-tailed test for less
        
        # Store results
        results[hypothesis] = {
            "description": details["description"],
            "intercept": intercept,
            "stderr": stderr,
            "z_score": z_score,
            "p_value": p_value,
            "mean_rating": subset_data["Rating"].mean()
        }
    else:
        results[hypothesis] = {"description": details["description"], "error": "No data available for this hypothesis"}

# Display results
for hypothesis, output in results.items():
    print(f"{hypothesis}: {output['description']}")
    if "error" in output:
        print(output["error"])
    else:
        print(f"  Intercept: {output['intercept']:.3f}")
        print(f"  Std. Err: {output['stderr']:.3f}")
        print(f"  Z-Score: {output['z_score']:.3f}")
        print(f"  P-Value: {output['p_value']:.3e}")
        print(f"  Mean Rating: {output['mean_rating']:.3f}\n")
