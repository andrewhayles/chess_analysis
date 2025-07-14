# tukey_checker.py
# A quick script to verify the correct function name for Tukey's biweight
# in your installed version of the statsmodels library.

import statsmodels.api as sm
import sys

print(f"Python version: {sys.version}")
print(f"Statsmodels version: {sm.__version__}")
print("-" * 30)

try:
    # Attempt to access the function name that caused the error before
    # This will likely fail, which is expected.
    tukey_hampel = sm.robust.norms.TukeyHampel
    print("❌ Found 'TukeyHampel'. This is the old name and likely not what the main script expects.")

except AttributeError:
    print("✅ As expected, 'TukeyHampel' was NOT found.")

try:
    # Attempt to access the corrected function name
    tukey_biweight = sm.robust.norms.TukeyBiweight
    print("\n✅ SUCCESS: Found 'TukeyBiweight'!")
    print("   The main analysis script should now run without the previous error.")

except AttributeError:
    print("\n❌ ERROR: Could not find 'TukeyBiweight'.")
    print("   There might be an issue with your statsmodels installation or version.")
    print("   Please check the library documentation for the version you have installed.")

